import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from src.inverse.curve_encoder import CurveEncoder
from src.inverse.experiment_utils import (
    build_target_feature,
    compute_joint_metrics_batch,
    load_or_create_fixed_split,
    select_hard_test_indices,
    stack_target_features,
)
from src.inverse.gnn_policy import GNNPolicy, policy_load_incompatibilities
from src.inverse.mcts import MCTS
from src.inverse.rl_agent import PPOAgent
from src.inverse.rl_env import MechanismEnv, apply_j_operator, load_frozen_surrogate, validate_graph_structure
from src.inverse.train_il import ensure_expert_paths


ALL_METHODS = ['retrieval', 'il_greedy', 'il_mcts', 'rl_greedy', 'rl_mcts']


def _load_inverse_bundle(ckpt_path: str, cfg: dict, device):
    if not os.path.exists(ckpt_path):
        return None

    curve_cfg = cfg.get('curve_encoder', {})
    curve_encoder = CurveEncoder(
        input_dim=curve_cfg.get('input_dim', 800),
        hidden_dims=curve_cfg.get('hidden_dims', [512, 256]),
        latent_dim=curve_cfg.get('latent_dim', 128),
    ).to(device)
    policy = GNNPolicy(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_state = policy.load_state_dict(ckpt['policy'], strict=False)
    missing_keys, unexpected_keys = policy_load_incompatibilities(policy_state)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint '{ckpt_path}' is incompatible with current policy. "
            f"Missing={missing_keys}, unexpected={unexpected_keys}"
        )
    curve_encoder.load_state_dict(ckpt['curve_encoder'], strict=False)
    policy.eval()
    curve_encoder.eval()
    agent = PPOAgent(policy, curve_encoder, cfg, device)
    return {
        'policy': policy,
        'curve_encoder': curve_encoder,
        'agent': agent,
    }


def _encode_target(curve_encoder, sample: dict, device):
    with torch.no_grad():
        return curve_encoder(
            sample['y_foot'].unsqueeze(0).to(device),
            sample['y_knee'].unsqueeze(0).to(device),
            sample['y_ankle'].unsqueeze(0).to(device),
        )


def _apply_sample_action(sample: dict):
    return apply_j_operator(
        sample['base_data'],
        int(sample['action_topo'][0].item()),
        int(sample['action_topo'][1].item()),
        int(sample['action_topo'][2].item()),
        sample['action_geo'][:2].detach().cpu().numpy(),
        sample['action_geo'][2:].detach().cpu().numpy(),
    )


def _generate_greedy(bundle: dict, sample: dict, cfg: dict, device):
    z_c = _encode_target(bundle['curve_encoder'], sample, device)
    actions, _, _ = bundle['agent'].batch_select_actions([sample['base_data']], z_c, deterministic=True)
    action = actions[0]
    if action is None:
        return None
    return apply_j_operator(
        sample['base_data'],
        action['u'],
        action['v'],
        action['w'],
        action['n1'],
        action['n2'],
    )


def _generate_mcts(bundle: dict, surrogate, sample: dict, cfg: dict, device):
    z_c = _encode_target(bundle['curve_encoder'], sample, device)
    env = MechanismEnv(
        surrogate,
        cfg.get('reward', {}),
        max_steps=cfg.get('rl_training', {}).get('steps_per_episode', 1),
        device=device,
        constraint_cfg=cfg.get('constraints', {}),
    )
    target = {
        'y_foot': sample['y_foot'],
        'y_knee': sample['y_knee'],
        'y_ankle': sample['y_ankle'],
    }
    env.reset(target, sample['base_data'], z_c)
    mcts = MCTS(bundle['policy'], surrogate, env, cfg, device)
    action, _ = mcts.search(sample['base_data'], z_c, target)
    if action is None:
        return None
    return apply_j_operator(
        sample['base_data'],
        action['u'],
        action['v'],
        action['w'],
        action['n1'],
        action['n2'],
    )


def _predict_metrics(graph, sample: dict, surrogate, cfg: dict, device):
    penalty_metric = 1e6
    base_metrics = {
        'joint_score': penalty_metric,
        'foot_score': penalty_metric,
        'foot_nrmse': penalty_metric,
        'foot_chamfer_norm': penalty_metric,
        'knee_nrmse': penalty_metric,
        'ankle_nrmse': penalty_metric,
        'smoothness': penalty_metric,
        'success': 0.0,
        'valid': 0.0,
    }
    if graph is None:
        return base_metrics

    is_valid, _ = validate_graph_structure(graph, cfg.get('constraints', {}))
    if not is_valid:
        return base_metrics

    try:
        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch)
        metrics = compute_joint_metrics_batch(
            pred_foot.cpu(),
            pred_knee.cpu(),
            pred_ankle.cpu(),
            {
                'y_foot': sample['y_foot'],
                'y_knee': sample['y_knee'],
                'y_ankle': sample['y_ankle'],
            },
            cfg.get('reward', {}),
        )
        out = {
            'joint_score': float(metrics['joint_score'][0].item()),
            'foot_score': float(metrics['foot_score'][0].item()),
            'foot_nrmse': float(metrics['foot_nrmse'][0].item()),
            'foot_chamfer_norm': float(metrics['foot_chamfer_norm'][0].item()),
            'knee_nrmse': float(metrics['knee_nrmse'][0].item()),
            'ankle_nrmse': float(metrics['ankle_nrmse'][0].item()),
            'smoothness': float(metrics['smoothness'][0].item()),
            'success': 1.0,
            'valid': 1.0,
        }
        return out
    except Exception:
        return base_metrics


def _aggregate_metrics(per_sample_metrics, elapsed_times):
    if not per_sample_metrics:
        return {}
    keys = sorted(per_sample_metrics[0].keys())
    summary = {key: float(np.mean([metrics[key] for metrics in per_sample_metrics])) for key in keys}
    summary['avg_inference_sec'] = float(np.mean(elapsed_times)) if elapsed_times else 0.0
    return summary


def _evaluate_method(method_name: str, samples, surrogate, cfg, device, train_samples=None, train_features=None, bundle=None):
    per_sample_metrics = []
    elapsed_times = []

    for sample in samples:
        start_t = time.time()
        if method_name == 'retrieval':
            target_feature = build_target_feature(sample, cfg.get('experiment', {}).get('retrieval_weights', {}))
            distances = torch.norm(train_features - target_feature.unsqueeze(0), dim=1)
            nearest_idx = int(torch.argmin(distances).item())
            graph = _apply_sample_action(train_samples[nearest_idx])
        elif method_name == 'il_greedy':
            graph = _generate_greedy(bundle, sample, cfg, device)
        elif method_name == 'il_mcts':
            graph = _generate_mcts(bundle, surrogate, sample, cfg, device)
        elif method_name == 'rl_greedy':
            graph = _generate_greedy(bundle, sample, cfg, device)
        elif method_name == 'rl_mcts':
            graph = _generate_mcts(bundle, surrogate, sample, cfg, device)
        else:
            raise ValueError(f'Unsupported method: {method_name}')

        elapsed_times.append(time.time() - start_t)
        per_sample_metrics.append(_predict_metrics(graph, sample, surrogate, cfg, device))

    return {
        'summary': _aggregate_metrics(per_sample_metrics, elapsed_times),
        'num_samples': len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description='Run GraphMetaMat-LINKS joint-curve experiment.')
    parser.add_argument('--config', type=str, default='src/config_inverse.yaml')
    parser.add_argument('--methods', type=str, default='all', help='Comma-separated subset of methods or "all".')
    parser.add_argument('--num_samples', type=int, default=None, help='Optional cap for each evaluated split.')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Experiment] Using device: {device}')

    expert_paths = ensure_expert_paths(
        pkl_path=cfg['paths']['pkl_dataset'],
        output_path=cfg['paths']['il_dataset_output'],
        use_cached=True,
    )
    split = load_or_create_fixed_split(
        num_samples=len(expert_paths),
        val_ratio=cfg['il_training'].get('val_ratio', 0.1),
        test_ratio=cfg['il_training'].get('test_ratio', 0.1),
        split_seed=cfg['il_training'].get('split_seed', 42),
        split_path=cfg['paths'].get('il_split_output', 'artifacts/il_split_indices.pt'),
    )

    hard_indices = select_hard_test_indices(
        expert_paths,
        split,
        hard_fraction=cfg.get('experiment', {}).get('hard_test_fraction', 0.25),
        min_hard_samples=cfg.get('experiment', {}).get('hard_test_min_samples', 256),
    )

    train_samples = [expert_paths[idx] for idx in split['train_indices']]
    test_samples = [expert_paths[idx] for idx in split['test_indices']]
    hard_test_samples = [expert_paths[idx] for idx in hard_indices]

    sample_cap = args.num_samples if args.num_samples is not None else cfg.get('experiment', {}).get('eval_num_samples')
    if sample_cap:
        test_samples = test_samples[:sample_cap]
        hard_test_samples = hard_test_samples[:sample_cap]

    surrogate, _ = load_frozen_surrogate(
        cfg['paths']['forward_model'],
        cfg['paths']['config_forward'],
        device,
    )

    methods = ALL_METHODS if args.methods == 'all' else [item.strip() for item in args.methods.split(',') if item.strip()]
    method_bundles = {}
    if any(method.startswith('il_') for method in methods):
        bundle = _load_inverse_bundle(cfg['paths']['il_model_output'], cfg, device)
        if bundle is not None:
            method_bundles['il'] = bundle
    if any(method.startswith('rl_') for method in methods):
        bundle = _load_inverse_bundle(cfg['paths']['rl_model_output'], cfg, device)
        if bundle is not None:
            method_bundles['rl'] = bundle

    retrieval_weights = cfg.get('experiment', {}).get('retrieval_weights', {})
    train_features = stack_target_features(train_samples, weights=retrieval_weights)

    report = {
        'config': args.config,
        'split': {
            'train_size': len(split['train_indices']),
            'val_size': len(split['val_indices']),
            'test_size': len(split['test_indices']),
            'hard_test_size': len(hard_indices),
            'sample_cap': sample_cap,
        },
        'methods': {},
    }

    for method in methods:
        print(f'[Experiment] Evaluating {method} ...')
        if method == 'retrieval':
            bundle = None
        elif method.startswith('il_'):
            bundle = method_bundles.get('il')
        else:
            bundle = method_bundles.get('rl')

        if method != 'retrieval' and bundle is None:
            print(f'[Experiment] Skipping {method}: checkpoint missing.')
            continue

        report['methods'][method] = {
            'test': _evaluate_method(
                method,
                test_samples,
                surrogate,
                cfg,
                device,
                train_samples=train_samples,
                train_features=train_features,
                bundle=bundle,
            ),
            'hard_test': _evaluate_method(
                method,
                hard_test_samples,
                surrogate,
                cfg,
                device,
                train_samples=train_samples,
                train_features=train_features,
                bundle=bundle,
            ),
        }

    if 'il_greedy' in report['methods'] and 'il_mcts' in report['methods']:
        greedy = report['methods']['il_greedy']['test']['summary']['joint_score']
        mcts = report['methods']['il_mcts']['test']['summary']['joint_score']
        report.setdefault('comparisons', {})['il_mcts_gain_vs_greedy'] = float(greedy - mcts)
    if 'rl_greedy' in report['methods'] and 'rl_mcts' in report['methods']:
        greedy = report['methods']['rl_greedy']['test']['summary']['joint_score']
        mcts = report['methods']['rl_mcts']['test']['summary']['joint_score']
        report.setdefault('comparisons', {})['rl_mcts_gain_vs_greedy'] = float(greedy - mcts)

    report_path = Path(cfg.get('experiment', {}).get('eval_report_output', 'artifacts/experiment_report.json'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'[Experiment] Report saved to {report_path}')


if __name__ == '__main__':
    main()
