import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from sim import find_path, get_G, solve_rev_vectorized, draw_mechanism
except ImportError:
    # Try appending parent dir if run from subdir
    sys.path.insert(0, os.path.dirname(os.getcwd()))
    from sim import find_path, get_G, solve_rev_vectorized, draw_mechanism

def create_random_base_4bar():
    """
    Creates a random 4-bar mechanism.
    """
    num_nodes = 4
    A = np.zeros((num_nodes, num_nodes))
    
    # Topology: Fixed(0,1), Crank(0->2), Coupler(2->3), Rocker(3->1)
    links = [(0, 2), (2, 3), (3, 1), (0, 1)]
    for i, j in links:
        A[i, j] = 1
        A[j, i] = 1

    # Randomize dimensions
    # Ground length (dist 0-1): Fixed at L_g ~ 3.0 +/- 1.0
    L_g = np.random.uniform(2.0, 4.0)
    
    # Crank length (dist 0-2): Small enough to rotate? 
    # Or just random. Let's say 0.5 to 1.5
    L_c = np.random.uniform(0.5, 1.5)
    
    # Rocker length (dist 1-3): 1.5 to 3.5
    L_r = np.random.uniform(1.5, 3.5)
    
    # Coupler length (dist 2-3): 2.0 to 4.0
    L_coupler = np.random.uniform(2.0, 4.0)
    
    # Build Coordinates
    # Node 0 at (0,0)
    x0 = np.zeros((4, 2))
    x0[0] = [0.0, 0.0]
    
    # Node 1 at (L_g, 0) (can rotate frame later if needed, but standard frame is fine)
    # Randomize orientation of ground? No, keep ground horizontal for simplicity, 
    # mechanism orientation is relative.
    x0[1] = [L_g, 0.0]
    
    # Node 2 (Crank tip) at initial angle
    theta_c = np.random.uniform(0, 2*np.pi)
    x0[2] = [L_c * np.cos(theta_c), L_c * np.sin(theta_c)]
    
    # Node 3 (Rocker/Coupler joint)
    # Intersection of circle from 2 (radius L_coupler) and from 1 (radius L_r)
    # .. math ..
    d = np.linalg.norm(x0[2] - x0[1])
    
    # Invalid geometry check
    if d > L_coupler + L_r or d < abs(L_coupler - L_r) or d < 1e-9:
        # Fallback to a valid default if random params fail
        return create_random_base_4bar() # Retry recursively
        
    a = (L_coupler**2 - L_r**2 + d**2) / (2*d)
    h = np.sqrt(max(0, L_coupler**2 - a**2))
    
    p2 = x0[1] - x0[2] 
    x3_base = x0[2][0] + a * p2[0] / d
    y3_base = x0[2][1] + a * p2[1] / d
    
    # Choose on of two solutions
    sign = np.random.choice([-1, 1])
    x3_x = x3_base + sign * h * p2[1] / d
    x3_y = y3_base - sign * h * p2[0] / d
    
    x0[3] = [x3_x, x3_y]
    
    node_types = np.array([1, 1, 0, 0]) 
    
    return A, x0, node_types

def add_dyad(A, x0, node_types):
    """
    Adds a Dyad with randomized connection references.
    """
    current_n = A.shape[0]
    n1 = current_n
    n2 = current_n + 1
    
    new_n = current_n + 2
    A_new = np.zeros((new_n, new_n))
    A_new[:current_n, :current_n] = A
    
    x0_new = np.zeros((new_n, 2))
    x0_new[:current_n] = x0
    
    node_types_new = np.zeros(new_n)
    node_types_new[:current_n] = node_types
    node_types_new[n1] = 0
    node_types_new[n2] = 0
    
    # Stephenson III / Watt Topology Extension
    # 1. Select u, v from Moving nodes
    moving_indices = np.where(node_types == 0)[0]
    if len(moving_indices) < 2:
        return None, None, None, None
        
    u = np.random.choice(moving_indices)
    possible_v = moving_indices[moving_indices != u]
    v = np.random.choice(possible_v)
    
    # 2. Select w 
    all_indices = np.arange(current_n)
    possible_w = np.where(node_types == 1)[0] # Fixed
    if len(possible_w) == 0: possible_w = all_indices
    # Allow w to be moving? User asked for "connection method" changes.
    # If we attach to moving, it's still a 6-bar but different loop type (Watt II?).
    # Strategy B implies specific Hip-Knee-Ankle chain.
    # Hip->Knee is Base->Dyad. If w is moving, Knee path is different.
    # Let's keep w fixed (Ground) to maintain "Leg" stability, but allow randomization if selected.
    # Let's stick to Ground for stability but randomize WHOLE geometry.
    w = np.random.choice(possible_w)
    
    # Initialize Positions with HIGH VARIANCE
    # n1: Triangulate from u, v
    len_u_n1 = np.random.uniform(0.5, 5.0) # Increased range
    len_v_n1 = np.random.uniform(0.5, 5.0)
    
    dist_uv = np.linalg.norm(x0[u] - x0[v])
    
    # Sanity fix
    if dist_uv > len_u_n1 + len_v_n1 or dist_uv < abs(len_u_n1 - len_v_n1):
         len_u_n1 = dist_uv * 0.6 + np.random.uniform(0.1, 1.0)
         len_v_n1 = dist_uv * 0.6 + np.random.uniform(0.1, 1.0)
    
    # Calculate n1
    d = dist_uv
    if d < 1e-9:
        return None, None, None, None
    a = (len_u_n1**2 - len_v_n1**2 + d**2) / (2*d)
    h = np.sqrt(max(0, len_u_n1**2 - a**2))
    
    p2 = x0[v] - x0[u]
    x2 = x0[u][0] + a * p2[0] / d
    y2 = x0[u][1] + a * p2[1] / d
    
    # Random Solution
    sign = np.random.choice([-1, 1])
    x3_1 = x2 + sign * h * p2[1] / d
    y3_1 = y2 - sign * h * p2[0] / d
    
    x0_new[n1] = [x3_1, y3_1]
    
    # n2: Connects n1 (known) and w (known).
    len_n1_n2 = np.random.uniform(0.5, 5.0)
    len_w_n2 = np.random.uniform(0.5, 5.0)
    
    dist_n1w = np.linalg.norm(x0_new[n1] - x0[w])
    if dist_n1w > len_n1_n2 + len_w_n2 or dist_n1w < abs(len_n1_n2 - len_w_n2):
        len_n1_n2 = dist_n1w * 0.6 + np.random.uniform(0.1, 1.0)
        len_w_n2 = dist_n1w * 0.6 + np.random.uniform(0.1, 1.0)
        
    d = dist_n1w
    if d < 1e-9:
        return None, None, None, None
    a = (len_n1_n2**2 - len_w_n2**2 + d**2) / (2*d)
    h = np.sqrt(max(0, len_n1_n2**2 - a**2))
    
    p2 = x0[w] - x0_new[n1]
    x2 = x0_new[n1][0] + a * p2[0] / d
    y2 = x0_new[n1][1] + a * p2[1] / d
    
    sign = np.random.choice([-1, 1])
    x3_2 = x2 + sign * h * p2[1] / d
    y3_2 = y2 - sign * h * p2[0] / d
    
    x0_new[n2] = [x3_2, y3_2]

    edges = [(u, n1), (v, n1), (n1, n2), (w, n2)]
    for i, j in edges:
        A_new[i, j] = 1
        A_new[j, i] = 1
        
    return A_new, x0_new, node_types_new, {'u': u, 'v': v, 'w': w, 'n1': n1, 'n2': n2}

def solve_kinematics(A, x0, node_types):
    motor = [0, 2] 
    fixed_nodes = np.where(node_types)[0]
    
    try:
        path, path_found = find_path(A, motor=motor, fixed_nodes=fixed_nodes)
        if not path_found or len(path) == 0:
            return None
            
        G = get_G(x0)
        thetas = np.linspace(0, 2*np.pi, 200)
        
        solve_res = solve_rev_vectorized(path, x0, G, motor, fixed_nodes, thetas)
        x_sol = solve_res[0]
        valid = solve_res[1]
        
        # Stricter locking check: must be valid for at least 50% of range?
        # Or FULL range as before. Let's stick to full range for high quality.
        if np.sum(valid) != len(thetas):
            return None
            
        return x_sol
        
    except Exception as e:
        if os.environ.get('LINKS_DEBUG_SOLVER', '0') == '1':
            print(f'[solve_kinematics] Solver exception: {e}')
        return None

def identify_strategy_b(A, node_types, x_sol, gen_info):
    hip_idx = 0
    knee_idx = gen_info['u'] 
    ankle_idx = gen_info['n1']
    foot_idx = gen_info['n2']
    return hip_idx, knee_idx, ankle_idx, foot_idx

def analyze_sample(sample_id, A, x0, types, x_sol, gen_info, strategy='B'):
    h, k, a, f = identify_strategy_b(A, types, x_sol, gen_info)
    if f == -1: return None
    
    foot_traj = x_sol[f]
    rom_x = foot_traj[:, 0].max() - foot_traj[:, 0].min()
    rom_y = foot_traj[:, 1].max() - foot_traj[:, 1].min()
    valid_rom = (rom_x > 0.5) and (rom_y > 0.2)
    
    return {
        "id": sample_id,
        "strategy": strategy,
        "hip": h,
        "knee": k,
        "ankle": a,
        "foot": f,
        "valid_rom": valid_rom,
        "x_sol": x_sol
    }

def generate_dataset(strategy_name, n_samples):
    print(f"\nGenerating Diverse Dataset {strategy_name} ({n_samples} samples)...")
    valid_samples = []
    attempts = 0
    max_attempts = n_samples * 100 
    
    while len(valid_samples) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Use RANDOM base
        A, x0, types = create_random_base_4bar()
        
        res = add_dyad(A, x0, types)
        if res[0] is None: continue
        A, x0, types, gen_info = res
        
        x_sol = solve_kinematics(A, x0, types)
        
        if x_sol is not None:
            analysis = analyze_sample(len(valid_samples), A, x0, types, x_sol, gen_info, strategy=strategy_name)
            
            if analysis and analysis['valid_rom']:
                sample_data = {
                    'A': A, 'x0': x0, 'types': types,
                    'analysis': analysis,
                    'gen_info': gen_info,  # 保存 J算子路径，用于 IL 专家路径提取
                }
                valid_samples.append(sample_data)
                
        if attempts % 1000 == 0:
            print(f"  Attempts: {attempts}, Valid: {len(valid_samples)} ({(len(valid_samples)/max(1,attempts))*100:.1f}%)")
    
    print(f"\n--- Statistics ---")
    print(f"Total Attempts: {attempts}")
    print(f"Valid Samples: {len(valid_samples)}")
    print(f"Success Rate: {len(valid_samples)/max(1,attempts)*100:.2f}%")
    
    return valid_samples

def visualize_samples(samples, filename, title_suffix):
    if not samples: return
    plt.figure(figsize=(15, 5))
    count = min(3, len(samples))
    for i in range(count):
        s = samples[i]
        plt.subplot(1, 3, i+1)
        A = s['A']
        x0 = s['x0']
        fixed = np.where(s['types'])[0]
        motor = [0, 2]
        
        draw_mechanism(A, x0, fixed, motor, solve=True)
        
        ana = s['analysis']
        pts = {
            'Hip': (ana['hip'], 'black'),
            'Knee': (ana['knee'], 'blue'),
            'Ankle': (ana['ankle'], 'green'),
            'Foot': (ana['foot'], 'red')
        }
        
        for name, (idx, color) in pts.items():
            if idx != -1:
                plt.scatter(x0[idx, 0], x0[idx, 1], c=color, s=150, label=name, zorder=25, edgecolors='white', linewidth=2)
        
        skel = [ana['hip'], ana['knee'], ana['ankle'], ana['foot']]
        skel_valid = [idx for idx in skel if idx != -1]
        skel_coords = x0[skel_valid]
        if len(skel_coords) > 1:
            plt.plot(skel_coords[:,0], skel_coords[:,1], 'k--', linewidth=2, alpha=0.5)

        plt.legend()
        plt.title(f"{title_suffix} Sample {i}")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    TARGET_SAMPLES = 80000
    print(f"Starting generation of {TARGET_SAMPLES} DIVERSE samples using Strategy B...")
    start_time = time.time()
    
    data_b = generate_dataset('B', TARGET_SAMPLES)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Done in {duration:.2f} seconds ({duration/60:.2f} min)")
    
    output_file = "biological_6bar_dataset_80k_diverse.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data_b, f)
    print(f"Dataset saved to {output_file}")
    
    visualize_samples(data_b, "strategy_b_80k_diverse_viz.png", "Diverse Strategy B")

if __name__ == "__main__":
    main()
