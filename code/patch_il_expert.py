import torch
import os
print("Loading il_expert_paths_80k.pt...")
path = "f:/LINKS4Meta/LINKS-main/il_expert_paths_80k.pt"
data = torch.load(path, map_location='cpu', weights_only=False)
for d in data:
    d['base_data'].keypoints = torch.tensor([5, 2, 4], dtype=torch.long)
torch.save(data, path)
print("Successfully patched `il_expert_paths_80k.pt` and injected keypoints [5, 2, 4].")
