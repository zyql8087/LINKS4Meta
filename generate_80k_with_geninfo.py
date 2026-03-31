import sys, time, os, pickle
sys.path.insert(0, 'f:/LINKS4Meta/LINKS-main')
from dataset_builder import generate_dataset

print('Starting 80k dataset generation with gen_info...')
print('This will take approximately 30-60 minutes.\n')
start = time.time()

data = generate_dataset('B', 80000)

elapsed = time.time() - start
print(f'\nGeneration complete in {elapsed/60:.1f} minutes.')
print(f'Total valid samples: {len(data)}')

sample = data[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'gen_info present: {"gen_info" in sample}')
print(f'gen_info value: {sample.get("gen_info")}')

output_path = 'f:/LINKS4Meta/LINKS-main/biological_6bar_dataset_80k_with_geninfo.pkl'
print(f'\nSaving to {output_path} ...')
with open(output_path, 'wb') as f:
    pickle.dump(data, f)

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f'[OK] Saved! File size: {size_mb:.1f} MB')
