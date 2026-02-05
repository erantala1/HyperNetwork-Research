import numpy as np
import os
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt

# Load all turbulence data into memory
def load_turbulence_data(data_dir, start_idx=10000, stop_idx=19999, normalize=False):
    data = {}
    print(f"Loading {stop_idx - start_idx + 1} files...")
    for idx in range(start_idx, stop_idx + 1):
        file_path = os.path.join(data_dir, f"{idx}.mat")
        data[idx] = loadmat(file_path)['Omega']
    
    if normalize:
        all_data = np.stack(list(data.values()))
        mean, std = np.mean(all_data), np.std(all_data)
        for idx in data:
            data[idx] = (data[idx] - mean) / std
    
    return data

# Create a dataloader that yields random batches
def create_turbulence_dataloader(data, batch_size, dt=5, shuffle=True, seed=0):
    # Create valid pairs of indices (input, target) separated by dt
    indices = sorted(list(data.keys()))
    pairs = [(indices[i], indices[i+dt]) for i in range(len(indices)-dt) 
             if indices[i+dt] - indices[i] == dt]
    
    num_samples = len(pairs)
    num_batches = num_samples // batch_size
    
    # Initialize random generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Shuffle all indices if needed
    if shuffle:
        perm = torch.randperm(num_samples, generator=generator).tolist()
        shuffled_pairs = [pairs[i] for i in perm]
    else:
        shuffled_pairs = pairs
    
    # Yield batches
    for i in range(num_batches):
        batch_pairs = shuffled_pairs[i * batch_size:(i + 1) * batch_size]
        
        inputs = []
        targets = []
        for input_idx, target_idx in batch_pairs:
            inputs.append(data[input_idx])
            targets.append(data[target_idx])
        
        yield torch.tensor(np.expand_dims(inputs, axis=1), dtype=torch.float32), torch.tensor(np.expand_dims(targets, axis=1), dtype=torch.float32)

