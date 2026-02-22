import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
# ==========================================
# 1. AI ARCHITECTURE (GRU + PINN)
# ==========================================
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
class AlienGRU_PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), Sine(w0=30.0),
            nn.Linear(hidden_dim, hidden_dim), Sine(),
            nn.Linear(hidden_dim, 2)
        )
        self.gamma = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor([5.0], dtype=torch.float32))
    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_memory = gru_out[:, -1, :]
        return self.decoder(final_memory)
# ==========================================
# 2. SEQUENCE DATA PIPELINE
# ==========================================
class AlienSeqDataset(Dataset):
    def __init__(self, file, seq_len=5):
        df = pd.read_csv(file)
        dt = 0.01
        df['ax'] = df.groupby('sim_id')['vx'].diff().shift(-1) / dt
        df['ay'] = df.groupby('sim_id')['vy'].diff().shift(-1) / dt
        df = df.dropna()
        df['r'] = (df['x'] ** 2 + df['y'] ** 2) ** 0.5
        df = df[df['r'] > 0.5]
        x_seqs, a_seqs = [], []
        for _, group in df.groupby('sim_id'):
            coords = group[['x', 'y']].values
            accels = group[['ax', 'ay']].values
            for i in range(len(coords) - seq_len + 1):
                x_seqs.append(coords[i: i + seq_len])
                a_seqs.append(accels[i + seq_len - 1])
        self.x = torch.tensor(np.array(x_seqs), dtype=torch.float32)
        self.a = torch.tensor(np.array(a_seqs), dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.a[i]
# ==========================================
# 3. BULLETPROOF PHYSICS LOSS
# ==========================================
def hybrid_loss(model, seq_coords, true_accel, lambda_phys=2.0):
    pred_accel = model(seq_coords)
    loss_data = torch.mean((pred_accel - true_accel) ** 2)
    current_pos = seq_coords[:, -1, :]
    r = torch.norm(current_pos, dim=1, keepdim=True)
    r_safe = torch.clamp(r, min=0.5)
    physics_res = pred_accel + (model.k / (r_safe ** model.gamma)) * (current_pos / r_safe)
    loss_func = nn.SmoothL1Loss()
    loss_phys = loss_func(physics_res, torch.zeros_like(physics_res))
    return loss_data + (lambda_phys * loss_phys)
# ==========================================
# 4. HIGH-SPEED EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    print("🧧 Loading universe into VRAM Cache...")
    ds = AlienSeqDataset('alien_train_data.csv', seq_len=5)
    full_x_gpu = ds.x.to(device)
    full_a_gpu = ds.a.to(device)
    dataset_size = len(full_x_gpu)
    batch_size = 1024
    model = AlienGRU_PINN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    print("🧨 Starting Ultra-Fast AdamW Phase...")
    # --- PHASE 1: ADAMW ---
    for epoch in range(201):
        permutation = torch.randperm(dataset_size, device=device)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = full_x_gpu[indices].contiguous()
            batch_a = full_a_gpu[indices].contiguous()
            optimizer.zero_grad()
            loss = hybrid_loss(model, batch_x, batch_a, lambda_phys=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Learned Gamma: {model.gamma.item():.4f}")
    # --- PHASE 2: L-BFGS FINISHER ---
    print("\n🧧 Starting L-BFGS Phase (Safely sub-sampled for 6GB VRAM)...")
    # Sub-sample to avoid 22GB VRAM crash
    lbfgs_sample_size = 4096
    sample_indices = torch.randperm(dataset_size, device=device)[:lbfgs_sample_size]
    lbfgs_x = full_x_gpu[sample_indices].contiguous()
    lbfgs_a = full_a_gpu[sample_indices].contiguous()
    l_opt = optim.LBFGS(model.parameters(), lr=1, max_iter=50)
    def closure():
        l_opt.zero_grad()
        l = hybrid_loss(model, lbfgs_x, lbfgs_a, lambda_phys=2.0)
        l.backward()
        return l
    l_opt.step(closure)
    print(f"🔴 Final Discovery: Gamma = {model.gamma.item():.4f}")
    torch.save(model.state_dict(), "alien_model_v3_fast.pth")
    print("✅ Model saved successfully!")
