import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --- 1. ARCHITECTURE ---
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x): return torch.sin(self.w0 * x)
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
        return self.decoder(gru_out[:, -1, :])
# --- 2. GENERATE RK4 TRAJECTORY ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing AI Astrophysics Engine on: {device}")
    model = AlienGRU_PINN().to(device)
    model.load_state_dict(torch.load("alien_model_v3_fast.pth", weights_only=True))
    model.eval()
    sim_id_to_test = 0
    df_input = pd.read_csv('alien_blind_test_input.csv')
    df_ans = pd.read_csv('alien_blind_test_answer.csv')
    input_traj = df_input[df_input['sim_id'] == sim_id_to_test].copy()
    true_ans = df_ans[df_ans['sim_id'] == sim_id_to_test].copy()
    seq_len = 5
    rolling_buffer = input_traj.iloc[-seq_len:][['x', 'y']].values.tolist()
    current_vx = input_traj.iloc[-1]['vx']
    current_vy = input_traj.iloc[-1]['vy']
    dt = 0.01
    x_arr, y_arr, vx_arr, vy_arr = [], [], [], []
    def get_ai_accel(temp_buffer):
        seq_tensor = torch.tensor([temp_buffer], dtype=torch.float32).to(device)
        return model(seq_tensor)[0].cpu().numpy()
    print(f"🧨 Extracting AI Predictions for Sim {sim_id_to_test}...")
    with torch.no_grad():
        for _ in range(len(true_ans)):
            current_x, current_y = rolling_buffer[-1][0], rolling_buffer[-1][1]
            k1_v = get_ai_accel(rolling_buffer)
            k1_x = np.array([current_vx, current_vy])
            mid_buffer_1 = rolling_buffer.copy()
            mid_buffer_1[-1] = [current_x + k1_x[0] * (dt / 2), current_y + k1_x[1] * (dt / 2)]
            k2_v = get_ai_accel(mid_buffer_1)
            k2_x = np.array([current_vx + k1_v[0] * (dt / 2), current_vy + k1_v[1] * (dt / 2)])
            mid_buffer_2 = rolling_buffer.copy()
            mid_buffer_2[-1] = [current_x + k2_x[0] * (dt / 2), current_y + k2_x[1] * (dt / 2)]
            k3_v = get_ai_accel(mid_buffer_2)
            k3_x = np.array([current_vx + k2_v[0] * (dt / 2), current_vy + k2_v[1] * (dt / 2)])
            end_buffer = rolling_buffer.copy()
            end_buffer[-1] = [current_x + k3_x[0] * dt, current_y + k3_x[1] * dt]
            k4_v = get_ai_accel(end_buffer)
            k4_x = np.array([current_vx + k3_v[0] * dt, current_vy + k3_v[1] * dt])
            current_vx += (dt / 6.0) * (k1_v[0] + 2 * k2_v[0] + 2 * k3_v[0] + k4_v[0])
            current_vy += (dt / 6.0) * (k1_v[1] + 2 * k2_v[1] + 2 * k3_v[1] + k4_v[1])
            pred_x = current_x + (dt / 6.0) * (k1_x[0] + 2 * k2_x[0] + 2 * k3_x[0] + k4_x[0])
            pred_y = current_y + (dt / 6.0) * (k1_x[1] + 2 * k2_x[1] + 2 * k3_x[1] + k4_x[1])
            x_arr.append(pred_x);
            y_arr.append(pred_y)
            vx_arr.append(current_vx);
            vy_arr.append(current_vy)
            rolling_buffer.append([pred_x, pred_y])
            rolling_buffer.pop(0)
    # --- 3. DEEP PHYSICS ANALYTICS ---
    print("🔭 Calculating Thermodynamics & Momentum...")
    x = np.array(x_arr);
    y = np.array(y_arr)
    vx = np.array(vx_arr);
    vy = np.array(vy_arr)
    time_steps = np.arange(len(x)) * dt
    r = np.sqrt(x ** 2 + y ** 2)
    v2 = vx ** 2 + vy ** 2
    k_ai = model.k.item()
    gamma_ai = model.gamma.item()
    KE = 0.5 * v2
    PE = -k_ai / ((gamma_ai - 1.0) * (r ** (gamma_ai - 1.0)))
    TE = KE + PE
    L = x * vy - y * vx
    # --- 4. DASHBOARD VISUALIZATION ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"AI Physics Analytics Dashboard (Sim {sim_id_to_test})", color='#00FFCC', fontsize=18,
                 fontweight='bold')
    ax1 = fig.add_subplot(2, 2, (1, 3))
    ax1.scatter(0, 0, color='#FFD700', s=400, label='Alien Sun', zorder=5)
    ax1.plot(x, y, color='#00FFCC', linewidth=2, label='AI Predicted Orbit')
    ax1.set_title("Predicted Trajectory", color='white')
    ax1.set_xlabel("X");
    ax1.set_ylabel("Y")
    ax1.legend(facecolor='#222222')
    ax1.grid(color='#333333', linestyle=':')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_steps, KE, color='#FF3366', label='Kinetic Energy (KE)')
    ax2.plot(time_steps, PE, color='#3366FF', label='Potential Energy (PE)')
    ax2.plot(time_steps, TE, color='#FFFFFF', linewidth=2, linestyle='--', label='Total Energy (TE)')
    ax2.axhline(0, color='grey', linewidth=1)
    ax2.fill_between(time_steps, 0, max(max(KE), 1), color='red', alpha=0.1, label='Escape Velocity Zone')
    ax2.set_title("Conservation of Energy", color='white')
    ax2.set_ylabel("Energy Joules (J)")
    ax2.legend(facecolor='#222222', loc='upper right')
    ax2.grid(color='#333333', linestyle=':')
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(time_steps, L, color='#FFCC00', linewidth=2, label='Angular Momentum (L)')
    ax3.set_title("Conservation of Angular Momentum", color='white')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Momentum")
    l_variance = np.var(L)
    ax3.text(0.05, 0.1, f"Momentum Variance: {l_variance:.6f}", transform=ax3.transAxes, color='white', fontsize=10,
             bbox=dict(facecolor='#222222', alpha=0.7))
    ax3.legend(facecolor='#222222')
    ax3.grid(color='#333333', linestyle=':')
    plt.tight_layout()
    plt.show()
