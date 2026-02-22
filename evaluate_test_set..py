import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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


# --- 2. BULK EVALUATION WITH RK4 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Mass RK4 Evaluation on: {device}")

    # Load Model
    model = AlienGRU_PINN().to(device)
    model.load_state_dict(torch.load("alien_model_v3_fast.pth", weights_only=True))
    model.eval()

    # Load Full Test Datasets
    df_input = pd.read_csv('alien_blind_test_input.csv')
    df_ans = pd.read_csv('alien_blind_test_answer.csv')

    unique_sims = df_input['sim_id'].unique()
    total_sims = len(unique_sims)

    print(f"🧨 Testing AI on {total_sims} unseen alien trajectories using Aerospace RK4 math...")

    all_mse = []
    dt = 0.01
    seq_len = 5


    # 🔴 HELPER FUNCTION: Allows RK4 to query the AI at intermediate steps
    def get_ai_accel(temp_buffer):
        seq_tensor = torch.tensor([temp_buffer], dtype=torch.float32).to(device)
        return model(seq_tensor)[0].cpu().numpy()


    with torch.no_grad():
        for sim_id in unique_sims:
            input_traj = df_input[df_input['sim_id'] == sim_id]
            true_ans = df_ans[df_ans['sim_id'] == sim_id]

            if len(input_traj) < seq_len or len(true_ans) == 0:
                continue

            rolling_buffer = input_traj.iloc[-seq_len:][['x', 'y']].values.tolist()
            current_vx = input_traj.iloc[-1]['vx']
            current_vy = input_traj.iloc[-1]['vy']

            sim_errors = []

            for idx, row in true_ans.iterrows():
                true_x, true_y = row['x'], row['y']

                current_x = rolling_buffer[-1][0]
                current_y = rolling_buffer[-1][1]

                # --- THE RUNGE-KUTTA 4 (RK4) ALGORITHM ---
                # k1: Initial slope
                k1_v = get_ai_accel(rolling_buffer)
                k1_x = np.array([current_vx, current_vy])

                # k2: Midpoint slope 1
                mid_buffer_1 = rolling_buffer.copy()
                mid_buffer_1[-1] = [current_x + k1_x[0] * (dt / 2), current_y + k1_x[1] * (dt / 2)]
                k2_v = get_ai_accel(mid_buffer_1)
                k2_x = np.array([current_vx + k1_v[0] * (dt / 2), current_vy + k1_v[1] * (dt / 2)])

                # k3: Midpoint slope 2
                mid_buffer_2 = rolling_buffer.copy()
                mid_buffer_2[-1] = [current_x + k2_x[0] * (dt / 2), current_y + k2_x[1] * (dt / 2)]
                k3_v = get_ai_accel(mid_buffer_2)
                k3_x = np.array([current_vx + k2_v[0] * (dt / 2), current_vy + k2_v[1] * (dt / 2)])

                # k4: Endpoint slope
                end_buffer = rolling_buffer.copy()
                end_buffer[-1] = [current_x + k3_x[0] * dt, current_y + k3_x[1] * dt]
                k4_v = get_ai_accel(end_buffer)
                k4_x = np.array([current_vx + k3_v[0] * dt, current_vy + k3_v[1] * dt])

                # Combine with weighted averages
                current_vx += (dt / 6.0) * (k1_v[0] + 2 * k2_v[0] + 2 * k3_v[0] + k4_v[0])
                current_vy += (dt / 6.0) * (k1_v[1] + 2 * k2_v[1] + 2 * k3_v[1] + k4_v[1])

                pred_x = current_x + (dt / 6.0) * (k1_x[0] + 2 * k2_x[0] + 2 * k3_x[0] + k4_x[0])
                pred_y = current_y + (dt / 6.0) * (k1_x[1] + 2 * k2_x[1] + 2 * k3_x[1] + k4_x[1])

                # Calculate Error
                sq_error = (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2
                sim_errors.append(sq_error)

                # Slide the window forward
                rolling_buffer.append([pred_x, pred_y])
                rolling_buffer.pop(0)

            sim_mse = np.mean(sim_errors)
            all_mse.append(sim_mse)

            # Print progress
            if sim_id % 10 == 0:
                print(f"   Sim {sim_id:02d} processed | MSE: {sim_mse:.6f}")

    # --- 3. FINAL RESULTS ---
    final_avg_mse = np.mean(all_mse)
    worst_sim_id = unique_sims[np.argmax(all_mse)]
    worst_mse = np.max(all_mse)

    print("\n==========================================")
    print("🧧 RK4 BLIND TEST EVALUATION RESULTS 🧧")
    print("==========================================")
    print(f"Total Trajectories Evaluated: {len(all_mse)}")
    print(f"Model Discovered Gamma:       {model.gamma.item():.4f}")
    print(f"Overall Average MSE:          {final_avg_mse:.6f}")
    print(f"🔴 WORST Simulation ID:       {worst_sim_id} (MSE: {worst_mse:.6f})")
    print("==========================================")

    if final_avg_mse < 1.0:
        print("✅ PERFORMANCE: OUTSTANDING. RK4 mitigated the integration drift!")
    else:
        print("⚠️ PERFORMANCE: HIGH ERROR. The slingshot singularity still caps accuracy.")