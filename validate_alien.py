import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --- 1. REBUILD THE GRU-PINN ARCHITECTURE ---
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
        final_memory = gru_out[:, -1, :]  # Extract the final thought
        return self.decoder(final_memory)


# --- 2. EXECUTION & SEQUENCE SIMULATION (RK4 UPGRADE) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlienGRU_PINN().to(device)
    model.load_state_dict(torch.load("alien_model_v3_fast.pth", weights_only=True))
    model.eval()

    # Let's test the broken Slingshot Anomaly!
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

    predicted_x, predicted_y = [], []


    # 🔴 HELPER FUNCTION: Ask the AI for acceleration at any given position
    def get_ai_accel(temp_buffer):
        seq_tensor = torch.tensor([temp_buffer], dtype=torch.float32).to(device)
        return model(seq_tensor)[0].cpu().numpy()


    print("🚩 GRU-PINN Simulating future trajectory using Aerospace RK4 Integration...")
    with torch.no_grad():
        for _ in range(len(true_ans)):
            current_x = rolling_buffer[-1][0]
            current_y = rolling_buffer[-1][1]

            # --- THE RUNGE-KUTTA 4 (RK4) ALGORITHM ---

            # k1: Initial slope (current state)
            k1_v = get_ai_accel(rolling_buffer)
            k1_x = np.array([current_vx, current_vy])

            # k2: Predict midpoint slope
            mid_buffer_1 = rolling_buffer.copy()
            mid_buffer_1[-1] = [current_x + k1_x[0] * (dt / 2), current_y + k1_x[1] * (dt / 2)]
            k2_v = get_ai_accel(mid_buffer_1)
            k2_x = np.array([current_vx + k1_v[0] * (dt / 2), current_vy + k1_v[1] * (dt / 2)])

            # k3: Refine midpoint slope
            mid_buffer_2 = rolling_buffer.copy()
            mid_buffer_2[-1] = [current_x + k2_x[0] * (dt / 2), current_y + k2_x[1] * (dt / 2)]
            k3_v = get_ai_accel(mid_buffer_2)
            k3_x = np.array([current_vx + k2_v[0] * (dt / 2), current_vy + k2_v[1] * (dt / 2)])

            # k4: Predict endpoint slope
            end_buffer = rolling_buffer.copy()
            end_buffer[-1] = [current_x + k3_x[0] * dt, current_y + k3_x[1] * dt]
            k4_v = get_ai_accel(end_buffer)
            k4_x = np.array([current_vx + k3_v[0] * dt, current_vy + k3_v[1] * dt])

            # 🔴 RK4 WEIGHTED AVERAGE COMBINATION
            final_vx = current_vx + (dt / 6.0) * (k1_v[0] + 2 * k2_v[0] + 2 * k3_v[0] + k4_v[0])
            final_vy = current_vy + (dt / 6.0) * (k1_v[1] + 2 * k2_v[1] + 2 * k3_v[1] + k4_v[1])

            final_x = current_x + (dt / 6.0) * (k1_x[0] + 2 * k2_x[0] + 2 * k3_x[0] + k4_x[0])
            final_y = current_y + (dt / 6.0) * (k1_x[1] + 2 * k2_x[1] + 2 * k3_x[1] + k4_x[1])

            # Update physical state
            current_vx, current_vy = final_vx, final_vy

            predicted_x.append(final_x)
            predicted_y.append(final_y)

            # Slide the window forward into the future
            rolling_buffer.append([final_x, final_y])
            rolling_buffer.pop(0)

    # --- 3. PLOTTING ---
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))

    plt.scatter(0, 0, color='#FFD700', s=300, label='Alien Sun', zorder=5)
    plt.plot(input_traj['x'], input_traj['y'], color='white', linestyle='--', label='Known Past (Input)')
    plt.plot(true_ans['x'], true_ans['y'], color='#555555', linewidth=4, label='True Alien Physics')
    plt.plot(predicted_x, predicted_y, color='#00FFCC', linewidth=2, label='AI Prediction (RK4 Upgrade)', zorder=4)

    plt.title(f"AI Alien Law Verification with RK4 Integration (Sim {sim_id_to_test})", color='#00FFCC', fontsize=16,
              fontweight='bold')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(facecolor='#222222', edgecolor='#00FFCC')
    plt.grid(color='#333333', linestyle=':')

    print("🧧 Rendering the Sequence-Aware Alien Universe...")
    plt.show()