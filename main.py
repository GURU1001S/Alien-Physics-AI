import numpy as np
import pandas as pd


def generate_alien_data(num_sims=100, timesteps=500, dt=0.01, k=5.0):
    all_data = []
    for sim_id in range(num_sims):
        # Random initial position and velocity
        x, y = np.random.uniform(2, 5), 0.0
        vx, vy = 0.0, np.random.uniform(0.5, 1.5)

        traj = []
        for t in range(timesteps):
            r = np.sqrt(x ** 2 + y ** 2)
            # THE ALIEN LAW: Inverse Cube (r**3)
            accel = -k / (r ** 3)
            ax, ay = accel * (x / r), accel * (y / r)

            # Update Velocity and Position (Euler-Cromer)
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt

            traj.append([sim_id, t, x, y, vx, vy])
            if r < 0.1 or r > 20: break  # Stop if it crashes or escapes

        all_data.extend(traj)

    return pd.DataFrame(all_data, columns=['sim_id', 't', 'x', 'y', 'vx', 'vy'])


# Create your three datasets
train_df = generate_alien_data(num_sims=1000)
test_full = generate_alien_data(num_sims=100)

# Split test_full into Input (first 50 steps) and Answer (the rest)
test_input = test_full[test_full['t'] < 50]
test_answer = test_full[test_full['t'] >= 50]

print("Datasets Generated! Training rows:", len(train_df))
# Save the datasets to your local folder
train_df.to_csv('alien_train_data.csv', index=False)
test_input.to_csv('alien_blind_test_input.csv', index=False)
test_answer.to_csv('alien_blind_test_answer.csv', index=False)

print("Files successfully saved to your folder! 🚀")
import matplotlib.pyplot as plt

# Plot one simulation to see the "Alien" spiral
one_sim = train_df[train_df['sim_id'] == 0]
plt.plot(one_sim['x'], one_sim['y'], 'r-')
plt.title("Alien Inverse-Cube Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()