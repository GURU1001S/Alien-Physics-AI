# 🪐 Alien Physics AI: GRU-PINN Orbital Mechanics

### 📊 Model Architecture & Core Parameters
* **Core Architecture:** Gated Recurrent Unit (GRU) + Physics-Informed Neural Network (PINN).
* **Input State:** 3D Tensor `[Batch, Sequence_Length=5, Features=2]` (Sliding Window).
* **Integration Method:** Runge-Kutta 4th Order (RK4) replacing standard Euler.
* **Physics Law Target:** Inverse-Cube Gravity ($F \propto 1/r^3$).
* **Loss Function:** Hybrid SmoothL1Loss + Mean Squared Error (Data Loss + Physics Residual).
* **Singularity Safety Net:** `torch.clamp(r, min=0.5)` to prevent `NaN` gradient explosions.

### ⚡ Hardware & MLOps Optimization (RTX)
* **Data Pipeline:** 100% VRAM Dataset Caching (Zero PCIe bottleneck).
* **Training Time:** Reduced from ~40 minutes to **< 2 minutes**.
* **Memory Management:** `.contiguous()` applied to defragment C++ cuDNN memory blocks.
* **L-BFGS Optimization:** Sub-sampled to 4,096 sequences to prevent 22GB+ VRAM OOM crashes.

### 🧪 Quantitative Results (100 Unseen Trajectories)
* **Target Gamma ($\gamma$):** 3.0000
* **AI Discovered Gamma:** **2.9945**
* **Best Case Orbit (Sim 80) MSE:** `0.000009` (Near-perfect kinematic tracking).
* **Average Stable Orbit MSE:** `~0.000039`
* **Overall 100-Sim Average MSE:** `1.1270` (Skewed heavily by singularity edge-cases).
* **Worst Case Orbit (Sim 67) MSE:** `33.7969` (The "Slingshot Anomaly").

### 🔭 Physics & Behavioral Observations
* **The Slingshot Anomaly:** The AI successfully tracks normal orbits but mathematically fails during extreme solar approaches (Sim 67). Observation proves the `clamp(min=0.5)` safety net forces the AI to "cap" maximum gravity, resulting in smooth curve predictions while the true $1/r^3$ physics engine triggers violent, straight-line ejections.
* **Conservation of Angular Momentum ($L = \vec{r} \times \vec{v}$):** The AI organically learned Kepler's Laws. Analytics on RK4-integrated outputs show an Angular Momentum Variance of just **0.000198** across stable orbits.
* **Thermodynamic Tracking:** Model outputs successfully map the inverse relationship between Kinetic ($KE$) and Potential Energy ($PE$), accurately plotting Escape Velocity triggers when Total Energy ($TE > 0$).

### 📸 Visual Analytics
* **[Perfect Orbit (Sim 0)](assets/perfect_orbit.png)** - Demonstrating GRU momentum tracking and RK4 integration.
* **[The Slingshot Anomaly (Sim 67)](assets/slingshot_anomaly.png)** - Visualizing the AI's capped-gravity blind spot vs. True Physics.
* **[Deep Physics Dashboard](assets/physics_dashboard.png)** - Extracted Thermodynamics and Momentum Conservation data.
