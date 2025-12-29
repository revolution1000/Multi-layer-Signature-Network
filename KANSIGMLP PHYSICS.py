import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

# --- 1. REPRODUCIBILITY ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_seed(42)

# ==========================================
# MODEL 1: YOUR SIGNATURE NETWORK (VERBATIM)
# ==========================================
class SignatureBlock(nn.Module):
    # The terms of K should be integers
    def __init__(self, d_in, m, S, K):
        super().__init__()
        self.linear = nn.Linear(d_in, m)
        self.register_buffer("K", K)

    def forward(self, x):
        z = self.linear(x)
        # z shape: (batch, m)
        # K shape: (S, m) -> unsqueeze to (1, S, m)
        # z unsqueeze to (batch, 1, m)
        powered = z.unsqueeze(1) ** self.K.unsqueeze(0)
        signature = powered.prod(dim=-1) # (batch, S)
        return signature

class SignatureNetwork(nn.Module):
    def __init__(self, d_0, m, S, N, output_dim, K):
        super().__init__()
        self.blocks = nn.ModuleList()
        # Every block now receives (current_features + original_input_d0)
        for i in range(N):
            input_dim = d_0 if i == 0 else (S + d_0)
            self.blocks.append(SignatureBlock(input_dim, m, S, K))
        
        # Final projection takes the concatenation of all N block outputs
        # plus the output of the very last block (as requested)
        self.final_projection = nn.Linear(S * N, output_dim)

    def forward(self, x_orig):
        block_outputs = []
        x_current = x_orig
        
        for i, block in enumerate(self.blocks):
            if i == 0:
                # First block only takes the original input
                out = block(x_current)
            else:
                # Subsequent blocks take [previous_output, original_input]
                combined_input = torch.cat([x_current, x_orig], dim=-1)
                out = block(combined_input)
            
            x_current = out
            block_outputs.append(out)
        
        # Linear combination of all block outputs
        all_features = torch.cat(block_outputs, dim=-1)
        
        return self.final_projection(all_features)

# ==========================================
# MODEL 2: KAN (Kolmogorov-Arnold Network) - FIXED
# ==========================================
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid for B-splines
        h = (1 - (-1)) / grid_size
        # Shape: (grid_size + 2*spline_order + 1,) -> 1D Tensor
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h - 1).float()
        self.register_buffer("grid", grid)
        
        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        # Initialization
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * 0.1) 
        nn.init.normal_(self.spline_weight, std=0.01)

    def b_splines(self, x):
        # x shape: (batch, in)
        x = x.unsqueeze(-1) # (batch, in, 1)
        grid = self.grid    # (total_grid_pts,)

        # FIX: Slicing for 1D grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        for k in range(1, self.spline_order + 1):
            # FIX: Slicing for 1D grid
            # Left part
            numer1 = (x - grid[:-(k + 1)])
            denom1 = (grid[k:-1] - grid[:-(k + 1)])
            term1 = (numer1 / denom1) * bases[..., :-1]
            
            # Right part
            numer2 = (grid[k + 1:] - x)
            denom2 = (grid[k + 1:] - grid[1:-k])
            term2 = (numer2 / denom2) * bases[..., 1:]
            
            bases = term1 + term2
            
        return bases

    def forward(self, x):
        # 1. Base activation
        base_output = F.silu(x) @ self.base_weight.t()
        
        # 2. Spline activation
        # Normalize x to [-1, 1] to fit the grid
        x_norm = torch.tanh(x) 
        spline_basis = self.b_splines(x_norm) 
        
        # (batch, in, coeff) * (out, in, coeff) -> sum over coeff and in
        spline_output = torch.einsum('bic,oic->bo', spline_basis, self.spline_weight)
        
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(KANLayer(in_dim, hidden_dim))
        for _ in range(layers - 1):
            self.layers.append(KANLayer(hidden_dim, hidden_dim))
        self.final = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final(x)

# ==========================================
# MODEL 3: MLP (Baseline)
# ==========================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

# ==========================================
# EXPERIMENT RUNNER
# ==========================================
def run_feynman_experiment():
    print("=== EXPERIMENT 3: The Feynman Benchmark (Gravity) ===")
    print("Target: F = m1 * m2 / r^2")
    
    # 1. Generate Data (Inverse Square Law)
    # Range [1, 3] to avoid division by zero and extremely large values
    N = 2000
    X = torch.rand(N, 3) * 2 + 1 # Features: m1, m2, r (Range 1.0 to 3.0)
    
    m1, m2, r = X[:, 0], X[:, 1], X[:, 2]
    y = (m1 * m2 / (r ** 2)).unsqueeze(1)
    
    # Split
    train_size = int(0.8 * N)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 2. Setup Models
    d_in = 3
    
    # A. Signature Network (Yours)
    # Note: K includes -2 to capture 1/r^2
    K_init = torch.randint(-2, 3, (10, 10)).float() * (torch.rand(10, 10) > 0.5).float()
    model_sig = SignatureNetwork(d_0=d_in, m=10, S=10, N=2, output_dim=1, K=K_init)
    
    # B. KAN (The Competitor)
    model_kan = KAN(in_dim=d_in, hidden_dim=8, out_dim=1) 
    
    # C. MLP (The Baseline)
    model_mlp = MLP(d_in, 64, 1)

    # 3. Training Loop
    models = {
        'SignatureNet': model_sig,
        'KAN': model_kan,
        'MLP': model_mlp
    }
    
    # Trackers
    history_loss = {k: [] for k in models}
    history_time = {k: [] for k in models}
    final_times = {}

    loss_fn = nn.MSELoss()
    EPOCHS = 200
    
    print(f"\nTraining for {EPOCHS} epochs...")
    
    for name, model in models.items():
        # Optimizer
        opt = optim.Adam(model.parameters(), lr=0.01)
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            
            if torch.isnan(loss):
                # Just a warning to the user to tune hyperparameters
                print(f"Warning: {name} NaN loss at epoch {epoch}")
                history_loss[name].append(float('nan'))
                history_time[name].append(time.time() - start_time)
                break

            loss.backward()
            
            # Gradient Clipping (Standard practice, does not change architecture)
            if name == 'SignatureNet':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            opt.step()
            
            # Record
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test)
                    test_loss = loss_fn(test_pred, y_test).item()
                    history_loss[name].append(test_loss)
                    history_time[name].append(time.time() - start_time)
        
        total_time = time.time() - start_time
        final_times[name] = total_time
        # Use last valid loss
        final_loss = history_loss[name][-1] if len(history_loss[name]) > 0 else float('nan')
        print(f"{name} finished in {total_time:.2f}s | Final MSE: {final_loss:.2e}")

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot A: Convergence by Epoch
    for name in models:
        ax1.plot(history_loss[name], label=name, linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title('Convergence Speed (By Epochs)')
    ax1.set_xlabel('Epoch Steps (x5)')
    ax1.set_ylabel('MSE Loss (Log)')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Plot B: Convergence by Wall-Clock Time
    for name in models:
        if len(history_time[name]) == len(history_loss[name]):
            ax2.plot(history_time[name], history_loss[name], label=name, linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title('Real-World Efficiency (By Time)')
    ax2.set_xlabel('Training Time (Seconds)')
    ax2.set_ylabel('MSE Loss (Log)')
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_feynman_experiment()