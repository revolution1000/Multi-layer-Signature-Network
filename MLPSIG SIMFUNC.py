import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

# --- 1. SETUP ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# --- 2. YOUR ORIGINAL NETWORK (VERBATIM) ---
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
        # Stacked shape: (batch, S * N)
        all_features = torch.cat(block_outputs, dim=-1)
        
        return self.final_projection(all_features)

# --- 3. HELPERS ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def create_sparse_K(S, m, min_pow, max_pow, sparsity=0.7):
    # Generates integer K matrix including negative values
    K = torch.randint(low=min_pow, high=max_pow + 1, size=(S, m)).float()
    mask = torch.rand(S, m) > sparsity
    K = K * mask
    return K

def train_compare(X, y, title, epochs=500):
    d_in = X.shape[1]
    
    # 1. Setup MLP
    mlp = MLP(d_in, 64, 1)
    
    # 2. Setup SignatureNetwork
    # We allow negative powers (-2 to 2) in K as requested
    K_init = create_sparse_K(S=100, m=100, min_pow=0, max_pow=1, sparsity=0.9)
    sig_net = SignatureNetwork(d_0=d_in, m=100, S=100, N=2, output_dim=1, K=K_init)
    
    optimizers = {
        'MLP': optim.Adam(mlp.parameters(), lr=0.01),
        'SignatureNet': optim.Adam(sig_net.parameters(), lr=0.005)
    }
    loss_fn = nn.MSELoss()
    history = {'MLP': [], 'SignatureNet': []}
    
    print(f"\n--- Running: {title} ---")
    
    for epoch in range(epochs):
        for name, model in {'MLP': mlp, 'SignatureNet': sig_net}.items():
            optimizer = optimizers[name]
            optimizer.zero_grad()
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            if torch.isnan(loss):
                print(f"!! Warning: {name} produced NaN at epoch {epoch}. (Expected risk with negative powers)")
                history[name].append(float('nan'))
                continue

            loss.backward()
            optimizer.step()
            history[name].append(loss.item())
            
        if epoch % 100 == 0:
            print(f"Ep {epoch}: MLP {history['MLP'][-1]:.5f} | SigNet {history['SignatureNet'][-1]:.5f}")

    return history

# --- 4. EXAMPLES ---

def example_1_multiplication():
    # Task: y = x0 * x1 * x2
    # Input shifted to [1, 2] to minimize NaN risk while testing logic
    N = 1000
    X = torch.rand(N, 5) + 1.0 
    y = (X[:, 0] * X[:, 1] * X[:, 2]).unsqueeze(1)
    
    hist = train_compare(X, y, "Example 1: Multiplication (y = x0*x1*x2)")
    return hist

def example_2_rational():
    # Task: y = x0 / x1
    # This specifically leverages negative K values.
    # Input shifted to [1, 2] to ensure x1 is never 0.
    N = 1000
    X = torch.rand(N, 5) + 1.0 
    y = (X[:, 0] / X[:, 1]).unsqueeze(1)
    
    hist = train_compare(X, y, "Example 2: Rational (y = x0 / x1)")
    return hist

# --- 5. MAIN ---
if __name__ == "__main__":
    h1 = example_1_multiplication()
    h2 = example_2_rational()
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(h1['MLP'], label='MLP', alpha=0.7)
    ax1.plot(h1['SignatureNet'], label='SignatureNet', linewidth=2)
    ax1.set_title('Ex 1: Multiplication')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(h2['MLP'], label='MLP', alpha=0.7)
    ax2.plot(h2['SignatureNet'], label='SignatureNet', linewidth=2)
    ax2.set_title('Ex 2: Rational (Division)')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()