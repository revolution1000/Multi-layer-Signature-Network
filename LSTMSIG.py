import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. YOUR SIGNATURE NETWORK (UNCHANGED)
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
# 2. BASELINE: LSTM
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :] 
        return self.fc(last_out)

# ==========================================
# 3. DATA GENERATION & UTILS
# ==========================================

def create_dataset(data, window_size):
    """Converts a time series into (window, target) pairs."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def get_sine_wave(n_points=1000):
    """Sub-Example 1: Simple clean Sine wave"""
    t = np.linspace(0, 50, n_points)
    data = np.sin(t)
    return data

def get_complex_wave(n_points=1000):
    """Sub-Example 2: Sine + Cosine + High Freq Noise (Harder)"""
    t = np.linspace(0, 50, n_points)
    # sin(x) + 0.5*cos(2x) + slight modulation
    data = np.sin(t) + 0.5 * np.cos(2 * t) + 0.2 * np.sin(5 * t)
    return data

def generate_sparse_binary_K(S, m, density=0.3):
    """Generates K with only 0s and 1s, with given density."""
    K = torch.zeros(S, m)
    # Randomly select indices to set to 1
    num_ones = int(S * m * density)
    indices = torch.randperm(S * m)[:num_ones]
    rows = indices // m
    cols = indices % m
    K[rows, cols] = 1.0
    
    # Ensure no row is all zeros (dead feature) - force at least one '1' per row
    row_sums = K.sum(dim=1)
    zero_rows = (row_sums == 0).nonzero(as_tuple=False).squeeze()
    if zero_rows.numel() > 0:
        if zero_rows.dim() == 0: zero_rows = zero_rows.unsqueeze(0)
        random_cols = torch.randint(0, m, (zero_rows.size(0),))
        K[zero_rows, random_cols] = 1.0
        
    return K

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train_model(model, X_train, y_train, epochs=200, lr=0.01, model_type="mlp"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Convert to Tensors
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    
    # LSTM expects (Batch, Seq, Features)
    # SignatureNet expects (Batch, Features)
    if model_type == "lstm":
        X_t = X_t.unsqueeze(-1) 
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.5f}")
            
    return losses

def predict(model, X_data, model_type="mlp"):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_data)
        if model_type == "lstm":
            X_t = X_t.unsqueeze(-1)
        preds = model(X_t)
    return preds.numpy()

# ==========================================
# 5. MAIN EXPERIMENT
# ==========================================
def run_comparison(data_name, data_func, window_size=20):
    print(f"\n========================================")
    print(f"   TASK: {data_name}")
    print(f"========================================")
    
    # 1. Prepare Data
    raw_data = data_func()
    X, y = create_dataset(raw_data, window_size)
    
    # Split (First 800 train, rest test)
    split_idx = 800
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # 2. Train LSTM
    print("\nTraining LSTM...")
    # input_size=1 (just the value), hidden=32
    lstm = LSTMModel(input_size=1, hidden_size=32, output_size=1)
    train_model(lstm, X_train, y_train, epochs=300, lr=0.01, model_type="lstm")
    lstm_preds = predict(lstm, X_test, model_type="lstm")
    
    # 3. Train Signature Network
    print("\nTraining Signature Network...")
    # Config
    d_0 = window_size # Input is the flattened window
    m = 50
    S = 50
    N = 3
    
    # ---> Constraint: Sparse Binary K <---
    K_vals = generate_sparse_binary_K(S, m, density=0.2) # 20% sparsity
    
    sig_net = SignatureNetwork(d_0, m, S, N, output_dim=1, K=K_vals)
    train_model(sig_net, X_train, y_train, epochs=300, lr=0.005, model_type="mlp")
    sig_preds = predict(sig_net, X_test, model_type="mlp")
    
    # 4. Evaluation
    mse_lstm = np.mean((y_test - lstm_preds.flatten())**2)
    mse_sig = np.mean((y_test - sig_preds.flatten())**2)
    
    print(f"\n--- Final Comparison ({data_name}) ---")
    print(f"LSTM MSE:      {mse_lstm:.6f}")
    print(f"Signature MSE: {mse_sig:.6f}")
    
    # Simple ASCII Plot of the first 50 test points
    print("\nVisual check (First 20 test points):")
    print("GT : ", np.round(y_test[:10], 2))
    print("LSTM: ", np.round(lstm_preds.flatten()[:10], 2))
    print("SigN: ", np.round(sig_preds.flatten()[:10], 2))

if __name__ == "__main__":
    # Sub-Example 1: Pure Sine
    run_comparison("Simple Sine Wave", get_sine_wave)
    
    # Sub-Example 2: Complex Composite Wave
    run_comparison("Complex Composite Wave", get_complex_wave)