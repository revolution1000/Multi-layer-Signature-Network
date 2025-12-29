import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
# 2. BASELINE: STANDARD LSTM
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 1 Layer LSTM is standard for this benchmark
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (Batch, Seq_Len, 2)
        lstm_out, _ = self.lstm(x)
        # We only care about the result after processing the whole sequence
        last_hidden = lstm_out[:, -1, :] 
        return self.fc(last_hidden)

# ==========================================
# 3. EXPERIMENT UTILS
# ==========================================

def get_adding_problem_data(batch_size, seq_len):
    """
    Generates the Adding Problem dataset.
    Returns:
        X: (batch, seq_len, 2)
        y: (batch, 1)
    """
    # Channel 0: Random values in [0, 1]
    values = torch.rand(batch_size, seq_len)
    
    # Channel 1: Mask (Binary)
    # Set all to 0 initially
    mask = torch.zeros(batch_size, seq_len)
    
    # Randomly pick two distinct indices per sample to set to 1
    for i in range(batch_size):
        indices = torch.randperm(seq_len)[:2]
        mask[i, indices] = 1.0
        
    X = torch.stack([values, mask], dim=2) # (Batch, Seq, 2)
    
    # Calculate target: Sum of values where mask is 1
    # Dot product along seq dimension
    y = (values * mask).sum(dim=1).unsqueeze(1) 
    
    return X, y

def generate_sparse_binary_K(S, m, density=0.2):
    """Generates K with only 0s and 1s."""
    K = torch.zeros(S, m)
    # Randomly set some entries to 1
    num_ones = int(S * m * density)
    indices = torch.randperm(S * m)[:num_ones]
    rows = indices // m
    cols = indices % m
    K[rows, cols] = 1.0
    
    # Safety: Ensure no row is dead (all zeros)
    row_sums = K.sum(dim=1)
    zero_rows = (row_sums == 0).nonzero(as_tuple=False).squeeze()
    if zero_rows.numel() > 0:
        if zero_rows.dim() == 0: zero_rows = zero_rows.unsqueeze(0)
        random_cols = torch.randint(0, m, (zero_rows.size(0),))
        K[zero_rows, random_cols] = 1.0
        
    return K

def train_network(model, model_name, seq_len, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    # We generate new data on the fly (infinite dataset regime) 
    # to prevent overfitting to a small fixed set
    batch_size = 64
    losses = []
    
    print(f"Training {model_name}...")
    for epoch in range(epochs):
        model.train()
        X, y = get_adding_problem_data(batch_size, seq_len)
        
        # Prepare inputs based on model type
        if model_name == "SignatureNetwork":
            # Flatten: (Batch, Seq, 2) -> (Batch, Seq*2)
            X_in = X.view(batch_size, -1)
        else:
            X_in = X
            
        optimizer.zero_grad()
        preds = model(X_in)
        loss = criterion(preds, y)
        loss.backward()
        
        # Clip grads for LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.5f}")

    return losses

# ==========================================
# 4. MAIN RUNNER
# ==========================================
def run_comparison(seq_len):
    print(f"\n=============================================")
    print(f" Adding Problem: Sequence Length = {seq_len}")
    print(f"=============================================")
    
    # --- 1. LSTM Setup ---
    # Hidden size 128 is standard for this task
    lstm = LSTMModel(input_size=2, hidden_size=128, output_size=1)
    
    # --- 2. Signature Network Setup ---
    # Input dim is flattened sequence (seq_len * 2)
    d_0 = seq_len * 2 
    m = 128   # Hidden dimension
    S = 64    # Number of signatures per block
    N = 2     # Number of blocks
    
    # Sparse Binary K constraint
    K_vals = generate_sparse_binary_K(S, m, density=0.1) # Sparse!
    
    sig_net = SignatureNetwork(d_0, m, S, N, output_dim=1, K=K_vals)
    
    # --- 3. Run Training ---
    # Using fewer epochs for demonstration; increase for paper results
    loss_lstm = train_network(lstm, "LSTM", seq_len, epochs=500)
    loss_sig = train_network(sig_net, "SignatureNetwork", seq_len, epochs=500)
    
    # --- 4. Final Validation ---
    print("\n--- Validation Test (New Data) ---")
    X_val, y_val = get_adding_problem_data(1000, seq_len)
    
    # LSTM Eval
    lstm.eval()
    with torch.no_grad():
        p_lstm = lstm(X_val)
        mse_lstm = nn.MSELoss()(p_lstm, y_val).item()
        
    # SigNet Eval
    sig_net.eval()
    with torch.no_grad():
        p_sig = sig_net(X_val.view(1000, -1))
        mse_sig = nn.MSELoss()(p_sig, y_val).item()
        
    print(f"Final MSE (LSTM):      {mse_lstm:.6f}")
    print(f"Final MSE (Signature): {mse_sig:.6f}")
    
    if mse_sig < mse_lstm:
        print(">> RESULT: Signature Network Wins!")
    else:
        print(">> RESULT: LSTM Wins (or task requires more training)")

if __name__ == "__main__":
    # Sub-Example 1: Moderate length (Standard benchmark)
    run_comparison(seq_len=50)
    
    # Sub-Example 2: Longer length (Harder for LSTM)
    # LSTMs often struggle to converge here without many epochs
    # Signature Network sees the whole vector, so length just increases input dim
    run_comparison(seq_len=100)