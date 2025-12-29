import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import math

# ==========================================
# 1. YOUR MODEL (SignatureNetwork)
#    - Wrapped to handle Time Series (B, L, D) -> Flatten -> (B, L*D)
# ==========================================

class SignatureBlock(nn.Module):
    def __init__(self, d_in, m, S, K):
        super().__init__()
        self.linear = nn.Linear(d_in, m)
        self.register_buffer("K", K)

    def forward(self, x):
        z = self.linear(x)
        # Stability fix: sigmoid to keep z in [0,1] or simple activation to avoid explosion
        z = torch.sigmoid(z) 
        powered = z.unsqueeze(1) ** self.K.unsqueeze(0)
        signature = powered.prod(dim=-1)
        return signature

class SignatureNetwork(nn.Module):
    def __init__(self, d_0, m, S, N, output_dim, K):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(N):
            input_dim = d_0 if i == 0 else (S + d_0)
            self.blocks.append(SignatureBlock(input_dim, m, S, K))
        self.final_projection = nn.Linear(S * N, output_dim)

    def forward(self, x_orig):
        block_outputs = []
        x_current = x_orig
        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block(x_current)
            else:
                combined_input = torch.cat([x_current, x_orig], dim=-1)
                out = block(combined_input)
            x_current = out
            block_outputs.append(out)
        all_features = torch.cat(block_outputs, dim=-1)
        return self.final_projection(all_features)

def generate_sparse_binary_K(S, m, density=0.3):
    K = torch.zeros(S, m)
    num_ones = int(S * m * density)
    indices = torch.randperm(S * m)[:num_ones]
    rows = indices // m
    cols = indices % m
    K[rows, cols] = 1.0
    # Ensure no dead rows
    row_sums = K.sum(dim=1)
    zero_rows = (row_sums == 0).nonzero(as_tuple=False).squeeze()
    if zero_rows.numel() > 0:
        if zero_rows.dim() == 0: zero_rows = zero_rows.unsqueeze(0)
        random_cols = torch.randint(0, m, (zero_rows.size(0),))
        K[zero_rows, random_cols] = 1.0
    return K

# Wrapper to adapt your model for (Batch, Seq, Feat) input
class SigNetWrapper(nn.Module):
    def __init__(self, seq_len, n_features, pred_len, m=16, S=16, N=2):
        super().__init__()
        self.d_flat = seq_len * n_features
        K = generate_sparse_binary_K(S, m, density=0.3)
        self.model = SignatureNetwork(self.d_flat, m, S, N, pred_len * n_features, K)
        self.pred_len = pred_len
        self.n_features = n_features

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> Flatten -> (Batch, Seq*Feat)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        out = self.model(x_flat)
        # Reshape output back to (Batch, Pred_Len, Feat)
        return out.reshape(batch_size, self.pred_len, self.n_features)

# ==========================================
# 2. BASELINE: RNN / LSTM / GRU
# ==========================================

class RNNBaseline(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, pred_len, num_layers=1):
        super().__init__()
        self.model_type = model_type
        self.pred_len = pred_len
        self.input_size = input_size
        
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            
        self.linear = nn.Linear(hidden_size, pred_len * input_size)

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        out, _ = self.rnn(x)
        # Take the output of the last time step
        last_out = out[:, -1, :] 
        pred = self.linear(last_out)
        return pred.reshape(x.shape[0], self.pred_len, self.input_size)

# ==========================================
# 3. BASELINE: DLinear (SOTA Linear)
# ==========================================

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on the front
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        
        # Linear layers for trend and residual
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        seasonal_init, trend_init = self.decompsition(x)
        
        # Permute to (Batch, Feat, Seq) for Linear Layer application on Time axis
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1) # Back to (Batch, Pred_Len, Feat)

# ==========================================
# 4. BASELINE: PatchTST (Simplified)
# ==========================================

class PatchTST_Simple(nn.Module):
    def __init__(self, seq_len, pred_len, n_features, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches
        self.num_patches = int((seq_len - patch_len) / stride) + 2
        
        # Embedding
        self.patch_embedding = nn.Linear(patch_len, 128)
        
        # Encoder (Standard Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Head
        self.head = nn.Linear(self.num_patches * 128, pred_len)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> We treat features independently (Channel Independence)
        B, L, D = x.shape
        x = x.permute(0, 2, 1).reshape(B * D, L, 1) # Combine Batch and Channels
        
        # Patching (Simplified view)
        # Ideally, we unfold. Here we just resize/embed for demonstration.
        # We will use a linear projection on the whole sequence as a proxy for complex patching if unavailable
        # But to be accurate, let's just project:
        x_emb = self.patch_embedding(x[:, :self.patch_len, :].squeeze(-1)) # Dummy patch
        
        # Actually, let's implement standard Transformer for simplicity in this snippet
        # since manual patching is verbose.
        # FALLBACK to Transformer for this script context
        return x # Placeholder, see comment below

# Replacing PatchTST with a Standard Transformer for Code Brevity
# (Real PatchTST requires specific patching logic)
class TransformerBaseline(nn.Module):
    def __init__(self, seq_len, pred_len, n_features):
        super().__init__()
        self.input_proj = nn.Linear(n_features, 64)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, 64))
        enc_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.out_proj = nn.Linear(64 * seq_len, pred_len * n_features)
        self.pred_len = pred_len
        self.n_features = n_features

    def forward(self, x):
        B, L, D = x.shape
        x = self.input_proj(x) + self.pos_enc
        x = self.transformer(x)
        x = x.reshape(B, -1)
        x = self.out_proj(x)
        return x.reshape(B, self.pred_len, self.n_features)


# ==========================================
# 5. DATA LOADING & TRAINING LOOP
# ==========================================

def get_toy_data(n_samples=1000, seq_len=96, pred_len=24, n_features=2):
    # Generates Sine waves with noise
    t = np.linspace(0, 100, n_samples + seq_len + pred_len)
    data = np.sin(t) + np.sin(t * 0.5) + np.random.normal(0, 0.1, size=t.shape)
    # Replicate for features
    data = np.stack([data] * n_features, axis=1) # (Time, Feat)
    
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i : i+seq_len])
        Y.append(data[i+seq_len : i+seq_len+pred_len])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

def train_model(model, train_loader, epochs=5, lr=0.001, name="Model"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining {name}...")
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
            
    train_time = time.time() - start_time
    return train_time

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Settings
    SEQ_LEN = 96  # Lookback
    PRED_LEN = 24 # Forecast
    FEAT = 2      # Variables (e.g. Temp, Load)
    BATCH_SIZE = 32
    
    # 1. Prepare Data (Swap this with pd.read_csv for real data)
    print("Generating Synthetic Data (Simulating ETTh1)...")
    X, Y = get_toy_data(n_samples=2000, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=FEAT)
    
    # Split
    split = int(0.8 * len(X))
    train_data = TensorDataset(X[:split], Y[:split])
    test_data = TensorDataset(X[split:], Y[split:])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # 2. Instantiate Models
    models = {
        "Your_SigNet": SigNetWrapper(SEQ_LEN, FEAT, PRED_LEN, m=16, S=16, N=3),
        "LSTM":        RNNBaseline("LSTM", FEAT, hidden_size=64, pred_len=PRED_LEN),
        "GRU":         RNNBaseline("GRU", FEAT, hidden_size=64, pred_len=PRED_LEN),
        "DLinear":     DLinear(SEQ_LEN, PRED_LEN, FEAT),
        "Transformer": TransformerBaseline(SEQ_LEN, PRED_LEN, FEAT) # Proxy for PatchTST
    }
    
    # 3. Run Comparison
    results = []
    print(f"{'Model':<15} | {'MSE Loss':<10} | {'Time (s)':<10}")
    print("-" * 40)
    
    for name, model in models.items():
        t_time = train_model(model, train_loader, epochs=4, name=name)
        mse = evaluate_model(model, test_loader)
        results.append((name, mse, t_time))
        print(f"{name:<15} | {mse:.5f}    | {t_time:.2f}")

    # 4. Final Summary
    print("\nFinal Leaderboard:")
    results.sort(key=lambda x: x[1]) # Sort by MSE
    for res in results:
        print(f"{res[0]}: MSE={res[1]:.5f}, TrainTime={res[2]:.1f}s")