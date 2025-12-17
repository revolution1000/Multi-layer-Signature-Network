import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




"""
Following the design of 2.1. This version 2.3 does not use residual connection. Instead, we return to 
the original definition of signature transform and more fundamentally, the original construction of 
polynomials.
"""





import torch
import torch.nn as nn

class SignatureBlock(nn.Module):
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






# 2. Data Preparation
def generate_data(n_samples=1000, input_dim=10):
    """Generates synthetic regression data: y = sum(x^2) + noise"""
    X = torch.randn(n_samples, input_dim)
    y = torch.sum(X**2, dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return X, y

# Generate data
input_dim = 10
X_raw, y_raw = generate_data(n_samples=2000, input_dim=input_dim)

# Split into Train (80%) and Test (20%)
train_size = int(0.8 * len(X_raw))
X_train, X_test = X_raw[:train_size], X_raw[train_size:]
y_train, y_test = y_raw[:train_size], y_raw[train_size:]

# DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. Setup Training Components 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
m = 200        # Hidden dimension
S = 10       # Signature dimension
N = 5       # Number of blocks
output_dim = 1 #  output
K = torch.zeros(S, m)
for s in range(S):
    num_ones = torch.randint(1, 3, (1,)).item()  # 1 or 2
    idx = torch.randperm(m)[:num_ones]
    K[s, idx] = 1.0

model = SignatureNetwork(d_0=input_dim, m=m, S=S, N=N, output_dim=output_dim, K=K).to(device)

# Loss and Optimizer
criterion = nn.MSELoss() # Mean Squared Error
#  SGD with Momentum
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#4. Training and Testing Loop
def train_model(num_epochs=30):
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # A. Zero the parameter gradients
            optimizer.zero_grad()
            
            # B. Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # C. Backward Pass
            loss.backward()
            
            
            # E. Optimize
            optimizer.step()
            
            running_loss += loss.item()
            
        # Evaluation phase (End of every epoch)
        avg_train_loss = running_loss / len(train_loader)
        test_loss = evaluate_model()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")

def evaluate_model():
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# 5. Run 
if __name__ == "__main__":
    train_model(num_epochs=100)
    
    # Final check on a few samples
    print("\n--- Final Prediction Check ---")
    x_sample = X_test[:3].to(device)
    y_sample = y_test[:3].to(device)
    with torch.no_grad():
        y_pred = model(x_sample)
    
    for i in range(3):
        print(f"Target: {y_sample[i].item():.2f}, Prediction: {y_pred[i].item():.2f}")

        print(K)
