import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




"""
Key changes of this model is 
We want to control the behavior of the exponent matrix better.  To do this, we train an auxiliary parameter K_raw instead of K.
We let K=sigmoid(K_raw). So that K is always positive.
"""



# class ShiftNormalizer(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.register_buffer("a", torch.tensor(0.0))
#        self.register_buffer("b", torch.tensor(0.0))
#
#   def fit(self, x, y):
#      """
#     x: (N, d) or (N,)
#     y: (N, 1) or (N,)
#   """
#  self.a = x.min()
# self.b = y.min()
#
# def transform(self, x, y=None):
#     x_t = x - self.a
#       if y is not None:
#          y_t = y - self.b
#         return x_t, y_t
#    return x_t
#
#   def inverse_y(self, y_t):
#      return y_t + self.b









 # 1. The Model Architecture
class SignatureBlock(nn.Module):
    def __init__(self, d_in, m, S):
        super().__init__()
        self.linear = nn.Linear(d_in, m)
        # Initialize K with small weights to prevent exploding 
        self.K_raw = nn.Parameter(torch.randn(S, m) *0.1)
      

    def forward(self, x):
        z = self.linear(x)
        o = F.softplus(z)
        K=3*F.sigmoid(self.K_raw)
        o_log = torch.log(o)
        log_signature = F.linear(o_log, K)
        signature = torch.exp(log_signature)
        return signature

class SignatureNetwork(nn.Module):
    def __init__(self, d_0, m, S, N, output_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(N):
            input_dim = d_0 if i == 0 else S
            self.blocks.append(SignatureBlock(input_dim, m, S))
        self.final_projection = nn.Linear(S, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_projection(x)
        return x

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
S = 40       # Signature dimension
N = 5       # Number of blocks
output_dim = 1 #  output

model = SignatureNetwork(d_0=input_dim, m=m, S=S, N=N, output_dim=output_dim).to(device)

# Loss and Optimizer
criterion = nn.MSELoss() # Mean Squared Error
#  SGD with Momentum
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)


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
    train_model(num_epochs=500)
    
    # Final check on a few samples
    print("\n--- Final Prediction Check ---")
    x_sample = X_test[:3].to(device)
    y_sample = y_test[:3].to(device)
    with torch.no_grad():
        y_pred = model(x_sample)
    
    for i in range(3):
        print(f"Target: {y_sample[i].item():.2f}, Prediction: {y_pred[i].item():.2f}")