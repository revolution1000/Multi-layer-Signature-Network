import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym # Requires: pip install gym

# ==========================================
# 1. YOUR SIGNATURE NETWORK (UNCHANGED)
# ==========================================
class SignatureBlock(nn.Module):
    def __init__(self, d_in, m, S, K):
        super().__init__()
        self.linear = nn.Linear(d_in, m)
        self.register_buffer("K", K)

    def forward(self, x):
        z = self.linear(x)
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

# ==========================================
# 2. BASELINE MLP POLICY
# ==========================================
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class SignaturePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Config
        m = 32
        S = 32
        N = 2
        # Sparse Binary K
        K_vals = torch.zeros(S, m)
        # Random sparse connections
        indices = torch.randperm(S*m)[:int(S*m*0.2)]
        K_vals[indices // m, indices % m] = 1.0
        
        # We need Softmax at the end for probabilities
        self.net = SignatureNetwork(obs_dim, m, S, N, act_dim, K_vals)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.net(x)
        return self.softmax(logits)

# ==========================================
# 3. REINFORCE AGENT
# ==========================================
class REINFORCE:
    def __init__(self, policy_net, lr=1e-3):
        self.policy = policy_net
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs):
        R = 0
        policy_loss = []
        returns = []
        # Calculate discounted returns
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        # Normalize returns (Critical for stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        # Gradient clipping for SigNet stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

# ==========================================
# 4. EXPERIMENT LOOP
# ==========================================
def run_rl_experiment(episodes=500):
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # 1. Train MLP
    print("\n--- Training Baseline MLP ---")
    mlp_policy = MLPPolicy(obs_dim, act_dim)
    agent_mlp = REINFORCE(mlp_policy)
    
    scores_mlp = []
    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0] # Gym version compat
        log_probs = []
        rewards = []
        done = False
        while not done:
            action, log_prob = agent_mlp.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        agent_mlp.update(rewards, log_probs)
        scores_mlp.append(sum(rewards))
        if ep % 50 == 0:
            print(f"Episode {ep} | Reward: {sum(rewards)}")

    # 2. Train SignatureNet
    print("\n--- Training Signature Policy ---")
    sig_policy = SignaturePolicy(obs_dim, act_dim)
    agent_sig = REINFORCE(sig_policy, lr=0.0005) # Lower LR often helps polynomial nets
    
    scores_sig = []
    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        log_probs = []
        rewards = []
        done = False
        while not done:
            action, log_prob = agent_sig.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        agent_sig.update(rewards, log_probs)
        scores_sig.append(sum(rewards))
        if ep % 50 == 0:
            print(f"Episode {ep} | Reward: {sum(rewards)}")
            
    print("\nFinal Results (Average Last 50 Episodes):")
    print(f"MLP: {np.mean(scores_mlp[-50:]):.1f}")
    print(f"Sig: {np.mean(scores_sig[-50:]):.1f}")

if __name__ == "__main__":
    run_rl_experiment()