import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)  # Outputs a single state value
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)  # Outputs Q-values for each action
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantages = self.advantage_layer(features)

        # Handle single-batch case for mean computation
        if advantages.dim() == 1:  # Single sample (1D tensor)
            advantages_mean = advantages.mean(dim=0, keepdim=True)  # Mean over actions
        else:  # Batch case (2D tensor)
            advantages_mean = advantages.mean(dim=1, keepdim=True)

        q_values = value + (advantages - advantages_mean)
        return q_values



    # def save(self, file_name='model.pth'):
    #     model_folder_path = './model'
    #     if not os.path.exists(model_folder_path):
    #         os.makedirs(model_folder_path)

    #     file_name = os.path.join(model_folder_path, file_name)
    #     torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, trace_decay=0.9):
        #print(done)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # eligibility_traces = [torch.zeros_like(p) for p in self.model.parameters()]  # Initialize traces
        # for idx in range(len(done)):  # Update for each sample in the batch
        #     for param, trace in zip(self.model.parameters(), eligibility_traces):
        #         trace.data = trace_decay * self.gamma * trace.data + param.grad.data
        #         param.data += self.lr * trace  # Adjust parameters using eligibility traces
        
        self.optimizer.step()


