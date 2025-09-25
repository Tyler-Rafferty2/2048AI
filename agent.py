import torch
import random
import numpy as np
from collections import deque
from game import GameAI
from model import Linear_QNet, QTrainer
from helper import plot
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(16, [512, 256], 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """Normalize the game state for better neural network training"""
        game_state = np.array(game.arr, dtype=np.float32)
        
        # Log-scale normalization to handle exponential growth of 2048 tiles
        # Convert 0->0, 2->1, 4->2, 8->3, 16->4, etc.
        normalized_state = np.zeros_like(game_state)
        non_zero_mask = game_state > 0
        normalized_state[non_zero_mask] = np.log2(game_state[non_zero_mask])
        
        # Scale to roughly [-1, 1] range (assuming max tile around 2048)
        normalized_state = normalized_state / 11.0  # log2(2048) = 11
        
        return normalized_state.tolist()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Improved epsilon-greedy action selection"""
        # More gradual epsilon decay
        epsilon_min = 0.01
        epsilon_decay = 0.995
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
        
        final_move = [0,0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def calculate_reward(self, old_arr, new_arr, merge_score, moved, game_over):
        """Improved reward function focused on key game mechanics"""
        if game_over:
            return -50  # Moderate penalty for game over
            
        if not moved:
            return -5   # Small penalty for invalid moves
        
        reward = 0
        
        # 1. Primary reward: merge score (actual points gained)
        reward += merge_score
        
        # 2. Secondary reward: empty cells (encourages keeping board open)
        empty_cells = new_arr.count(0)
        reward += empty_cells * 2
        
        # 3. Positional bonus: highest tile in corner
        highest_tile = max(new_arr)
        corners = [new_arr[0], new_arr[3], new_arr[12], new_arr[15]]
        if highest_tile in corners:
            reward += 10
        
        # 4. Small bonus for creating higher tiles
        max_old = max(old_arr)
        max_new = max(new_arr)
        if max_new > max_old:
            reward += max_new * 0.1
            
        return reward


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = GameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)
        old_arr = game.arr.copy()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        merge_reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Calculate improved reward
        moved = old_arr != game.arr
        reward = agent.calculate_reward(old_arr, game.arr, merge_reward, moved, done)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            agent.n_games += 1
            highest_tile = max(game.arr)
            
            print(f'Game {agent.n_games}: Score {score}, Highest Tile: {highest_tile}, Record: {record}')
            print(f'Epsilon: {agent.epsilon:.4f}')
            
            plot_scores.append(highest_tile)
            total_score += highest_tile
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            game.reset()
            agent.train_long_memory()

            if highest_tile > record:
                record = highest_tile
                # Optionally save model when achieving new record
                # agent.model.save()

if __name__ == '__main__':
    train()