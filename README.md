# RL for 2048

This project implements a Deep Q-Network (DQN) using PyTorch for reinforcement learning of the game 2048.

## Features  

- **2048 Game Environment**: Fully functional 2048 game implemented in Python, serving as the environment for training the AI agent.  
- **Deep Q-Network (DQN) Agent**: Utilizes a neural network to learn optimal moves based on past game experiences.  
- **Heuristic-Based Reward System**: Encourages merging high-value tiles and maintaining a structured board layout to improve gameplay efficiency.  
- **Experience Replay**: Stores past game states and samples mini-batches to improve training stability and reduce correlation between consecutive moves.  
- **PyTorch-Based Neural Network**: Implements a multi-layer feedforward network using PyTorch for decision-making.  
- **Exploration vs. Exploitation**: Balances random moves and learned strategies using an epsilon-greedy approach to refine gameplay over time.  
- **Training with Discount Factor**: Uses a gamma discount factor to prioritize long-term rewards over immediate gains, leading to smarter moves.  
- **Performance Tracking**: Logs scores and highest tiles achieved, plotting performance trends over multiple games.  

