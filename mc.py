import numpy as np
import random
from copy import deepcopy
from game_no_ui import GameAI

# --- Monte Carlo AI Logic ---
class MonteCarloAI:
    def __init__(self, num_simulations=100):
        self.num_simulations = num_simulations

    def simulate_move(self, game, move, rollout_depth=10):
        #Simulate a given move with depth-limited rollouts
        cloned_game = deepcopy(game)
        game_over, score, valid_move = cloned_game.play_step(move)

        if not valid_move:
            return -float('inf')

        total_score = 0
        for _ in range(self.num_simulations):
            cloned_simulation = deepcopy(cloned_game)
            for _ in range(rollout_depth):
                if game_over:
                    break
                random_move = self.get_weighted_random_move(cloned_simulation)
                cloned_simulation.play_step(random_move)
            total_score += self.evaluate_board(cloned_simulation)

        return total_score / self.num_simulations

    def get_weighted_random_move(self, game):
        #Choose a weighted random move based on game state
        move_weights = np.zeros(4)
        for move in range(4):
            cloned_game = deepcopy(game)
            game_over, _, valid_move = cloned_game.play_step(move)
            if valid_move:
                move_weights[move] = max(self.evaluate_board(cloned_game), 0)  # Ensure non-negative weights

        total_weight = np.sum(move_weights)
        if total_weight > 0:
            move_weights /= total_weight  # Normalize weights to probabilities
        else:
            move_weights[:] = 1 / 4  # Equal probabilities if all moves are invalid

        return np.random.choice([0, 1, 2, 3], p=move_weights)

    def evaluate_board(self, game):
        """
        Evaluate the board based on advanced heuristics.
        
        Heuristics include:
        1. Empty spaces
        2. Highest tile in a corner
        3. Smoothness (penalizes sharp differences)
        4. Monotonicity (reward rows/columns that increase or decrease consistently)
        5. Merging potential
        6. Snake pattern alignment
        """
        arr = game.arr  # Flat list of 16 integers
        score = 0

        # 1. Reward empty spaces
        empty_spaces = arr.count(0)
        score += empty_spaces * 30

        # 2. Reward highest tile in a corner
        highest_tile = max(arr)
        if arr[15] == highest_tile:  # Check if the highest tile is in the bottom-right corner
            score += highest_tile * 25
        else:
            score -= highest_tile * 10  # Penalize if it's not in the corner

        # 3. Penalize unsmooth boards
        smoothness = 0
        for i in range(4):
            # Check row smoothness
            row = arr[i * 4:(i + 1) * 4]
            smoothness -= sum(abs(row[j] - row[j + 1]) for j in range(3))
            # Check column smoothness
            col = [arr[i + j * 4] for j in range(4)]
            smoothness -= sum(abs(col[j] - col[j + 1]) for j in range(3))
        score += smoothness * 1  # Low weight since it's secondary to monotonicity

        # 4. Reward monotonicity
        monotonicity = 0
        for i in range(4):
            # Check row monotonicity
            row = arr[i * 4:(i + 1) * 4]
            if all(row[j] <= row[j + 1] for j in range(3)) or all(row[j] >= row[j + 1] for j in range(3)):
                monotonicity += highest_tile * 12
            # Check column monotonicity
            col = [arr[i + j * 4] for j in range(4)]
            if all(col[j] <= col[j + 1] for j in range(3)) or all(col[j] >= col[j + 1] for j in range(3)):
                monotonicity += highest_tile * 12
        score += monotonicity

        # 5. Reward tile merging potential
        merge_reward = 0
        for i in range(16):
            if i % 4 != 3:  # Check horizontally
                if arr[i] == arr[i + 1]:
                    merge_reward += arr[i]
            if i < 12:  # Check vertically
                if arr[i] == arr[i + 4]:
                    merge_reward += arr[i]
        score += merge_reward * 8

        # 6. Reward snake pattern alignment
        snake_pattern = [
            15, 14, 13, 12,  # Bottom row: right-to-left
            8, 9, 10, 11,    # Second row: left-to-right
            7, 6, 5, 4,      # Third row: right-to-left
            0, 1, 2, 3       # Top row: left-to-right
        ]

        snake_score = 0
        for i in range(15):  # Iterate through the snake pattern
            if arr[snake_pattern[i]] >= arr[snake_pattern[i + 1]]:  # Ensure descending or equal values
                snake_score += arr[snake_pattern[i]]
            else:
                break  # Stop if the snake pattern breaks
        score += snake_score * 100  # High weight to prioritize snake alignment

        return score


    def get_dynamic_simulations(self, game):
        #Dynamically adjust simulations based on the board's state.
        empty_tiles = game.arr.count(0)
        return min(800, max(200, (16 - empty_tiles) * 20))

    def get_best_move(self, game):
        #Evaluate all possible moves and return the best one.
        scores = []
        for move in range(4):  # up, down, left, right
            avg_score = self.simulate_move(game, move)
            scores.append(avg_score)
        return np.argmax(scores)  # Move with the highest average score

def main():
    game = GameAI()
    ai = MonteCarloAI()

    print("Starting 2048 with Monte Carlo AI!")
    done = False
    while not done:
        ai.num_simulations = ai.get_dynamic_simulations(game)
        best_move = ai.get_best_move(game)
        done, score, moved = game.play_step(best_move)
        print(game.arr)

    print("Game Over!")
    print("Final Score:", score)
    print("Final Board:")
    print(game.arr[0], " ", game.arr[1], " ", game.arr[2], " ", game.arr[3])
    print(game.arr[4], " ", game.arr[5], " ", game.arr[6], " ", game.arr[7])
    print(game.arr[8], " ", game.arr[9], " ", game.arr[10], " ", game.arr[11])
    print(game.arr[12], " ", game.arr[13], " ", game.arr[14], " ", game.arr[15])
        

if __name__ == '__main__':
    main()
