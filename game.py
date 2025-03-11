import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
TANB = (235,188,117)
TAN = (255,231,193)

class GameAI:

    def __init__(self, w=500, h=500):
        self.w = w
        self.h = h
        self.score = 0
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('2048')
        self.reset()


    def reset(self):
        self.score = 0
        self.arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.arr[random.randint(0, 15)] = 2
        ran = random.randint(0, 15)
        while(self.arr[ran] != 0):
            ran = random.randint(0, 15)
        self.arr[ran] = 2
        self._update_ui()


    def board_value(self):
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
        arr = self.arr  # Flat list of 16 integers
        score = 0

        # 1. Reward empty spaces
        empty_spaces = arr.count(0)
        score += empty_spaces * 30

        # 2. Reward highest tile in a corner
        # 2. Reward highest tile in a corner
        highest_tile = max(arr)
        corner_positions = [0, 3, 12, 15]  # Top-left, top-right, bottom-left, bottom-right
        if arr[15] == highest_tile:  # Prefer bottom-right corner
            score += highest_tile * 25  # Strong incentive for keeping the highest tile in the bottom-right corner
        else:
            # Penalize moving the highest tile out of the bottom-right corner
            score -= highest_tile * 10 if highest_tile not in corner_positions else highest_tile * 5
        # Penalize if it's not in the corner

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


    def play_step(self,action):
        #Did game end
        game_over = False
        move = self._can_move()
        if move == False:
            game_over = True
        # 1. collect user input
        self._update_ui()
        if not game_over:
            for event in pygame.event.get():  # Wait for an event
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            # 2. move
            moved, reward = self._move(action) # update the head
            if(moved):
                self._update_ui()
                ran = random.randint(0, 15)
                while(self.arr[ran] != 0):
                    ran = random.randint(0, 15)
                self.arr[ran] = 2
                reward += self.board_value()                
                self.score += reward
                
                return reward, game_over, self.score
            else:
                 #print("repeat")
                 return -10, game_over, self.score
        return -100, game_over, self.score

    def _update_ui(self):
        self.display.fill((0,0,51))
        pygame.draw.rect(self.display, TANB, (50, 50, self.w - 100, self.h - 100))
        
        for index, box in enumerate(self.arr):
            color = self.get_color(box)
            pos = self.get_position(index)
            pygame.draw.rect(self.display, color, pos)
            if(box != 0):
                text_surface = font.render(str(box), True, (0, 0, 0))  # Black text
                x, y, width, height = pos
                center_x = x + width // 2
                center_y = y + height // 2  
                text_rect = text_surface.get_rect(center=(center_x,center_y))
                self.display.blit(text_surface, text_rect)

        # text = font.render("Score: " + str(self.score), True, WHITE)
        # self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def get_color(self,value):
        if value == 0:
            return TAN  # Default background color for empty tiles
        elif value == 2:
            return (255, 0, 0)  # Red
        elif value == 4:
            return (255, 128, 0)  # Orange
        elif value == 8:
            return (255, 255, 0)  # Yellow
        elif value == 16:
            return (0, 255, 0)  # Green
        elif value == 32:
            return (0, 0, 255)  # Blue
        elif value == 64:
            return (127, 0, 255)  # Purple
        elif value == 128:
            return (255, 153, 153)  # Pink
        elif value == 256:
            return (165, 42, 42)  # Brown
        elif value == 512:
            return (128, 128, 128)  # Gray
        elif value == 1024:
            return (0, 0, 0)  # Black
        elif value == 2048:
            return (255, 255, 255)  # White
        return (153, 200, 255)  # Default color for values beyond 2048

    
    def get_position(self,value):
        if value == 0:
            return (65, 65, 80, 80)
        elif value == 1:
            return (160, 65, 80, 80)
        elif value == 2:
            return (255, 65, 80, 80)
        elif value == 3:
            return (350, 65, 80, 80)
        elif value == 4:
            return (65, 160, 80, 80)
        elif value == 5:
            return (160, 160, 80, 80)
        elif value == 6:
            return (255, 160, 80, 80)
        elif value == 7:
            return (350, 160, 80, 80)
        elif value == 8:
            return (65, 255, 80, 80)
        elif value == 9:
            return (160, 255, 80, 80)
        elif value == 10:
            return (255, 255, 80, 80)
        elif value == 11:
            return (350, 255, 80, 80)
        elif value == 12:
            return (65, 350, 80, 80)
        elif value == 13:
            return (160, 350, 80, 80)
        elif value == 14:
            return (255, 350, 80, 80)
        elif value == 15:
            return (350, 350, 80, 80)
        return TAN  # Default color if value doesn't match
    
    def _move_right(self):
        for i in range(4):
                if(self.arr[2 + i*4] != 0 and self.arr[3 + i*4] == 0):
                    self.arr[3 + i*4] = self.arr[2 + i*4]
                    self.arr[2 + i*4] = 0
                if(self.arr[1 + i*4] != 0 and self.arr[2 + i*4] == 0):
                    if(self.arr[3 + i*4] == 0):
                        self.arr[3 + i*4] = self.arr[1 + i*4]
                        self.arr[1 + i*4] = 0
                    else:
                        self.arr[2 + i*4] = self.arr[1 + i*4]
                        self.arr[1 + i*4] = 0
                if(self.arr[0 + i*4] != 0 and self.arr[1 + i*4] == 0):
                    if(self.arr[3 + i*4] == 0):
                        self.arr[3 + i*4] = self.arr[0 + i*4]
                        self.arr[0 + i*4] = 0
                    elif(self.arr[2 + i*4] == 0):
                        self.arr[2 + i*4] = self.arr[0 + i*4]
                        self.arr[0 + i*4] = 0
                    else:
                        self.arr[1 + i*4] = self.arr[0 + i*4]
                        self.arr[0 + i*4] = 0   
    def _move_left(self):
        for i in range(4):
                if(self.arr[1 + i*4] != 0 and self.arr[0 + i*4] == 0):
                    self.arr[0 + i*4] = self.arr[1 + i*4]
                    self.arr[1 + i*4] = 0
                if(self.arr[2 + i*4] != 0 and self.arr[1 + i*4] == 0):
                    if(self.arr[0 + i*4] == 0):
                        self.arr[0 + i*4] = self.arr[2 + i*4]
                        self.arr[2 + i*4] = 0
                    else:
                        self.arr[1 + i*4] = self.arr[2 + i*4]
                        self.arr[2 + i*4] = 0
                if(self.arr[3 + i*4] != 0 and self.arr[2 + i*4] == 0):
                    if(self.arr[0 + i*4] == 0):
                        self.arr[0 + i*4] = self.arr[3 + i*4]
                        self.arr[3 + i*4] = 0
                    elif(self.arr[1 + i*4] == 0):
                        self.arr[1 + i*4] = self.arr[3 + i*4]
                        self.arr[3 + i*4] = 0
                    else:
                        self.arr[2 + i*4] = self.arr[3 + i*4]
                        self.arr[3 + i*4] = 0   
    def _move_up(self):
        for i in range(4):
                if(self.arr[4 + i] != 0 and self.arr[0 + i] == 0):
                    self.arr[0 + i] = self.arr[4 + i]
                    self.arr[4 + i] = 0
                if(self.arr[8 + i] != 0 and self.arr[4 + i] == 0):
                    if(self.arr[0 + i] == 0):
                        self.arr[0 + i] = self.arr[8 + i]
                        self.arr[8 + i] = 0
                    else:
                        self.arr[4 + i] = self.arr[8 + i]
                        self.arr[8 + i] = 0
                if(self.arr[12 + i] != 0 and self.arr[8 + i] == 0):
                    if(self.arr[0 + i] == 0):
                        self.arr[0 + i] = self.arr[12 + i]
                        self.arr[12 + i] = 0
                    elif(self.arr[4 + i] == 0):
                        self.arr[4 + i] = self.arr[12 + i]
                        self.arr[12 + i] = 0
                    else:
                        self.arr[8 + i] = self.arr[12 + i]
                        self.arr[12 + i] = 0
    def _move_down(self):
        for i in range(4):
                if(self.arr[8 + i] != 0 and self.arr[12 + i] == 0):
                    self.arr[12 + i] = self.arr[8 + i]
                    self.arr[8 + i] = 0
                if(self.arr[4 + i] != 0 and self.arr[8 + i] == 0):
                    if(self.arr[12 + i] == 0):
                        self.arr[12 + i] = self.arr[4 + i]
                        self.arr[4 + i] = 0
                    else:
                        self.arr[8 + i] = self.arr[4 + i]
                        self.arr[4 + i] = 0
                if(self.arr[0 + i] != 0 and self.arr[4 + i] == 0):
                    if(self.arr[12 + i] == 0):
                        self.arr[12 + i] = self.arr[0 + i]
                        self.arr[0 + i] = 0
                    elif(self.arr[8 + i] == 0):
                        self.arr[8 + i] = self.arr[0 + i]
                        self.arr[0 + i] = 0
                    else:
                        self.arr[4 + i] = self.arr[0 + i]
                        self.arr[0 + i] = 0

    def _combine_right(self):
        reward = 0
        for i in range(4):
            if(self.arr[2 + i*4] == self.arr[3 + i*4]):
                    self.arr[3 + i*4] = self.arr[3 + i*4] + self.arr[3 + i*4]
                    self.arr[2 + i*4] = 0
                    reward += self.arr[3 + i*4]
            if(self.arr[1 + i*4] == self.arr[2 + i*4]):
                    self.arr[2 + i*4] = self.arr[2 + i*4] + self.arr[2 + i*4]
                    self.arr[1 + i*4] = 0
                    reward += self.arr[2 + i*4]
            if(self.arr[0 + i*4] == self.arr[1 + i*4]):
                    self.arr[1 + i*4] = self.arr[1 + i*4] + self.arr[1 + i*4]
                    self.arr[0 + i*4] = 0
                    reward += self.arr[1 + i*4]
        return reward
    
    def _combine_left(self):
        reward = 0
        for i in range(4):
            if(self.arr[1 + i*4] == self.arr[0 + i*4]):
                    self.arr[0 + i*4] = self.arr[0 + i*4] + self.arr[0 + i*4]
                    self.arr[1 + i*4] = 0
                    reward += self.arr[0 + i*4]
            if(self.arr[2+ i*4] == self.arr[1 + i*4]):
                    self.arr[1 + i*4] = self.arr[1 + i*4] + self.arr[1 + i*4]
                    self.arr[2 + i*4] = 0
                    reward += self.arr[1 + i*4]
            if(self.arr[3 + i*4] == self.arr[2 + i*4]):
                    self.arr[2 + i*4] = self.arr[2 + i*4] + self.arr[2 + i*4]
                    self.arr[3 + i*4] = 0
                    reward += self.arr[2 + i*4]
        return reward
    
    def _combine_down(self):
        reward = 0
        for i in range(4):
            if(self.arr[8 + i] == self.arr[12 + i]):
                    self.arr[12 + i] = self.arr[12 + i] + self.arr[12 + i]
                    self.arr[8 + i] = 0
                    reward += self.arr[12 + i]
            if(self.arr[4 + i] == self.arr[8 + i]):
                    self.arr[8 + i] = self.arr[8 + i] + self.arr[8 + i]
                    self.arr[4 + i] = 0
                    reward += self.arr[8 + i]
            if(self.arr[0 + i] == self.arr[4 + i]):
                    self.arr[4 + i] = self.arr[4 + i] + self.arr[4 + i]
                    self.arr[0 + i] = 0
                    reward += self.arr[4 + i]
        return reward
    
    def _combine_up(self):
        reward = 0
        for i in range(4):
            if(self.arr[0 + i] == self.arr[4 + i]):
                    self.arr[0 + i] = self.arr[0 + i] + self.arr[0 + i]
                    self.arr[4 + i] = 0
                    reward += self.arr[0 + i]
            if(self.arr[4 + i] == self.arr[8 + i]):
                    self.arr[4 + i] = self.arr[4 + i] + self.arr[4 + i]
                    self.arr[8 + i] = 0
                    reward += self.arr[4 + i]
            if(self.arr[8 + i] == self.arr[12 + i]):
                    self.arr[8 + i] = self.arr[8 + i] + self.arr[8 + i]
                    self.arr[12 + i] = 0
                    reward += self.arr[8 + i]
        return reward
            
    
    def _move(self, action):
        reward = 0
        old_arr = self.arr.copy()
        if action == [1,0,0,0]: # Move Right
            self._move_right()
            reward += self._combine_right()
            self._move_right()   
                
        elif action == [0,1,0,0]: # Move Left
            self._move_left()
            reward += self._combine_left()
            self._move_left()
                
        elif action == [0,0,1,0]: # Move Down
            self._move_down()
            reward += self._combine_down()
            self._move_down()
        else: # Move Up Possible CHANGE
            self._move_up()
            reward += self._combine_up()
            self._move_up()
        if(old_arr == self.arr):
            return False, reward
        else:
            return True, reward
    
    def _can_move(self):
        amount_left = 0
        for box in self.arr:
            if(box == 0):
                amount_left += 1
        if(amount_left > 0):
            return True
        for idx in range(15):
            #Check up
            if(idx >= 4):
                if(self.arr[idx] == self.arr[idx - 4]):
                    amount_left += 1
            #Check left
            if((idx % 4) != 0):
                if(self.arr[idx] == self.arr[idx-1]):
                    amount_left += 1
            #Check Right
            if((idx % 4) != 3):
                if(self.arr[idx] == self.arr[idx+1]):
                    amount_left += 1
            #Check down
            if(idx < 12):
                if(self.arr[idx] == self.arr[idx + 4]):
                    amount_left += 1
        if(amount_left > 0):
            return True
        return False