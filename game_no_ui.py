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
        self.reset()


    def reset(self):
        self.score = 0
        self.arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.arr[random.randint(0, 15)] = 2
        ran = random.randint(0, 15)
        while(self.arr[ran] != 0):
            ran = random.randint(0, 15)
        self.arr[ran] = 2



    def play_step(self,action):
        #Did game end
        game_over = False
        move = self._can_move()
        if move == False:
            game_over = True
        # 1. collect user input
        if not game_over:
            for event in pygame.event.get():  # Wait for an event
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            # 2. move
            moved, reward = self._move(action) # update the head
            if(moved):
                ran = random.randint(0, 15)
                while(self.arr[ran] != 0):
                    ran = random.randint(0, 15)
                self.arr[ran] = 2                
                self.score += reward
                # if(action == 1 or action == 3):
                #     self.score += 5
                # elif(action == 2):
                #      self.score += 0
                # else:
                #     self.score -= 5

                return game_over, self.score, True
            else:
                 return game_over, self.score, False
        return game_over, self.score, False



    
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
        if action == 1: # Move Right
            self._move_right()
            reward += self._combine_right()
            self._move_right()   
                
        elif action == 2: # Move Left
            self._move_left()
            reward += self._combine_left()
            self._move_left()
                
        elif action == 3: # Move Down
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
