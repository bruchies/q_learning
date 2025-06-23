import numpy as np
import random
from config import ROWS, COLS, ACTIONS, EPSILON, GAMMA, reward_matrix, terminal_state

Q = np.zeros((ROWS, COLS, len(ACTIONS)))

def is_valid(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS and reward_matrix[r][c] != -999

def get_next_state(state, action):
    r, c = state
    delta = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    dr, dc = delta[action]
    candidate = (r + dr, c + dc)
    return candidate if is_valid(candidate) else state

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    r, c = state
    return ACTIONS[np.argmax(Q[r][c])]

def update_q(state, action, reward, next_state):
    r, c = state
    nr, nc = next_state
    Q[r][c][ACTIONS.index(action)] = reward + GAMMA * np.max(Q[nr][nc])
