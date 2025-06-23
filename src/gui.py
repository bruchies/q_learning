import pygame
import numpy as np
from config import ROWS, COLS, ACTIONS, reward_matrix, initial_state, terminal_state
from qlearning import Q

def draw_grid(screen, agent_pos, score, view_mode):
    block_size = 50
    font = pygame.font.SysFont(None, 24)

    if view_mode == 'heatmap':
        q_values = [np.max(Q[r][c]) for r in range(ROWS) for c in range(COLS) if reward_matrix[r][c] > -999 and (r,c) != terminal_state]
        min_q = min(q_values) if q_values else 0
        max_q = max(q_values) if q_values else 0
        delta = (max_q - min_q) if max_q > min_q else 1.0

    for r in range(ROWS):
        for c in range(COLS):
            reward = reward_matrix[r][c]
            rect = pygame.Rect(c * block_size, r * block_size + 40, block_size, block_size)
            color = (255, 255, 255)

            if reward == -999: color = (80, 80, 80)
            elif reward == -100: color = (0, 0, 0)
            elif (r, c) == terminal_state: color = (0, 255, 0)
            elif (r, c) == initial_state: color = (200, 50, 50)
            elif view_mode == 'heatmap':
                q_val = np.max(Q[r][c])
                norm_q = (q_val - min_q) / delta
                red = int(255 * norm_q)
                blue = int(255 * (1 - norm_q))
                color = (red, 0, blue)

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (150, 100, 0), rect, 1)

            if view_mode == 'arrows' and reward > -999 and (r, c) != terminal_state:
                best_action = ACTIONS[np.argmax(Q[r][c])]
                arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}[best_action]
                txt = font.render(arrow, True, (0, 0, 0))
                screen.blit(txt, (c * block_size + 18, r * block_size + 55))

    ar, ac = agent_pos
    pygame.draw.circle(screen, (0, 0, 255), (ac * block_size + 25, ar * block_size + 65), 15)

    score_txt = font.render(f"Pontuação: {int(score)}", True, (255, 255, 255))
    screen.blit(score_txt, (10, 5)) 

    # Q-table ao lado
    q_font = pygame.font.SysFont(None, 18)
    q_table_x_start = COLS * 50 + 10
    q_table_y_start = 45
    text_height = 18
    valid_states = [(r, c) for r in range(ROWS) for c in range(COLS) if reward_matrix[r][c] != -999]
    col_len = (len(valid_states) + 2) // 3

    for i, (r, c) in enumerate(valid_states):
        col_idx = i // col_len
        row_idx = i % col_len
        q_value = np.max(Q[r][c])
        q_txt = q_font.render(f"Q({r},{c})={q_value:.1f}", True, (255, 255, 255))
        screen.blit(q_txt, (q_table_x_start + col_idx * 100, q_table_y_start + row_idx * text_height))
