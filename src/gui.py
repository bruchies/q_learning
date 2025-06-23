import pygame
import numpy as np
from config import ROWS, COLS, ACTIONS, reward_matrix, initial_state, terminal_state, visit_frequency
from qlearning import Q

def get_optimal_path():
    path = []
    state = initial_state
    visited = set()
    while state != terminal_state and state not in visited:
        path.append(state)
        visited.add(state)
        r, c = state
        best_action_idx = np.argmax(Q[r][c])
        action = ACTIONS[best_action_idx]
        # Próximo estado
        if action == 'up': next_state = (r-1, c)
        elif action == 'down': next_state = (r+1, c)
        elif action == 'left': next_state = (r, c-1)
        elif action == 'right': next_state = (r, c+1)
        else: break
        # Verifica se é válido
        if (0 <= next_state[0] < ROWS and 0 <= next_state[1] < COLS and reward_matrix[next_state[0]][next_state[1]] != -999):
            state = next_state
        else:
            break
    path.append(state)  # Adiciona terminal
    return path

def draw_grid(screen, agent_pos, score, view_mode):
    block_size = 50
    font = pygame.font.SysFont(None, 24)

    if view_mode == 'heatmap':
        visit_values = [visit_frequency[r][c] for r in range(ROWS) for c in range(COLS) if reward_matrix[r][c] > -999]
        min_visits = min(visit_values) if visit_values else 0
        max_visits = max(visit_values) if visit_values else 0
        delta = (max_visits - min_visits) if max_visits > min_visits else 1.0

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
                visit_count = visit_frequency[r][c]
                norm_visits = (visit_count - min_visits) / delta
                red = int(255 * norm_visits)
                blue = int(255 * (1 - norm_visits))
                color = (red, 0, blue)
            # No modo optimal_path, apenas desenhe o grid normalmente

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (150, 100, 0), rect, 1)

            if view_mode == 'arrows' and reward > -999 and (r, c) != terminal_state:
                best_action = ACTIONS[np.argmax(Q[r][c])]
                arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}[best_action]
                txt = font.render(arrow, True, (0, 0, 0))
                screen.blit(txt, (c * block_size + 18, r * block_size + 55))

    # Desenhar o caminho ótimo no modo heatmap ou optimal_path
    if view_mode == 'heatmap' or view_mode == 'optimal_path':
        path = get_optimal_path()
        for idx in range(len(path)-1):
            r1, c1 = path[idx]
            r2, c2 = path[idx+1]
            x1 = c1 * block_size + block_size//2
            y1 = r1 * block_size + 40 + block_size//2
            x2 = c2 * block_size + block_size//2
            y2 = r2 * block_size + 40 + block_size//2
            pygame.draw.line(screen, (255, 255, 0), (x1, y1), (x2, y2), 6)

    ar, ac = agent_pos
    pygame.draw.circle(screen, (0, 0, 255), (ac * block_size + 25, ar * block_size + 65), 15)

    score_txt = font.render(f"Pontuação: {int(score)}", True, (255, 255, 255))
    screen.blit(score_txt, (10, 5)) 

    # Adicionar legenda para o modo
    if view_mode == 'heatmap':
        legend_txt = font.render("Heatmap: Frequência de Visitas + Caminho Ótimo (amarelo)", True, (255, 255, 255))
        screen.blit(legend_txt, (200, 5))
    elif view_mode == 'optimal_path':
        legend_txt = font.render("Caminho Ótimo (amarelo) - Tecla C", True, (255, 255, 0))
        screen.blit(legend_txt, (200, 5))
    else:
        legend_txt = font.render("Modo: Setas de Direção", True, (255, 255, 255))
        screen.blit(legend_txt, (200, 5))

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
