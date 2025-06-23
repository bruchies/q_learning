import pygame
import numpy as np
import random
import time

ROWS, COLS = 10, 12
ACTIONS = ['up', 'down', 'left', 'right']
GAMMA = 0.9
EPSILON = 0.3
EPISODES = 1000

reward_matrix = np.array([
    [1, 1, 1, 1, -100, 1, 1, 1, 1, 1, 1, -100],
    [1, -100, 1, 1, 1, 1, 1, 1, 1, 1, 1, -100],
    [-100, 1, -100, 1, 1, 1, -100, 1, -100, -100, 1, 1],
    [1, -100, 1, 1, 1, 1, 1, 1, -100, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
    [-999, -999, -999, -999, 1, 1, -100, 1, -999, -999, -999, -999],
    [-999, -999, -999, -999, 1, 1, 1, 1, -999, -999, -999, -999],
    [-999, -999, -999, -999, 1, 1, 1, 1, -999, -999, -999, -999],
    [-999, -999, -999, -999, 1, 1, -100, 1, -999, -999, -999, -999],
    [-999, -999, -999, -999, 1, 1, 1, 1, -999, -999, -999, -999],
])

initial_state = (9, 4)
terminal_state = (4, 11)

Q = np.zeros((ROWS, COLS, len(ACTIONS)))

def is_valid(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS and reward_matrix[r][c] != -999

def get_next_state(state, action):
    r, c = state
    candidate = {
        'up': (r - 1, c),
        'down': (r + 1, c),
        'left': (r, c - 1),
        'right': (r, c + 1)
    }[action]
    return candidate if is_valid(candidate) else state

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    r, c = state
    return ACTIONS[np.argmax(Q[r][c])]

def draw_grid(screen, agent_pos, score, view_mode):
    block_size = 50
    font = pygame.font.SysFont(None, 24)

    # --- Lógica para o Mapa de Calor ---
    if view_mode == 'heatmap':
        q_values = [np.max(Q[r][c]) for r in range(ROWS) for c in range(COLS) if reward_matrix[r][c] > -999 and (r,c) != terminal_state]
        min_q = min(q_values) if q_values else 0
        max_q = max(q_values) if q_values else 0
        delta = (max_q - min_q) if max_q > min_q else 1.0
    # --- Fim da lógica do Mapa de Calor ---

    for r in range(ROWS):
        for c in range(COLS):
            reward = reward_matrix[r][c]
            rect = pygame.Rect(c * block_size, r * block_size + 40, block_size, block_size)

            color = (255, 255, 255) # Cor padrão
            if view_mode == 'arrows':
                if reward == -999: color = (80, 80, 80)
                elif reward == -100: color = (0, 0, 0)
                elif (r, c) == terminal_state: color = (0, 255, 0)
                elif (r, c) == initial_state: color = (200, 50, 50)
                else: color = (255, 255, 255)
            
            elif view_mode == 'heatmap':
                if reward == -999: color = (80, 80, 80)
                elif reward == -100: color = (0, 0, 0)
                elif (r, c) == terminal_state: color = (0, 255, 0)
                else:
                    q_val = np.max(Q[r][c])
                    norm_q = (q_val - min_q) / delta
                    red = int(255 * norm_q)
                    blue = int(255 * (1 - norm_q))
                    color = (max(0, min(255, red)), 0, max(0, min(255, blue)))

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

    # tabela q no lado direito
    q_font = pygame.font.SysFont(None, 18)
    q_table_x_start = COLS * 50 + 10
    q_table_y_start = 45
    text_height = 18

    valid_states = []
    for r in range(ROWS):
        for c in range(COLS):
            if reward_matrix[r][c] != -999:
                valid_states.append((r, c))

    num_states = len(valid_states)
    num_cols = 3
    col_len = (num_states + num_cols - 1) // num_cols

    for i, state in enumerate(valid_states):
        r, c = state
        
        col_idx = i // col_len
        row_idx = i % col_len
        
        q_value = np.max(Q[r][c])
        q_txt = q_font.render(f"Q({r},{c})={q_value:.1f}", True, (255, 255, 255))
        
        x_pos = q_table_x_start + col_idx * 100
        y_pos = q_table_y_start + row_idx * text_height
        
        screen.blit(q_txt, (x_pos, y_pos))

def main():
    pygame.init()
    screen = pygame.display.set_mode(((COLS + 6) * 50, ROWS * 50 + 40))
    pygame.display.set_caption("Q-Learning - Treinamento")
    clock = pygame.time.Clock()

    state = initial_state
    score = 0
    episode = 0
    view_mode = 'arrows' # 'arrows' ou 'heatmap'

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    view_mode = 'heatmap' if view_mode == 'arrows' else 'arrows'

        if episode >= EPISODES:
            screen.fill((0,0,0))
            draw_grid(screen, state, score, view_mode)
            font = pygame.font.SysFont(None, 40)
            text = font.render("Treinamento Concluído", True, (255, 255, 0))
            text_rect = text.get_rect(center=(screen.get_width() / 2, 20))
            screen.blit(text, text_rect)
            pygame.display.flip()
            continue

        screen.fill((0, 0, 0))
        draw_grid(screen, state, score, view_mode)
        
        # Texto de ajuda para o modo de visualização
        font_help = pygame.font.SysFont(None, 24)
        mode_text_str = f"Modo: {'Setas' if view_mode == 'arrows' else 'Mapa de Calor'} (Pressione ESPAÇO)"
        mode_text = font_help.render(mode_text_str, True, (255, 255, 255))
        screen.blit(mode_text, (screen.get_width() - 350, 5))

        # Q-learning step
        if state != terminal_state:
            action = choose_action(state)
            next_state = get_next_state(state, action)
            
            r, c = state
            nr, nc = next_state
            
            reward = reward_matrix[nr][nc]
            best_next = np.max(Q[nr][nc])
            
            # Atualização da Tabela Q
            Q[r][c][ACTIONS.index(action)] = reward + GAMMA * best_next
            
            score += reward
            state = next_state
        else:
            # Reinicia o episódio
            state = initial_state
            score = 0
            episode += 1
        
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()