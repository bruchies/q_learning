import pygame
import numpy as np
import random
import time

ROWS, COLS = 10, 12
ACTIONS = ['up', 'down', 'left', 'right']
ALPHA = 0.1
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

def train():
    for _ in range(EPISODES):
        state = initial_state
        while state != terminal_state:
            action = choose_action(state)
            next_state = get_next_state(state, action)
            r, c = state
            nr, nc = next_state
            reward = reward_matrix[nr][nc]
            best_next = np.max(Q[nr][nc])
            Q[r][c][ACTIONS.index(action)] += ALPHA * (
                reward + GAMMA * best_next - Q[r][c][ACTIONS.index(action)]
            )
            state = next_state

def draw_grid(screen, agent_pos, score):
    block_size = 50
    font = pygame.font.SysFont(None, 24)

    for r in range(ROWS):
        for c in range(COLS):
            reward = reward_matrix[r][c]
            rect = pygame.Rect(c * block_size, r * block_size + 40, block_size, block_size)

            if reward == -999:
                color = (80, 80, 80)
            elif reward == -100:
                color = (0, 0, 0)
            elif (r, c) == terminal_state:
                color = (0, 255, 0)
            elif (r, c) == initial_state:
                color = (200, 50, 50)
            else:
                color = (255, 255, 255)

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (150, 100, 0), rect, 1)

            if reward != -100 and reward != -999 and (r, c) != terminal_state:
                best_action = ACTIONS[np.argmax(Q[r][c])]
                arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}[best_action]
                txt = font.render(arrow, True, (0, 0, 0))
                screen.blit(txt, (c * block_size + 18, r * block_size + 55))

    ar, ac = agent_pos
    pygame.draw.circle(screen, (0, 0, 255), (ac * block_size + 25, ar * block_size + 65), 15)

    score_txt = font.render(f"Pontuação: {int(score)}", True, (255, 255, 255))
    screen.blit(score_txt, (10, 5))

def main():
    train()
    pygame.init()
    screen = pygame.display.set_mode((COLS * 50, ROWS * 50 + 40))
    pygame.display.set_caption("Q-Learning - Mapa em T")
    clock = pygame.time.Clock()

    running = True
    agent_pos = initial_state
    score = 0

    while running:
        screen.fill((0, 0, 0))
        draw_grid(screen, agent_pos, score)
        pygame.display.flip()
        clock.tick(3)

        if agent_pos != terminal_state:
            action = choose_action(agent_pos)
            next_pos = get_next_state(agent_pos, action)
            r, c = next_pos
            score += reward_matrix[r][c]
            agent_pos = next_pos
        else:
            time.sleep(1)
            agent_pos = initial_state
            score = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()