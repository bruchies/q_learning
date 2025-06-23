import pygame
from config import EPISODES, reward_matrix, initial_state, terminal_state
from qlearning import choose_action, get_next_state, update_q
from gui import draw_grid

def main():
    pygame.init()
    screen = pygame.display.set_mode(((12 + 6) * 50, 10 * 50 + 40))
    pygame.display.set_caption("Q-Learning - Mapa em T")
    clock = pygame.time.Clock()

    state = initial_state
    score = 0
    episode = 0
    view_mode = 'arrows'

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                view_mode = 'heatmap' if view_mode == 'arrows' else 'arrows'

        if episode >= EPISODES:
            screen.fill((0, 0, 0))
            draw_grid(screen, state, score, view_mode)
            pygame.display.flip()
            continue

        screen.fill((0, 0, 0))
        draw_grid(screen, state, score, view_mode)

        if state != terminal_state:
            action = choose_action(state)
            next_state = get_next_state(state, action)
            reward = reward_matrix[next_state[0]][next_state[1]]
            update_q(state, action, reward, next_state)
            score += reward
            state = next_state
        else:
            state = initial_state
            score = 0
            episode += 1

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()
