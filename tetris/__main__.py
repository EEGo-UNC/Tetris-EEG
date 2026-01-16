from random import randint
import pygame
from .settings import *
from .difficulties import *
from .game import Game
from experiment import experiment as exp
import time
import os
import random

seed_value = int.from_bytes(os.urandom(8), byteorder='big')
random.seed(seed_value)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Tetris (Modular)")

font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()

difficulties = [
    #increase_difficulty_lines_cleared,
    # constant_difficulty,
    # increase_difficulty_adaptive,
    # increase_difficulty_blocks_placed,
    increase_difficulty_minimize_emotion_distance
]

difficulty = randint(0, len(difficulties) - 1)
increase_difficulty = difficulties[difficulty]

game = Game(difficulty_level=difficulty)


def draw_board(screen, board, colors):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            color = COLORS[colors[y, x]] if colors[y, x] else COLORS["bg"]
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, COLORS["grid"], rect, 1)


def draw_piece(screen, piece, offset_x=0, offset_y=0):
    for x, y in piece.get_coords():
        if y >= 0:
            rect = pygame.Rect(
                (x * BLOCK_SIZE) + offset_x,
                (y * BLOCK_SIZE) + offset_y,
                BLOCK_SIZE,
                BLOCK_SIZE,
            )
            pygame.draw.rect(screen, COLORS[piece.type], rect)
            pygame.draw.rect(screen, COLORS["border"], rect, 2)


def draw_mini_piece(screen, piece, pos_x, pos_y):
    # Draws a small preview of a piece at pos_x, pos_y
    shape = piece.shape
    for dy, row in enumerate(shape):
        for dx, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    pos_x + dx * BLOCK_SIZE // 2,
                    pos_y + dy * BLOCK_SIZE // 2,
                    BLOCK_SIZE // 2,
                    BLOCK_SIZE // 2,
                )
                pygame.draw.rect(screen, COLORS[piece.type], rect)
                pygame.draw.rect(screen, COLORS["border"], rect, 2)


def main():
    exp.set_global_session_id()
    exp.init_epoc_record()

    time.sleep(5)
    user_id = int(input("1-Vitor, 2-Nick, 3-Vish, 4-Jayasri: "))
    exp.EpocX.pow_data_batch.drop(exp.EpocX.pow_data_batch.index, inplace=True)

    fall_time = 0
    fall_speed = 450  # ms
    min_fall_speed = 50  # ms
    running = True
    move_delay = 80  # ms between moves when holding
    move_timer = 0
    hold_key_pressed = False
    while running:
        start_time = time.time()
        dt = clock.tick(60)
        fall_time += dt
        move_timer += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and not game.game_over:
                if event.key == pygame.K_UP:
                    game.rotate()
                # elif event.key == pygame.K_SPACE:
                #     game.hard_drop()
                # elif event.key == pygame.K_c and not hold_key_pressed:
                #     game.hold()
                #     hold_key_pressed = True
            # Rotate does not work if held down
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_c:
                    hold_key_pressed = False

        keys = pygame.key.get_pressed()
        if not game.game_over:
            if move_timer > move_delay:
                if keys[pygame.K_LEFT]:
                    game.move(-1, 0)
                if keys[pygame.K_RIGHT]:
                    game.move(1, 0)
                # if keys[pygame.K_DOWN]:
                #     game.drop()

                move_timer = 0


        if fall_time > fall_speed and not game.game_over:
            arousal, valence = game.tick(user_id, start_time, fall_speed, increase_difficulty.__name__)
            fall_time = 0

        if increase_difficulty is increase_difficulty and arousal is not None and valence is not None:   
            fall_speed = increase_difficulty(game, arousal, valence)
            game.fall_speed = fall_speed

        screen.fill(COLORS["bg"])
        board_grid, color_grid = game.board.get_state()
        draw_board(screen, board_grid, color_grid)
        # Draw current piece
        if not game.game_over:
            draw_piece(screen, game.current_piece)
        # Draw next piece preview
        next_label = font.render("Next", True, COLORS["text"])
        screen.blit(next_label, (GRID_WIDTH * BLOCK_SIZE + 30, 30))
        draw_mini_piece(screen, game.next_piece, GRID_WIDTH * BLOCK_SIZE + 30, 60)
        # Draw hold piece preview
        hold_label = font.render("Hold (C)", True, COLORS["text"])
        screen.blit(hold_label, (GRID_WIDTH * BLOCK_SIZE + 30, 140))
        if game.hold_piece:
            draw_mini_piece(screen, game.hold_piece, GRID_WIDTH * BLOCK_SIZE + 30, 170)
        # Draw score
        score_surface = font.render(f"Score: {game.score}", True, COLORS["text"])
        screen.blit(score_surface, (GRID_WIDTH * BLOCK_SIZE + 30, 250))
        # Draw lines cleared
        lines_surface = font.render(
            f"Lines: {game.lines_cleared}", True, COLORS["text"]
        )
        screen.blit(lines_surface, (GRID_WIDTH * BLOCK_SIZE + 30, 280))
        if game.game_over:
            over_surface = font.render("GAME OVER", True, (255, 0, 0))
            screen.blit(over_surface, (WINDOW_WIDTH // 2 - 80, WINDOW_HEIGHT // 2))

        pygame.display.flip()

    pygame.quit()

    while True:
        save_session_recordings = input("Do you want to save session recordings? [Y/N] ")
        if save_session_recordings.lower() == "y":
            exp.save_curr_sesh("dreamer_models/datasets/EEGO.csv", "dreamer_models/datasets/curr_sesh.csv")
            break
        elif save_session_recordings.lower() == "n":
            sure = input("Are you sure?[Y/N] ")
            if sure.lower() == "y":
                break


if __name__ == "__main__":
    main()
