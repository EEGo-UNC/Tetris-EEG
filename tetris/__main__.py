from random import randint
import pygame
from .settings import *
from .difficulties import *
from .game import Game
# from experiment import experiment as exp
from experiment import experiment_muse as exp
import time
import os
import random

seed_value = int.from_bytes(os.urandom(8), byteorder="big")
random.seed(seed_value)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Tetris (Modular)")

font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()

difficulties = [
    # increase_difficulty_lines_cleared,
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


def clamp_int(v, lo, hi):
    return max(lo, min(hi, int(v)))


def draw_valence_arousal_map_binary(
    screen,
    origin_x,
    origin_y,
    w=200,
    h=200,
    valence=None,
    arousal=None,
    center=2.5,
    dot_r=30,
):
    """
    Binary VA map:
      - Anything > center is High
      - Anything <= center is Low
    Accepts numeric (1..5), strings ("high"/"low"), or 0/1.
    No grid lines.
    Dot center is clamped so large radii don't get clipped (looks like a line).
    """

    panel = pygame.Rect(origin_x, origin_y, w, h)
    pygame.draw.rect(screen, (20, 20, 20), panel, border_radius=10)
    pygame.draw.rect(screen, COLORS["grid"], panel, 2, border_radius=10)

    pad = 16
    plot = pygame.Rect(origin_x + pad, origin_y + pad, w - 2 * pad, h - 2 * pad)

    cx = plot.x + plot.w // 2
    cy = plot.y + plot.h // 2
    pygame.draw.line(screen, (90, 90, 90), (cx, plot.y), (cx, plot.y + plot.h), 3)
    pygame.draw.line(screen, (90, 90, 90), (plot.x, cy), (plot.x + plot.w, cy), 3)

    title = font.render("Emotion Map", True, COLORS["text"])
    screen.blit(title, (origin_x, origin_y - 28))

    small = pygame.font.SysFont("Arial", 16)
    screen.blit(small.render("A High", True, (190, 190, 190)), (plot.x + 6, plot.y + 4))
    screen.blit(small.render("V High", True, (190, 190, 190)), (plot.x + plot.w - 58, plot.y + plot.h - 20))

    def to_bin(x):
        if x is None:
            return None

        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("high", "h", "1", "true", "t"):
                return True
            if s in ("low", "l", "0", "false", "f"):
                return False
            return None

        try:
            fx = float(x)
        except Exception:
            return None

        if fx in (0.0, 1.0):
            return bool(int(fx))

        return fx > center

    def bin_to_screen(v_high, a_high):
        vx = 0.25 if not v_high else 0.75
        ay = 0.25 if a_high else 0.75  # higher arousal => up

        sx = plot.x + vx * plot.w
        sy = plot.y + ay * plot.h

        margin = dot_r + 2
        sx = clamp_int(sx, plot.x + margin, plot.x + plot.w - margin)
        sy = clamp_int(sy, plot.y + margin, plot.y + plot.h - margin)
        return sx, sy

    v_bin = to_bin(valence)
    a_bin = to_bin(arousal)

    if v_bin is not None and a_bin is not None:
        px, py = bin_to_screen(v_bin, a_bin)

        BLUE = (0, 150, 255)
        pygame.draw.circle(screen, (0, 80, 140), (px, py), dot_r + 8)  # glow
        pygame.draw.circle(screen, BLUE, (px, py), dot_r)              # core
        pygame.draw.circle(screen, (0, 0, 0), (px, py), dot_r, 3)      # outline

        v_label = "High" if v_bin else "Low"
        a_label = "High" if a_bin else "Low"
        readout = font.render(f"V: {v_label}  A: {a_label}", True, COLORS["text"])
        screen.blit(readout, (origin_x, origin_y + h + 12))
    else:
        hint = pygame.font.SysFont("Arial", 18).render("Waiting for VA…", True, (160, 160, 160))
        screen.blit(hint, (origin_x, origin_y + h + 14))


def main():
    exp.set_global_session_id()
    # exp.init_epoc_record()
    exp.init_muse_record()

    time.sleep(5)
    user_id = int(input("1-Vitor, 2-Nick, 3-Vish, 4-Jayasri: "))

    fall_time = 0
    fall_speed = 450  # ms
    min_fall_speed = 50  # ms (kept)
    running = True
    move_delay = 80  # ms between moves when holding
    move_timer = 0
    hold_key_pressed = False

    # ORIGINAL game values (keep these exactly as tick returns)
    arousal = None
    valence = None

    # UI-only values (won't affect gameplay)
    arousal_ui = None
    valence_ui = None

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
                move_timer = 0

        # --- ORIGINAL tick logic (unchanged) ---
        if fall_time > fall_speed and not game.game_over:
            arousal, valence = game.tick(
                user_id, start_time, fall_speed, increase_difficulty.__name__
            )
            # UI: only update if both are present
            if arousal is not None and valence is not None:
                arousal_ui, valence_ui = arousal, valence
            fall_time = 0

        # --- ORIGINAL difficulty update logic (unchanged) ---
        if increase_difficulty is increase_difficulty and arousal is not None and valence is not None:
            fall_speed = increase_difficulty(game, arousal, valence)
            game.fall_speed = fall_speed

        screen.fill(COLORS["bg"])

        board_grid, color_grid = game.board.get_state()
        draw_board(screen, board_grid, color_grid)

        if not game.game_over:
            draw_piece(screen, game.current_piece)

        # Right panel UI
        sidebar_x = GRID_WIDTH * BLOCK_SIZE + 30

        next_label = font.render("Next", True, COLORS["text"])
        screen.blit(next_label, (sidebar_x, 30))
        draw_mini_piece(screen, game.next_piece, sidebar_x, 60)

        hold_label = font.render("Hold (C)", True, COLORS["text"])
        screen.blit(hold_label, (sidebar_x, 140))
        if game.hold_piece:
            draw_mini_piece(screen, game.hold_piece, sidebar_x, 170)

        score_surface = font.render(f"Score: {game.score}", True, COLORS["text"])
        screen.blit(score_surface, (sidebar_x, 250))

        lines_surface = font.render(f"Lines: {game.lines_cleared}", True, COLORS["text"])
        screen.blit(lines_surface, (sidebar_x, 280))

        # Emotion map uses UI-only values so dot doesn't vanish, but gameplay stays identical
        draw_valence_arousal_map_binary(
            screen,
            origin_x=sidebar_x,
            origin_y=340,
            w=200,
            h=200,
            valence=valence_ui,
            arousal=arousal_ui,
            center=2.5,
            dot_r=30,
        )

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