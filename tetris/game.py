# game.py

import pygame
from .piece import Piece
from .board import Board
from .settings import *
from .analytics import *
import time
from experiment import experiment as exp

class Game:
    def __init__(self, difficulty_level=0):
        self.board = Board()
        self.current_piece = Piece()
        self.next_piece = Piece()
        self.hold_piece = None
        self.hold_used = False
        self.score = 0
        self.lines_cleared = 0
        self.piece_count = 0
        self.game_over = False
        self.isolated_empty_sum = 0
        self.isolated_empty_count = 0
        self.fall_speed = 450  # Initial fall speed in milliseconds
        self.difficulty_level = difficulty_level  #For logging purposes, will not change during game
    def update_isolated_empty(self):
        val = self.board.count_isolated_empty()
        self.isolated_empty_sum += val
        self.isolated_empty_count += 1
    def get_isolated_empty_avg(self):
        if self.isolated_empty_count == 0:
            return 0
        return self.isolated_empty_sum / self.isolated_empty_count
    def hold(self):
        if self.hold_used:
            return
        if self.hold_piece is None:
            self.hold_piece = self.current_piece
            self.current_piece = self.next_piece
            self.next_piece = Piece()
        else:
            self.hold_piece, self.current_piece = self.current_piece, self.hold_piece
        self.current_piece.x = 3
        self.current_piece.y = 0
        self.hold_used = True

    def spawn_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = Piece()
        self.hold_used = False
        self.piece_count += 1
        if not self.board.valid_position(self.current_piece):
            self.game_over = True
            on_game_over(self.board.get_state(), self.score, self)

    def move(self, dx, dy):
        p = self.current_piece.copy()
        p.x += dx
        p.y += dy
        if self.board.valid_position(p):
            self.current_piece = p
            log_move(self.board.grid, self.current_piece, f"move_{dx}_{dy}")
            return True
        return False

    def rotate(self):
        p = self.current_piece.copy()
        p.rotate()
        if self.board.valid_position(p):
            self.current_piece = p
            log_move(self.board.grid, self.current_piece, "rotate")
            return True
        return False

    def drop(self, user_id, start_time, fall_speed, difficulty):
        # Soft drop
        if not self.move(0, 1):
            # Place the piece
            self.board.place_piece(self.current_piece)
            cleared = self.board.clear_lines()
            if cleared:
                self.score += [0, 40, 100, 300, 1200][cleared]
                self.lines_cleared += cleared
                log_move(self.board.grid, self.current_piece, "clear", reward=cleared)
            self.spawn_piece()
            arousal, valence = exp.predict_n_insert(
                user_id,
                self.piece_count,
                time.time() - start_time,  # In seconds
                -1,
                -1,
                fall_speed,
                difficulty,
            )
            return (arousal, valence)
        return (None, None)

    def hard_drop(self):
        while self.move(0, 1):
            pass
        self.drop()

    def tick(self, user_id, start_time, fall_speed, difficulty):
        if not self.game_over:
           arousal, valence = self.drop(user_id, start_time, fall_speed, difficulty)
           return (arousal, valence)