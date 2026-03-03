
from .game import Game
from .flow_computation import delta_s_vec

MAX_SPEED_CHANGE = 140

def increase_difficulty_lines_cleared(game: Game, arousal: int, valence: int, max_speed=100, min_speed=600):
    """Increase fall speed based on lines cleared."""
    lines = game.lines_cleared
    fall_speed = min_speed - (lines * 10)
    fall_speed = max(min(fall_speed, min_speed), max_speed)
    return fall_speed

def constant_difficulty(game: Game, speed=300):
    """Keep fall speed constant."""
    return speed

def increase_difficulty_adaptive(game: Game, arousal: int, valence: int, max_speed=100, min_speed=600):
    """Adjust fall speed based on average isolated empty blocks and pieces placed, as well as lines cleared."""
    avg_isolated_empty = game.get_isolated_empty_avg()
    piece_factor = game.piece_count * 10
    lines = game.lines_cleared
    fall_speed = min_speed + avg_isolated_empty*200 - (lines * 10)
    fall_speed = max(min(fall_speed, min_speed), max_speed)
    return fall_speed

def increase_difficulty_blocks_placed(game: Game, arousal: int, valence: int, max_speed=100, min_speed=600):
    """Increase fall speed based on number of pieces placed."""
    piece_factor = game.piece_count * 10
    fall_speed = min_speed - piece_factor
    fall_speed = max(min(fall_speed, min_speed), max_speed)
    return fall_speed

def increase_difficulty_minimize_emotion_distance(game: Game, arousal: int, valence: int, max_speed=50, min_speed=600):
    """Increase fall speed based on the distance from target arousal and valence."""
    
    delta_s = delta_s_vec(valence, arousal)
    # get current speed
    return_speed = game.fall_speed - (delta_s * MAX_SPEED_CHANGE)
    return max(min(return_speed, min_speed), max_speed)


def increase_difficulty_flow(game: Game, arousal: int, valence: int, max_speed=100, min_speed=600):
    """Increase fall speed based on the predicted flow by EEG data."""
    fall_speed = min_speed
    if arousal > 3:
        if valence > 3:
            return fall_speed # don't change game speed
        else:
            fall_speed -= (2.5-valence)*5
    else:
        fall_speed += arousal*2.5

    return fall_speed