# Tetris-EEG

This repository contains code to run an adaptation of Tetris that dynamically changes the game speed based on Machine Learning predictions of the user's current Arousal and Valence. This project was originally built to run an experiment called EEGoFeedback: Dynamic Flow Maintenance through Real-time EEG Emotion Prediction.

Repo Structure:
```
.
├── dreamer_model
|   ├── datasets  # all generated csv go here
|   ├── ML  # contains models, utils, and training split functions
|   ├── features.ipynb  # generates features tables from DREAMER
|   ├── predictor_model.ipynb   # train model to predict in-game
|   └── run_models.ipynb    # notebook to compile all models
├── experiment
|   ├── EpocX   # contains code to get and manipulate data from EEG
|   ├── Muse   # contains code to get and manipulate data from EEG
|   ├── experiment_muse.py   # functions for syncing EEG data with game
└── tetris
    └── main.py
```

## Initial Setup

Once you have set up your python environment, run
```zsh
pip install -r requirements.txt
```

## Playing the game/Data Collection
Current tetris implementation only tested with Python 3.13.7

To play the game, set the current working directory to Tetris-EEG/tetris, then run main.py. Make sure that a Muse EEG headset is close by and powered on.

First run:
```
muselsl stream
```

After streaming is up and running, run:
```zsh
python -m tetris
```

### Controls

| Button | Action | Description |
|---|---|---|
| Left Arrow | Move Left | Moves the current piece to the left |
| Right Arrow | Move Right | Moves the current piece to the right |
| Up Arrow | Rotate Piece | Rotates the current piece |
| Down Arrow | Soft Drop | Increases fall speed of the current piece |
| Space | Hard Drop | Instantly lands and locks current piece |
| C | Hold/Swap | Stashes current piece in the Hold cell, swaps with piece in hold cell if available |
