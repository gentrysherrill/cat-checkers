# Cat Checkers

A fun twist on classic Checkers where you play as orange cats against an AI controlling gray cats!

## Setup

1. Install Python 3.7 or higher
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Play

1. Run the game:
   ```
   python cat_checkers.py
   ```

2. Game Rules:
   - You play as the orange cats
   - Click on a cat to select it
   - Green circles show valid moves
   - Cats can only move diagonally forward
   - When a cat reaches the opposite end of the board, it becomes a king and can move backwards
   - Kings are marked with a crown
   - If you can capture an opponent's cat by jumping over it, you must take that move

## Features
- Cat-themed game pieces
- Smart AI opponent using minimax algorithm
- Mandatory capture moves
- Score tracking
- King pieces with crowns
- Game over detection and win screen
- Easy restart with spacebar when game ends
