import pygame
import sys
import os
from typing import List, Tuple, Optional
from ai_player import get_ai_move
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 800
BOARD_SIZE = 8
SQUARE_SIZE = WINDOW_SIZE // BOARD_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SCORE_HEIGHT = 60
WINDOW_HEIGHT = WINDOW_SIZE + SCORE_HEIGHT

# Initialize font
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 32)

# Load images
def load_images():
    images = {}
    for name in ['orange_cat', 'orange_cat_king', 'gray_cat', 'gray_cat_king']:
        image_path = os.path.join('assets', f'{name}.png')
        img = pygame.image.load(image_path)
        # Scale image to fit the square size with some padding
        scaled_size = int(SQUARE_SIZE * 0.8)  # 80% of square size
        img = pygame.transform.scale(img, (scaled_size, scaled_size))
        images[name] = img
    return images

IMAGES = load_images()

class Piece:
    def __init__(self, row: int, col: int, color: Tuple[int, int, int]):
        self.row = row
        self.col = col
        self.color = color
        self.king = False
        self.x = 0
        self.y = 0
        self.calc_pos()
    
    def calc_pos(self):
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2
    
    def make_king(self):
        self.king = True
    
    def draw(self, screen):
        # Determine which image to use
        if self.color == Board.ORANGE:
            img = IMAGES['orange_cat_king' if self.king else 'orange_cat']
        else:
            img = IMAGES['gray_cat_king' if self.king else 'gray_cat']
        
        # Calculate position to center the image
        img_rect = img.get_rect()
        img_rect.center = (self.x, self.y)
        screen.blit(img, img_rect)

class Board:
    # Class-level color constants
    ORANGE = (255, 165, 0)
    GRAY = (128, 128, 128)
    
    def __init__(self):
        self.board = []
        self.create_board()
        self.selected_piece = None
        self.valid_moves = {}
        self.orange_score = 0
        self.gray_score = 0
        self.turn = self.ORANGE  # Orange (player) goes first
        self.game_over = False
        self.winner = None

    def create_board(self):
        for row in range(BOARD_SIZE):
            self.board.append([])
            for col in range(BOARD_SIZE):
                if col % 2 == ((row + 1) % 2):
                    if row < 3:
                        self.board[row].append(Piece(row, col, self.GRAY))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, self.ORANGE))
                    else:
                        self.board[row].append(None)
                else:
                    self.board[row].append(None)
    
    def get_valid_moves(self, piece: Piece) -> dict:
        """Returns dictionary of valid moves: {(row, col): [captured pieces]}"""
        moves = {}
        
        # Check for jumps first (these are mandatory in checkers)
        jumps = self._get_jumps(piece)
        if jumps:
            return jumps
        
        # If no jumps available, get regular moves
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row
        
        # Orange pieces move up (negative row direction)
        if piece.color == self.ORANGE or piece.king:
            moves.update(self._traverse_left(row - 1, max(row - 3, -1), -1, piece.color, left))
            moves.update(self._traverse_right(row - 1, max(row - 3, -1), -1, piece.color, right))
        
        # Gray pieces move down (positive row direction)
        if piece.color == self.GRAY or piece.king:
            moves.update(self._traverse_left(row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, left))
            moves.update(self._traverse_right(row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, right))
        
        return moves

    def _traverse_left(self, start, stop, step, color, left, skipped=None):
        """Helper method to check diagonal moves to the left"""
        moves = {}
        last = []
        for r in range(start, stop, step):
            if left < 0:
                break
            
            current = self.board[r][left]
            if current is None:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, left)] = last + skipped
                else:
                    moves[(r, left)] = last
                
                if last:
                    if step == -1:
                        row = max(r - 3, -1)
                    else:
                        row = min(r + 3, BOARD_SIZE)
                    moves.update(self._traverse_left(r + step, row, step, color, left - 1, skipped=last))
                    moves.update(self._traverse_right(r + step, row, step, color, left + 1, skipped=last))
                break
            elif current.color == color:
                break
            else:
                last = [current]
            
            left -= 1
        
        return moves

    def _traverse_right(self, start, stop, step, color, right, skipped=None):
        """Helper method to check diagonal moves to the right"""
        moves = {}
        last = []
        for r in range(start, stop, step):
            if right >= BOARD_SIZE:
                break
            
            current = self.board[r][right]
            if current is None:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, right)] = last + skipped
                else:
                    moves[(r, right)] = last
                
                if last:
                    if step == -1:
                        row = max(r - 3, -1)
                    else:
                        row = min(r + 3, BOARD_SIZE)
                    moves.update(self._traverse_left(r + step, row, step, color, right - 1, skipped=last))
                    moves.update(self._traverse_right(r + step, row, step, color, right + 1, skipped=last))
                break
            elif current.color == color:
                break
            else:
                last = [current]
            
            right += 1
        
        return moves

    def _get_jumps(self, piece: Piece) -> dict:
        """Returns dictionary of valid jump moves"""
        jumps = {}
        row = piece.row
        left = piece.col - 1
        right = piece.col + 1
        
        # Orange pieces move up (negative row direction)
        if piece.color == self.ORANGE or piece.king:
            jumps.update(self._traverse_left(row - 1, max(row - 3, -1), -1, piece.color, left))
            jumps.update(self._traverse_right(row - 1, max(row - 3, -1), -1, piece.color, right))
        
        # Gray pieces move down (positive row direction)
        if piece.color == self.GRAY or piece.king:
            jumps.update(self._traverse_left(row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, left))
            jumps.update(self._traverse_right(row + 1, min(row + 3, BOARD_SIZE), 1, piece.color, right))
        
        # Only return jumps (moves that capture pieces)
        return {move: captures for move, captures in jumps.items() if captures}

    def select(self, row: int, col: int) -> bool:
        """Select a piece and calculate its valid moves"""
        if self.selected_piece:
            result = self._move(row, col)
            if not result:
                self.selected_piece = None
                self.select(row, col)
            return result

        piece = self.get_piece(row, col)
        if piece and piece.color == self.ORANGE:  # Only allow selecting orange pieces (player's pieces)
            self.selected_piece = piece
            self.valid_moves = self.get_valid_moves(piece)
            return True
        
        return False

    def _move(self, row: int, col: int) -> bool:
        """Move a piece and handle captures"""
        piece = self.selected_piece
        if piece and (row, col) in self.valid_moves:
            # Store original position for logging
            from_pos = (piece.row, piece.col)
            
            # Move the piece
            self.board[piece.row][piece.col] = None
            self.board[row][col] = piece
            piece.row = row
            piece.col = col
            piece.calc_pos()
            
            # Remove captured pieces and update score
            captured = self.valid_moves[(row, col)]
            if captured:
                for captured_piece in captured:
                    self.board[captured_piece.row][captured_piece.col] = None
                    if piece.color == self.ORANGE:
                        self.orange_score += 1
                    else:
                        self.gray_score += 1
            
            # Make kings
            if row == 0 and piece.color == self.ORANGE:
                piece.make_king()
            elif row == BOARD_SIZE - 1 and piece.color == self.GRAY:
                piece.make_king()
            
            self.selected_piece = None
            self.valid_moves = {}
            
            # Switch turns only if no more captures are available
            new_moves = self.get_valid_moves(piece)
            if not new_moves or not any(captures for captures in new_moves.values()):
                self.turn = self.GRAY if self.turn == self.ORANGE else self.ORANGE
            
            return True
        
        return False

    def ai_move(self):
        """Make AI move"""
        move = get_ai_move(self, self.GRAY)
        if move:
            piece_pos, new_pos, captures = move
            piece = self.board[piece_pos[0]][piece_pos[1]]
            self.selected_piece = piece
            self.valid_moves = {new_pos: captures}
            self._move(new_pos[0], new_pos[1])

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        return self.board[row][col]

    def draw(self, screen):
        """Draw the board, pieces, and move hints"""
        screen.fill(BLACK)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 0:
                    pygame.draw.rect(screen, WHITE, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        
        # Draw move hints
        if self.selected_piece:
            for move in self.valid_moves.keys():
                row, col = move
                # Draw a semi-transparent green circle to indicate valid moves
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(s, (0, 255, 0, 100), (SQUARE_SIZE // 2, SQUARE_SIZE // 2), SQUARE_SIZE // 4)
                screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Draw pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece:
                    piece.draw(screen)

    def draw_score(self, screen):
        """Draw the score at the bottom of the screen"""
        # Draw scores on the left and right sides
        font = pygame.font.SysFont('Arial', 24)
        orange_text = font.render(f"Orange: {self.orange_score}", True, self.ORANGE)
        gray_text = font.render(f"Gray: {self.gray_score}", True, self.GRAY)
        
        # Position scores with padding from the edges
        padding = 20
        screen.blit(orange_text, (padding, WINDOW_SIZE + 15))
        screen.blit(gray_text, (WINDOW_SIZE - gray_text.get_width() - padding, WINDOW_SIZE + 15))
        
        # Draw turn indicator centered below the scores
        turn_text = font.render("Your Turn" if self.turn == self.ORANGE else "AI's Turn", True, self.turn)
        screen.blit(turn_text, (WINDOW_SIZE // 2 - turn_text.get_width() // 2, WINDOW_SIZE + 45))

    def draw_game_over(self, screen):
        """Draw game over screen"""
        if not self.game_over:
            return

        # Create semi-transparent overlay
        overlay = pygame.Surface((WINDOW_SIZE, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        # Create game over message
        winner_text = "Orange Cats Win!" if self.winner == self.ORANGE else "Gray Cats Win!"
        font = pygame.font.SysFont('Arial', 64)
        text_surface = font.render(winner_text, True, self.winner)
        text_rect = text_surface.get_rect(center=(WINDOW_SIZE // 2, WINDOW_HEIGHT // 2 - 50))
        screen.blit(text_surface, text_rect)

        # Create score message
        score_text = f"Final Score - Orange: {self.orange_score}, Gray: {self.gray_score}"
        font = pygame.font.SysFont('Arial', 32)
        score_surface = font.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(center=(WINDOW_SIZE // 2, WINDOW_HEIGHT // 2 + 50))
        screen.blit(score_surface, score_rect)

        # Create restart message
        restart_text = "Press SPACE to play again"
        restart_surface = font.render(restart_text, True, WHITE)
        restart_rect = restart_surface.get_rect(center=(WINDOW_SIZE // 2, WINDOW_HEIGHT // 2 + 100))
        screen.blit(restart_surface, restart_rect)

    def check_winner(self):
        """Check if the game is over and determine the winner"""
        orange_pieces = gray_pieces = 0
        orange_moves = gray_moves = False

        # Count pieces and check for available moves
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece:
                    if piece.color == self.ORANGE:
                        orange_pieces += 1
                        if not orange_moves and self.get_valid_moves(piece):
                            orange_moves = True
                    else:
                        gray_pieces += 1
                        if not gray_moves and self.get_valid_moves(piece):
                            gray_moves = True

        # Check win conditions
        if orange_pieces == 0 or not orange_moves:
            self.game_over = True
            self.winner = self.GRAY
        elif gray_pieces == 0 or not gray_moves:
            self.game_over = True
            self.winner = self.ORANGE

def main():
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_HEIGHT))
    pygame.display.set_caption('Cat Checkers')
    board = Board()
    
    try:
        while True:
            if not board.game_over:
                if board.turn == board.GRAY:
                    # AI's turn
                    time.sleep(0.5)  # Add a small delay to make AI moves visible
                    board.ai_move()
                    board.check_winner()  # Check for winner after AI move
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN and board.game_over:
                    if event.key == pygame.K_SPACE:
                        # Reset the game
                        board = Board()
                        continue
                
                if event.type == pygame.MOUSEBUTTONDOWN and board.turn == board.ORANGE and not board.game_over:
                    pos = pygame.mouse.get_pos()
                    if pos[1] < WINDOW_SIZE:  # Only process clicks on the board
                        row = pos[1] // SQUARE_SIZE
                        col = pos[0] // SQUARE_SIZE
                        board.select(row, col)
                        board.check_winner()  # Check for winner after player move
            
            board.draw(screen)
            board.draw_score(screen)
            if board.game_over:
                board.draw_game_over(screen)
            pygame.display.flip()
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
