from copy import deepcopy
from typing import Tuple, List, Dict, Optional

# Evaluation weights
PIECE_VALUE = 1
KING_VALUE = 2
POSITION_VALUES = [
    [4, 4, 4, 4, 4, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4]
]

def get_all_moves(board, color) -> Dict[Tuple[int, int], Dict[Tuple[int, int], List]]:
    """Get all possible moves for a given color"""
    moves = {}
    for piece in get_all_pieces(board, color):
        valid_moves = board.get_valid_moves(piece)
        if valid_moves:
            moves[(piece.row, piece.col)] = valid_moves
    return moves

def get_all_pieces(board, color):
    """Get all pieces of a given color"""
    pieces = []
    for row in range(len(board.board)):
        for col in range(len(board.board[row])):
            piece = board.board[row][col]
            if piece and piece.color == color:
                pieces.append(piece)
    return pieces

def evaluate_board(board, color) -> float:
    """Evaluate the board state for the given color"""
    opponent_color = board.ORANGE if color == board.GRAY else board.GRAY
    
    score = 0
    for row in range(len(board.board)):
        for col in range(len(board.board[row])):
            piece = board.board[row][col]
            if piece:
                value = (KING_VALUE if piece.king else PIECE_VALUE) * POSITION_VALUES[row][col]
                if piece.color == color:
                    score += value
                else:
                    score -= value
    
    return score

def simulate_move(board, piece, move, captures) -> 'Board':
    """Simulate a move on a copy of the board"""
    board_copy = deepcopy(board)
    piece_copy = board_copy.board[piece[0]][piece[1]]
    
    # Move the piece
    board_copy.board[piece[0]][piece[1]] = None
    board_copy.board[move[0]][move[1]] = piece_copy
    piece_copy.row = move[0]
    piece_copy.col = move[1]
    
    # Remove captured pieces
    if captures:
        for captured in captures:
            board_copy.board[captured.row][captured.col] = None
    
    # Make kings
    if move[0] == 0 and piece_copy.color == board.ORANGE:
        piece_copy.make_king()
    elif move[0] == len(board.board) - 1 and piece_copy.color == board.GRAY:
        piece_copy.make_king()
    
    return board_copy

def minimax(board, depth: int, max_player: bool, alpha: float, beta: float, ai_color) -> Tuple[float, Optional[Tuple]]:
    """Minimax algorithm with alpha-beta pruning"""
    if depth == 0:
        return evaluate_board(board, ai_color), None
    
    color = ai_color if max_player else (board.ORANGE if ai_color == board.GRAY else board.GRAY)
    all_moves = get_all_moves(board, color)
    
    if not all_moves:
        return float('-inf') if max_player else float('inf'), None
    
    best_move = None
    if max_player:
        max_eval = float('-inf')
        for piece, moves in all_moves.items():
            for move, captures in moves.items():
                evaluation = minimax(simulate_move(board, piece, move, captures),
                                  depth - 1, False, alpha, beta, ai_color)[0]
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = (piece, move, captures)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for piece, moves in all_moves.items():
            for move, captures in moves.items():
                evaluation = minimax(simulate_move(board, piece, move, captures),
                                  depth - 1, True, alpha, beta, ai_color)[0]
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = (piece, move, captures)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
        return min_eval, best_move

def get_ai_move(board, color, depth=3):
    """Get the best move for the AI"""
    _, best_move = minimax(board, depth, True, float('-inf'), float('inf'), color)
    if best_move:
        return best_move
    return None
