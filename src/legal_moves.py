# src/legal_moves.py
"""
Legal move utilities for the Chess Challenge.

This module enforces "legal moves only" during generation by:
- Reconstructing a chess.Board from dataset tokens (WPe2e4, BNg8f6, ...)
- Computing legal moves with python-chess
- Mapping legal UCI moves back to dataset tokens
- Returning the corresponding token IDs present in the tokenizer vocabulary
"""

from __future__ import annotations

from typing import List, Optional

import chess
import torch
from transformers import PreTrainedTokenizer


# Tokenizer special tokens (literal strings)
SPECIAL_TOKENS = {"[PAD]", "[BOS]", "[EOS]", "[UNK]"}

PIECE_TYPE_TO_LETTER = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def token_id_to_token(tokenizer: PreTrainedTokenizer, token_id: int) -> str:
    """Convert a token_id to its string representation."""
    return tokenizer.convert_ids_to_tokens([int(token_id)])[0]


def _strip_annotations(token: str) -> str:
    """Remove capture / check / mate annotations."""
    for ch in ("x", "+", "#", "*", "!", "?"):
        token = token.replace(ch, "")
    return token


def dataset_token_to_uci(token: str, board: chess.Board) -> Optional[str]:
    """
    Convert a dataset token (e.g. WPe2e4, BNg8f6, WPe7e8Q+)
    into a valid UCI string for python-chess.
    """
    if token in SPECIAL_TOKENS:
        return None

    token = token.strip()
    if not token or len(token) < 6:
        return None

    # Castling (dataset variants)
    if "O-O" in token or "o-o" in token:
        queenside = "O-O-O" in token or "o-o-o" in token
        if board.turn == chess.WHITE:
            return "e1c1" if queenside else "e1g1"
        else:
            return "e8c8" if queenside else "e8g8"

    color = token[0]
    piece = token[1]
    if color not in {"W", "B"}:
        return None
    if piece not in {"P", "N", "B", "R", "Q", "K"}:
        return None

    rest = _strip_annotations(token[2:])

    if len(rest) < 4:
        return None

    from_sq = rest[0:2]
    to_sq = rest[2:4]

    promo = ""
    if len(rest) >= 5 and rest[4] in {"Q", "R", "B", "N"}:
        promo = rest[4].lower()

    return f"{from_sq}{to_sq}{promo}"


def uci_to_dataset_token(board: chess.Board, move: chess.Move) -> str:
    """
    Convert a legal python-chess move into a dataset token
    WITHOUT annotations (x, +, #).
    """
    color = "W" if board.turn == chess.WHITE else "B"
    piece = board.piece_at(move.from_square)

    piece_letter = (
        PIECE_TYPE_TO_LETTER[piece.piece_type] if piece else "P"
    )

    uci = move.uci()
    if len(uci) == 5:
        uci = uci[:4] + uci[4].upper()

    return f"{color}{piece_letter}{uci}"


def board_from_input_ids(
    input_ids: torch.LongTensor,
    tokenizer: PreTrainedTokenizer,
) -> chess.Board:
    """
    Reconstruct a chess.Board from a sequence of dataset token IDs.
    """
    ids = (
        input_ids.detach().cpu().view(-1).tolist()
        if isinstance(input_ids, torch.Tensor)
        else list(input_ids)
    )

    board = chess.Board()

    for tid in ids:
        token = token_id_to_token(tokenizer, tid)

        if token in SPECIAL_TOKENS:
            if token == "[EOS]":
                break
            continue

        uci = dataset_token_to_uci(token, board)
        if uci is None:
            continue

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue

        if move in board.legal_moves:
            board.push(move)

    return board


def legal_token_ids(board: chess.Board, tokenizer: PreTrainedTokenizer) -> List[int]:
    """
    Return token IDs corresponding to legal next moves.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if board.is_game_over(claim_draw=True):
        return [eos_id] if eos_id is not None else []

    legal_ids: List[int] = []
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for move in board.legal_moves:
        base_token = uci_to_dataset_token(board, move)

        # Try base token and a few annotated variants (if present in vocab)
        candidates = [base_token]

        if board.is_capture(move):
            candidates.append(base_token + "x")

        board.push(move)
        if board.is_checkmate():
            candidates.extend([base_token + "#", base_token + "*"])
        elif board.is_check():
            candidates.append(base_token + "+")
        board.pop()

        for tok in candidates:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == unk_id:
                continue
            legal_ids.append(tid)
            break

    if not legal_ids and eos_id is not None:
        legal_ids.append(eos_id)

    return legal_ids


def legal_moves_mask(board: chess.Board, tokenizer: PreTrainedTokenizer) -> torch.BoolTensor:
    """Return a boolean mask over vocab for legal next moves."""
    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    for tid in legal_token_ids(board, tokenizer):
        if tid is not None and 0 <= tid < tokenizer.vocab_size:
            mask[tid] = True
    return mask


def legal_moves_indices(board: chess.Board, tokenizer: PreTrainedTokenizer) -> List[int]:
    """Alias for legal_token_ids."""
    return legal_token_ids(board, tokenizer)


def is_legal_move_token(
    board: chess.Board,
    token_id: int,
    tokenizer: PreTrainedTokenizer,
) -> bool:
    """Check whether a token ID represents a legal move."""
    token = token_id_to_token(tokenizer, token_id)

    if token in SPECIAL_TOKENS:
        return token == "[EOS]" and board.is_game_over(claim_draw=True)

    uci = dataset_token_to_uci(token, board)
    if uci is None:
        return False

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False

    return move in board.legal_moves



