"""
Action space definition, decoding, encoding, and validation for the Roadster MDP Environment.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np

NUM_ACTIONS: int = 16

# Table of all 16 discrete actions with bitwise states (u1, u2, u3, u4)
# Bit 0: V1 (Rear lift)
# Bit 1: V2 (Front lift)
# Bit 2: V3 (Push right)
# Bit 3: V4 (Push left)
# Binary string format: V4 V3 V2 V1 (standard binary representation)
ACTION_TABLE: Dict[int, Dict[str, Any]] = {
    0:  {"valves": (0, 0, 0, 0), "bin": "0000", "desc": "Free fall (All off)"},
    1:  {"valves": (1, 0, 0, 0), "bin": "0001", "desc": "V1: Rear lift (pitch down / CW)"},
    2:  {"valves": (0, 1, 0, 0), "bin": "0010", "desc": "V2: Front lift (pitch up / CCW)"},
    3:  {"valves": (1, 1, 0, 0), "bin": "0011", "desc": "V1+V2: Balanced vertical lift (18,000 N)"},
    4:  {"valves": (0, 0, 1, 0), "bin": "0100", "desc": "V3: Push right (6,000 N)"},
    5:  {"valves": (1, 0, 1, 0), "bin": "0101", "desc": "V1+V3: Push right + rear lift"},
    6:  {"valves": (0, 1, 1, 0), "bin": "0110", "desc": "V2+V3: Push right + front lift"},
    7:  {"valves": (1, 1, 1, 0), "bin": "0111", "desc": "V1+V2+V3: Push right + balanced lift"},
    8:  {"valves": (0, 0, 0, 1), "bin": "1000", "desc": "V4: Push left (6,000 N)"},
    9:  {"valves": (1, 0, 0, 1), "bin": "1001", "desc": "V1+V4: Push left + rear lift"},
    10: {"valves": (0, 1, 0, 1), "bin": "1010", "desc": "V2+V4: Push left + front lift"},
    11: {"valves": (1, 1, 0, 1), "bin": "1011", "desc": "V1+V2+V4: Push left + balanced lift"},
    12: {"valves": (0, 0, 1, 1), "bin": "1100", "desc": "V3+V4: Opposed horizontal push (cancel)"},
    13: {"valves": (1, 0, 1, 1), "bin": "1101", "desc": "V1+V3+V4: Opposed horizontal + rear lift"},
    14: {"valves": (0, 1, 1, 1), "bin": "1110", "desc": "V2+V3+V4: Opposed horizontal + front lift"},
    15: {"valves": (1, 1, 1, 1), "bin": "1111", "desc": "V1+V2+V3+V4: All thrusters firing"},
}

ALL_ACTIONS: List[int] = list(range(NUM_ACTIONS))


def validate_action(action: Any) -> int:
    """
    Validate that action is an integer in [0, 15].
    Raises TypeError or ValueError on invalid action.
    """
    if isinstance(action, bool):
        raise TypeError(f"Action cannot be a boolean, got {action}")

    if isinstance(action, float):
        raise TypeError(f"Action must be an integer, got float: {action}")

    if isinstance(action, (int,)):
        a = int(action)
    elif hasattr(action, "item") and hasattr(action, "dtype") and not np.issubdtype(action.dtype, np.floating):
        try:
            a = int(action.item())
        except Exception:
            raise TypeError(f"Cannot convert action {action} of type {type(action)} to int")
    elif isinstance(action, str):
        raise TypeError(f"Action must be an integer, got string: {action}")
    else:
        raise TypeError(f"Action must be an integer, got type {type(action)}: {action}")

    if not (0 <= a < NUM_ACTIONS):
        raise ValueError(
            f"Action must be in range [0, {NUM_ACTIONS - 1}], got {a}. No action outside 16 combinations permitted."
        )
    return a


def decode_action(action: int) -> Tuple[int, int, int, int]:
    """
    Decode an action integer (0-15) into a tuple of 4 binary valve states:
    (u1, u2, u3, u4).
    """
    a = validate_action(action)
    u1 = a & 1
    u2 = (a >> 1) & 1
    u3 = (a >> 2) & 1
    u4 = (a >> 3) & 1
    return (u1, u2, u3, u4)


def encode_action(u1: int, u2: int, u3: int, u4: int) -> int:
    """
    Encode 4 binary valve states into an action integer in [0, 15].
    """
    a = (int(u1) & 1) | ((int(u2) & 1) << 1) | ((int(u3) & 1) << 2) | ((int(u4) & 1) << 3)
    return validate_action(a)


def action_to_bin_str(action: int) -> str:
    """
    Convert action integer (0-15) to deterministic 4-character binary string:
    '0000', '0001', ..., '1111' (representing V4 V3 V2 V1).
    """
    a = validate_action(action)
    return ACTION_TABLE[a]["bin"]


def bin_str_to_action(bin_str: str) -> int:
    """
    Convert a 4-character binary string ('0000' to '1111') to action integer (0-15).
    """
    if not isinstance(bin_str, str) or len(bin_str) != 4 or any(c not in "01" for c in bin_str):
        raise ValueError(f"Binary string must be a 4-character string of '0' and '1', got '{bin_str}'")
    action = int(bin_str, 2)
    return validate_action(action)


def get_action_description(action: int) -> str:
    """Get human-readable description of an action."""
    a = validate_action(action)
    return ACTION_TABLE[a]["desc"]
