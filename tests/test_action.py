"""
Unit tests for Action space, bitwise decoding, encoding, binary strings, and validation.
"""

import pytest

from environment.action import (
    NUM_ACTIONS,
    ALL_ACTIONS,
    ACTION_TABLE,
    validate_action,
    decode_action,
    encode_action,
    action_to_bin_str,
    bin_str_to_action,
    get_action_description,
)


def test_action_count_and_table():
    """Verify exactly 16 actions exist in the action space."""
    assert NUM_ACTIONS == 16
    assert len(ALL_ACTIONS) == 16
    assert len(ACTION_TABLE) == 16
    assert set(ALL_ACTIONS) == set(range(16))


def test_action_bitwise_decoding():
    """Verify bitwise decoding for all 16 actions matches valve combinations."""
    # Action 0: 0000 -> (0, 0, 0, 0)
    assert decode_action(0) == (0, 0, 0, 0)
    # Action 1: 0001 -> (1, 0, 0, 0) V1
    assert decode_action(1) == (1, 0, 0, 0)
    # Action 2: 0010 -> (0, 1, 0, 0) V2
    assert decode_action(2) == (0, 1, 0, 0)
    # Action 3: 0011 -> (1, 1, 0, 0) V1+V2
    assert decode_action(3) == (1, 1, 0, 0)
    # Action 4: 0100 -> (0, 0, 1, 0) V3
    assert decode_action(4) == (0, 0, 1, 0)
    # Action 8: 1000 -> (0, 0, 0, 1) V4
    assert decode_action(8) == (0, 0, 0, 1)
    # Action 12: 1100 -> (0, 0, 1, 1) V3+V4
    assert decode_action(12) == (0, 0, 1, 1)
    # Action 15: 1111 -> (1, 1, 1, 1) All
    assert decode_action(15) == (1, 1, 1, 1)


def test_action_encoding_decoding_invariance():
    """Verify that encode_action(*decode_action(a)) == a for all 16 actions."""
    for a in range(16):
        valves = decode_action(a)
        re_encoded = encode_action(*valves)
        assert re_encoded == a


def test_binary_string_conversions():
    """Verify binary string conversion '0000' to '1111'."""
    expected_bin_strings = [
        "0000", "0001", "0010", "0011",
        "0100", "0101", "0110", "0111",
        "1000", "1001", "1010", "1011",
        "1100", "1101", "1110", "1111",
    ]
    for a in range(16):
        bin_str = action_to_bin_str(a)
        assert bin_str == expected_bin_strings[a]
        re_action = bin_str_to_action(bin_str)
        assert re_action == a


def test_invalid_action_rejection():
    """Verify that actions outside [0, 15] or non-integers are rejected."""
    # Negative action
    with pytest.raises(ValueError, match="Action must be in range"):
        validate_action(-1)

    # Action too large
    with pytest.raises(ValueError, match="Action must be in range"):
        validate_action(16)
    with pytest.raises(ValueError, match="Action must be in range"):
        validate_action(100)

    # Float actions
    with pytest.raises(TypeError):
        validate_action(1.5)

    # Invalid binary string
    with pytest.raises(ValueError):
        bin_str_to_action("1020")
    with pytest.raises(ValueError):
        bin_str_to_action("01")
