"""
Unit tests for Curriculum Learning Manager.
"""

import pytest
from training.curriculum_manager import CurriculumManager, DEFAULT_LEVELS


def test_curriculum_initialization():
    cm = CurriculumManager(starting_level=1)
    assert cm.current_level == 1
    assert cm.level_episodes == 0
    assert cm.success_rate == 0.0
    assert cm.get_reset_options() == {"level": 1}


def test_curriculum_promotion():
    cm = CurriculumManager(starting_level=1, window_size=5)
    # Level 1 requires min_episodes=200, threshold=0.80.
    # Manually tweak spec for quick unit testing
    cm.current_spec.min_episodes = 5
    cm.current_spec.success_threshold = 0.80

    for _ in range(4):
        promoted = cm.record_episode(success=True)
        assert not promoted

    # 5th episode
    promoted = cm.record_episode(success=True)
    assert promoted
    assert cm.current_level == 2
    assert cm.level_episodes == 0
    assert len(cm.level_history) == 1
    assert cm.level_history[0]["level"] == 1


def test_curriculum_manual_override():
    cm = CurriculumManager(starting_level=1)
    cm.set_level(3)
    assert cm.current_level == 3
    assert cm.get_reset_options() == {"level": 3}

    with pytest.raises(ValueError):
        cm.set_level(99)
