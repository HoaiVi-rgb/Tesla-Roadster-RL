"""
Unit tests for State representation, normalization, validation, and dimension consistency.
"""

import math
import pytest
import numpy as np

from environment.state import (
    STATE_DIM,
    RawState,
    raw_to_normalized,
    normalized_to_raw,
    validate_state,
    SCALE_X,
    SCALE_H,
    SCALE_VX,
    SCALE_VY,
    SCALE_OMEGA,
    SCALE_FUEL,
    SCALE_TIME,
)


def test_state_dimension_and_dtype():
    """Verify state dimension is strictly 9 and dtype is float32."""
    raw = RawState(x=0.0, h=6.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=100.0, time=0.0)
    obs = raw_to_normalized(raw)
    
    assert obs.shape == (STATE_DIM,)
    assert obs.shape == (9,)
    assert obs.dtype == np.float32
    assert validate_state(obs) is True


def test_state_normalization_formulas():
    """Verify exact reference normalization formula values."""
    raw = RawState(
        x=5.0,        # 5 / 10 = 0.5
        h=7.5,        # 7.5 / 15 = 0.5
        vx=-2.0,      # -2 / 10 = -0.2
        vy=-3.0,      # -3 / 10 = -0.3
        theta=math.pi / 2.0,  # sin = 1.0, cos = 0.0
        omega=2.5,    # 2.5 / 5 = 0.5
        fuel=80.0,    # 80 / 100 = 0.8
        time=10.0,    # 10 / 20 = 0.5
    )
    obs = raw_to_normalized(raw)
    
    assert obs[0] == pytest.approx(0.5, abs=1e-5)
    assert obs[1] == pytest.approx(0.5, abs=1e-5)
    assert obs[2] == pytest.approx(-0.2, abs=1e-5)
    assert obs[3] == pytest.approx(-0.3, abs=1e-5)
    assert obs[4] == pytest.approx(1.0, abs=1e-5)
    assert obs[5] == pytest.approx(0.0, abs=1e-5)
    assert obs[6] == pytest.approx(0.5, abs=1e-5)
    assert obs[7] == pytest.approx(0.8, abs=1e-5)
    assert obs[8] == pytest.approx(0.5, abs=1e-5)


def test_state_validation_nan():
    """Verify validation strictly rejects NaN values."""
    bad_obs = np.zeros(9, dtype=np.float32)
    bad_obs[4] = 0.0
    bad_obs[5] = 1.0
    bad_obs[2] = np.nan
    
    with pytest.raises(ValueError, match="State contains NaN"):
        validate_state(bad_obs)


def test_state_validation_infinity():
    """Verify validation strictly rejects positive and negative Infinity."""
    bad_obs_pinf = np.zeros(9, dtype=np.float32)
    bad_obs_pinf[4] = 0.0
    bad_obs_pinf[5] = 1.0
    bad_obs_pinf[1] = np.inf
    
    with pytest.raises(ValueError, match="State contains Infinite"):
        validate_state(bad_obs_pinf)

    bad_obs_ninf = np.zeros(9, dtype=np.float32)
    bad_obs_ninf[4] = 0.0
    bad_obs_ninf[5] = 1.0
    bad_obs_ninf[3] = -np.inf
    
    with pytest.raises(ValueError, match="State contains Infinite"):
        validate_state(bad_obs_ninf)


def test_state_validation_dimension():
    """Verify validation rejects states with wrong length (< 9 or > 9)."""
    with pytest.raises(ValueError, match="State dimension must be exactly"):
        validate_state(np.zeros(8, dtype=np.float32))

    with pytest.raises(ValueError, match="State dimension must be exactly"):
        validate_state(np.zeros(10, dtype=np.float32))


def test_state_roundtrip():
    """Verify roundtrip conversion between raw and normalized states."""
    raw_orig = RawState(
        x=-1.5,
        h=5.2,
        vx=0.4,
        vy=-0.8,
        theta=0.25,
        omega=-0.1,
        fuel=92.5,
        time=3.5,
    )
    obs = raw_to_normalized(raw_orig)
    raw_recon = normalized_to_raw(obs)
    
    assert raw_recon.x == pytest.approx(raw_orig.x, abs=1e-4)
    assert raw_recon.h == pytest.approx(raw_orig.h, abs=1e-4)
    assert raw_recon.vx == pytest.approx(raw_orig.vx, abs=1e-4)
    assert raw_recon.vy == pytest.approx(raw_orig.vy, abs=1e-4)
    assert raw_recon.theta == pytest.approx(raw_orig.theta, abs=1e-4)
    assert raw_recon.omega == pytest.approx(raw_orig.omega, abs=1e-4)
    assert raw_recon.fuel == pytest.approx(raw_orig.fuel, abs=1e-4)
    assert raw_recon.time == pytest.approx(raw_orig.time, abs=1e-4)
