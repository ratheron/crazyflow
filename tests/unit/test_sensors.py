"""Unit tests for the sensors module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.sim import Sim
from crazyflow.sim.sensors import _camera_rays, build_render_depth_fn, render_depth
from crazyflow.sim.uwb import estimate_uwb_imu_state, reset_uwb_bias, simulate_imu, simulate_uwb


@pytest.mark.unit
def test_camera_rays():
    """Test that camera_rays produces arrays with correct shape and device."""
    resolution = (64, 48)
    rays = _camera_rays(resolution=resolution)
    # Check shape: should be (height, width, 3)
    expected_shape = (resolution[1], resolution[0], 3)
    assert rays.shape == expected_shape, f"Expected shape {expected_shape}, got {rays.shape}"
    # Check that rays are normalized
    norm = jnp.linalg.norm(rays, axis=-1)
    assert jnp.allclose(norm, 1.0, atol=1e-6), "Rays should be normalized"
    # Check that rays respect the FOV
    rays_narrow = _camera_rays(fov_y=np.pi / 6)
    rays_wide = _camera_rays(fov_y=np.pi / 3)
    # Corner rays should have different angles for different FOV
    # Check the top corner ray y-component (wider FOV should have larger y-component)
    corner_y_narrow = abs(rays_narrow[0, 0, 1])  # Top-left corner
    corner_y_wide = abs(rays_wide[0, 0, 1])
    assert corner_y_wide > corner_y_narrow, "Wider FOV should produce rays with larger y-components"


@pytest.mark.unit
def test_render_depth(device: str):
    """Test render_depth with different resolutions."""
    sim = Sim(n_worlds=2, device=device)
    dist = render_depth(sim, camera=0, resolution=(10, 10))
    assert dist.shape == (2, 10, 10), f"Expected shape (2, 10, 10), got {dist.shape}"
    assert dist.device == jax.devices(device)[0], f"Expected device {device}, got {dist.device}"


@pytest.mark.unit
def test_build_render_depth_fn():
    """Test build_render_depth_fn produces a callable that returns correct shapes."""
    sim = Sim(n_worlds=3)
    render_depth_fn = build_render_depth_fn(
        sim.mjx_model, camera=0, resolution=(20, 15), geomgroup=(1, 1, 0, 1, 1, 1, 1, 1)
    )
    dist = render_depth_fn(sim)
    assert dist.shape == (3, 15, 20), f"Expected shape (3, 15, 20), got {dist.shape}"


@pytest.mark.unit
def test_simulate_uwb_applies_persistent_bias():
    """Test that UWB simulation adds the configured persistent range bias."""
    sim = Sim(n_worlds=1, n_drones=1)
    range_bias = jnp.full_like(sim.data.uwb.range_bias, 0.1)
    sim.data = sim.data.replace(
        uwb=sim.data.uwb.replace(
            range_std=jnp.zeros_like(sim.data.uwb.range_std),
            range_bias=range_bias,
        )
    )

    sim.data = simulate_uwb(sim.data)

    true_ranges = jnp.linalg.norm(
        sim.data.states.pos[..., None, :] - sim.data.uwb.base_stations, axis=-1
    )
    assert jnp.allclose(sim.data.uwb.ranges, true_ranges + range_bias)


@pytest.mark.unit
def test_reset_uwb_bias_masked():
    """Test that reset-time UWB bias sampling respects the world mask."""
    sim = Sim(n_worlds=2, n_drones=1)
    initial_bias = jnp.full_like(sim.data.uwb.range_bias, 0.3)
    range_bias_max = jnp.full_like(sim.data.uwb.range_bias_max, 0.2)
    sim.data = sim.data.replace(
        uwb=sim.data.uwb.replace(range_bias=initial_bias, range_bias_max=range_bias_max)
    )

    data = reset_uwb_bias(sim.data, jnp.array([True, False]))

    assert jnp.all(data.uwb.range_bias[0] >= 0.0)
    assert jnp.all(data.uwb.range_bias[0] <= 0.2)
    assert jnp.all(data.uwb.range_bias[1] == initial_bias[1])


@pytest.mark.unit
def test_batched_uwb_imu_estimator_shapes():
    """Test that the UWB+IMU EKF supports batched worlds and drones."""
    sim = Sim(n_worlds=2, n_drones=2)
    data = simulate_uwb(sim.data)
    data = simulate_imu(data)
    data = estimate_uwb_imu_state(data)

    assert data.estimates.pos.shape == (2, 2, 3)
    assert data.estimates.quat.shape == (2, 2, 4)
    assert data.estimates.covariance.shape == (2, 2, 19, 19)
