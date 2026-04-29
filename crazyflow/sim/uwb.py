from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.control.control import controllable
from crazyflow.utils import leaf_replace

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


def simulate_uwb(data: SimData) -> SimData:
    """Simulate one UWB communication update."""
    uwb = data.uwb
    mask = controllable(data.core.steps, data.core.freq, uwb.steps, uwb.freq)

    key, subkey = jax.random.split(data.core.rng_key)
    ranges = jnp.linalg.norm(data.states.pos[..., None, :] - uwb.base_stations, axis=-1)
    noise = jax.random.normal(subkey, ranges.shape) * uwb.range_std
    ranges = jnp.maximum(ranges + noise, 0.0)

    uwb = leaf_replace(uwb, mask, ranges=ranges, steps=data.core.steps)
    return data.replace(uwb=uwb, core=data.core.replace(rng_key=key))


def estimate_uwb_state(data: SimData, process_accel_std: float = 2.0) -> SimData:
    """Estimate translational state from latest UWB ranges.

    The range-only estimator updates position and velocity. Attitude is copied into the estimate as
    a stand-in for an onboard attitude estimate; UWB alone does not observe orientation.
    """
    dt = 1.0 / data.core.freq
    updated = data.uwb.steps == data.core.steps

    x = jnp.concat((data.estimates.pos, data.estimates.vel), axis=-1)
    f = _constant_velocity_transition(dt, x.dtype)
    q = _constant_velocity_process_covariance(dt, process_accel_std, x.dtype)
    acc = _control_acceleration(data)
    pos_pred = data.estimates.pos + data.estimates.vel * dt + 0.5 * acc * dt**2
    vel_pred = data.estimates.vel + acc * dt
    x_pred = jnp.concat((pos_pred, vel_pred), axis=-1)
    p_pred = f @ data.estimates.covariance @ f.T + q

    z = multilaterate(data.uwb.base_stations, data.uwb.ranges)
    r = _range_position_covariance(data.uwb.base_stations, x_pred[..., :3], data.uwb.range_std)
    p_xz = p_pred[..., :, :3]
    s = p_pred[..., :3, :3] + r
    gain = jnp.swapaxes(jnp.linalg.solve(s, jnp.swapaxes(p_xz, -1, -2)), -1, -2)

    innovation = z - x_pred[..., :3]
    x_corr = x_pred + jnp.einsum("...ij,...j->...i", gain, innovation)
    p_corr = p_pred - gain @ s @ jnp.swapaxes(gain, -1, -2)
    p_corr = 0.5 * (p_corr + jnp.swapaxes(p_corr, -1, -2))
    p_corr = p_corr + jnp.eye(6, dtype=p_corr.dtype) * 1e-9

    x_mask = updated[..., None]
    p_mask = updated[..., None, None]
    x = jnp.where(x_mask, x_corr, x_pred)
    covariance = jnp.where(p_mask, p_corr, p_pred)

    estimates = data.estimates.replace(
        pos=x[..., :3],
        vel=x[..., 3:6],
        quat=data.states.quat,
        ang_vel=data.states.ang_vel,
        covariance=covariance,
    )
    return data.replace(estimates=estimates)


def multilaterate(base_stations: Array, ranges: Array) -> Array:
    """Estimate position by linear least-squares multilateration."""
    ref = base_stations[0]
    stations = base_stations[1:]
    a = 2.0 * (ref - stations)
    b = (
        ranges[..., 1:] ** 2
        - ranges[..., [0]] ** 2
        - jnp.sum(stations**2, axis=-1)
        + jnp.sum(ref**2)
    )
    pinv = jnp.linalg.pinv(a)
    return jnp.einsum("ij,...j->...i", pinv, b)


def _constant_velocity_transition(dt: float, dtype: jnp.dtype) -> Array:
    f = jnp.eye(6, dtype=dtype)
    return f.at[:3, 3:6].set(jnp.eye(3, dtype=dtype) * dt)


def _constant_velocity_process_covariance(dt: float, accel_std: float, dtype: jnp.dtype) -> Array:
    eye = jnp.eye(3, dtype=dtype)
    q = accel_std**2
    upper = jnp.concat((eye * (dt**4 / 4.0), eye * (dt**3 / 2.0)), axis=-1)
    lower = jnp.concat((eye * (dt**3 / 2.0), eye * (dt**2)), axis=-1)
    return jnp.concat((upper, lower), axis=0) * q


def _control_acceleration(data: SimData) -> Array:
    force_torque = data.controls.force_torque
    if force_torque is None:
        return jnp.zeros_like(data.estimates.pos)

    thrust = force_torque.cmd[..., 0:1]
    z_axis = _body_z_axis(data.estimates.quat)
    return z_axis * thrust / data.params.mass + data.params.gravity_vec


def _body_z_axis(quat: Array) -> Array:
    x, y, z, w = jnp.moveaxis(quat, -1, 0)
    return jnp.stack(
        (2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x**2 + y**2)), axis=-1
    )


def _range_position_covariance(base_stations: Array, pos: Array, range_std: Array) -> Array:
    diff = pos[..., None, :] - base_stations
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    jac = diff / jnp.maximum(dist, 1e-6)
    range_var = jnp.maximum(range_std[..., 0] ** 2, 1e-8)
    information = jnp.einsum("...ai,...aj->...ij", jac, jac) / range_var[..., None, None]
    return jnp.linalg.inv(information + jnp.eye(3, dtype=pos.dtype) * 1e-9)
