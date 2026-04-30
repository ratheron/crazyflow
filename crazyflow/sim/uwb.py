from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from drone_estimators.estimator import Estimator
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.control.control import controllable
from crazyflow.sim.data import UKFData
from crazyflow.utils import leaf_replace

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData


def simulate_uwb(data: SimData) -> SimData:
    """Simulate one UWB communication update."""
    uwb = data.uwb
    mask = controllable(data.core.steps, data.core.freq, uwb.steps, uwb.freq)

    key, subkey = jax.random.split(data.core.rng_key)
    ranges = jnp.linalg.norm(data.states.pos[..., None, :] - uwb.base_stations, axis=-1)
    ranges = jnp.maximum(ranges + jax.random.normal(subkey, ranges.shape) * uwb.range_std, 0.0)

    uwb = leaf_replace(uwb, mask, ranges=ranges, steps=data.core.steps)
    return data.replace(uwb=uwb, core=data.core.replace(rng_key=key))


def simulate_imu(data: SimData) -> SimData:
    """Simulate one IMU sample from ground-truth dynamics."""
    key, accel_key, gyro_key, accel_bias_key, gyro_bias_key = jax.random.split(data.core.rng_key, 5)
    dt_sqrt = jnp.sqrt(jnp.asarray(1.0 / data.core.freq, dtype=data.states.vel.dtype))
    accel_bias = data.imu.accel_bias + (
        jax.random.normal(accel_bias_key, data.imu.accel_bias.shape)
        * data.imu.accel_bias_walk_std
        * dt_sqrt
    )
    gyro_bias = data.imu.gyro_bias + (
        jax.random.normal(gyro_bias_key, data.imu.gyro_bias.shape)
        * data.imu.gyro_bias_walk_std
        * dt_sqrt
    )
    world_accel = data.states_deriv.acc
    accel = (
        R.from_quat(data.states.quat).inv().apply(world_accel - data.params.gravity_vec)
    )  # TODO sign is wrong!
    gyro = data.states.ang_vel
    accel = accel + accel_bias + jax.random.normal(accel_key, accel.shape) * data.imu.accel_std
    gyro = gyro + gyro_bias + jax.random.normal(gyro_key, gyro.shape) * data.imu.gyro_std
    return data.replace(
        imu=data.imu.replace(
            accel=accel,
            gyro=gyro,
            prev_vel=data.states.vel,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        ),
        core=data.core.replace(rng_key=key),
    )


def estimate_uwb_imu_state(data: SimData) -> SimData:
    """Estimate full state from one UWB tag and IMU measurements."""
    n_worlds, n_drones, n_anchors = data.uwb.ranges.shape
    state_dim = data.estimates.covariance.shape[-1]
    n_sigmas = 2 * state_dim + 1
    ukf_data = UKFData(
        pos=data.estimates.pos,
        quat=data.estimates.quat,
        vel=data.estimates.vel,
        ang_vel=data.estimates.ang_vel,
        rotor_vel=None,
        dist_f=None,
        dist_t=None,
        covariance=data.estimates.covariance,
        sigmas_f=jnp.zeros(
            (n_worlds, n_drones, n_sigmas, state_dim), dtype=data.estimates.pos.dtype
        ),
        sigmas_h=jnp.zeros(
            (n_worlds, n_drones, n_sigmas, n_anchors), dtype=data.estimates.pos.dtype
        ),
        u=jnp.concat((data.imu.accel, data.imu.gyro), axis=-1),
        z=data.uwb.ranges,
        dt=1.0 / data.core.freq,
        accel_bias=data.estimates.accel_bias,
        gyro_bias=data.estimates.gyro_bias,
    )
    ukf_data = uwb_imu_step(
        ukf_data,
        ranges=data.uwb.ranges,
        base_stations=data.uwb.base_stations,
        imu_accel=data.imu.accel,
        imu_gyro=data.imu.gyro,
        gravity_vec=data.params.gravity_vec,
        range_std=data.uwb.estimator_range_std,
        accel_std=data.imu.accel_std,
        gyro_std=data.imu.gyro_std,
        accel_bias_walk_std=data.imu.accel_bias_walk_std,
        gyro_bias_walk_std=data.imu.gyro_bias_walk_std,
        uwb_updated=data.uwb.steps == data.core.steps,
    )
    estimates = data.estimates.replace(
        pos=ukf_data.pos,
        quat=ukf_data.quat,
        vel=ukf_data.vel,
        ang_vel=ukf_data.ang_vel,
        accel_bias=ukf_data.accel_bias,
        gyro_bias=ukf_data.gyro_bias,
        covariance=ukf_data.covariance,
    )
    return data.replace(estimates=estimates)


def uwb_imu_step(
    data: UKFData,
    ranges: Array,
    base_stations: Array,
    imu_accel: Array,
    imu_gyro: Array,
    gravity_vec: Array,
    range_std: Array,
    accel_std: Array,
    gyro_std: Array,
    accel_bias_walk_std: Array,
    gyro_bias_walk_std: Array,
    uwb_updated: Array,
) -> UKFData:
    """Step a full-state UWB+IMU estimator using one UWB tag and IMU measurements.

    The state is [pos, quat, vel, ang_vel, accel_bias, gyro_bias]. IMU acceleration and gyro are
    used as propagation inputs after subtracting the estimated biases, while UWB ranges provide
    the correction.
    """
    x = UKFData.as_state_array(data)
    n = x.shape[-1]
    if n not in (13, 19):
        raise ValueError(f"UWB+IMU estimator expects a 13D or 19D state, got {n}D")
    dt = data.dt
    x_pred = _uwb_imu_process(x[..., None, :], imu_accel, imu_gyro, gravity_vec, dt)[..., 0, :]
    x_pred = _set_imu_attitude(x_pred, data.quat, imu_gyro, dt)
    p_pred = _uwb_imu_covariance_predict(
        data.covariance,
        x,
        imu_accel,
        imu_gyro,
        gravity_vec,
        dt,
        _uwb_imu_process_noise(dt, n, accel_std, gyro_std, accel_bias_walk_std, gyro_bias_walk_std),
    )
    p_pred = _regularize_covariance(p_pred)

    diff = x_pred[..., None, 0:3] - base_stations
    dist = jnp.linalg.norm(diff, axis=-1)
    z_pred = dist
    range_jac = diff / jnp.maximum(dist[..., None], 1e-6)
    h = jnp.zeros((*x_pred.shape[:-1], ranges.shape[-1], n), dtype=x_pred.dtype)
    h = h.at[..., :, 0:3].set(range_jac)
    r_cov = _range_noise_cov(range_std)
    s = h @ p_pred @ jnp.swapaxes(h, -1, -2) + r_cov
    p_ht = p_pred @ jnp.swapaxes(h, -1, -2)
    gain = jnp.swapaxes(jnp.linalg.solve(s, jnp.swapaxes(p_ht, -1, -2)), -1, -2)
    gain = gain.at[..., 3:7, :].set(0.0)
    gain = gain.at[..., 10:13, :].set(0.0)
    innovation = ranges - z_pred
    x_corr = _set_imu_attitude(
        x_pred + jnp.einsum("...ij,...j->...i", gain, innovation), data.quat, imu_gyro, dt
    )
    identity = jnp.eye(n, dtype=x.dtype)
    residual_map = identity - gain @ h
    p_corr = residual_map @ p_pred @ jnp.swapaxes(residual_map, -1, -2)
    p_corr = p_corr + gain @ r_cov @ jnp.swapaxes(gain, -1, -2)
    p_corr = _regularize_covariance(p_corr)

    x_out = jnp.where(uwb_updated[..., None], x_corr, x_pred)
    p_out = jnp.where(uwb_updated[..., None, None], p_corr, p_pred)

    return UKFData.from_state_array(
        data.replace(covariance=p_out, z=ranges, u=jnp.concat((imu_accel, imu_gyro), axis=-1)),
        x_out,
    )


class UWBIMUKalmanFilter(Estimator):
    """Functional Kalman filter wrapper for UWB range and IMU fusion."""

    def __init__(
        self,
        dt: float,
        base_stations: Array,
        range_std: float = 0.05,
        accel_std: float = 0.05,
        gyro_std: float = 0.005,
        accel_bias_walk_std: float = 0.0,
        gyro_bias_walk_std: float = 0.0,
    ):
        """Initialize the UWB+IMU filter wrapper."""
        dim_x, dim_u, dim_z = 19, 6, base_stations.shape[0]
        super().__init__(dim_x, dim_u, dim_z, dt)
        self.base_stations = jnp.asarray(base_stations)
        self.range_std = jnp.asarray(range_std)
        self.accel_std = jnp.asarray(accel_std)
        self.gyro_std = jnp.asarray(gyro_std)
        self.accel_bias_walk_std = jnp.asarray(accel_bias_walk_std)
        self.gyro_bias_walk_std = jnp.asarray(gyro_bias_walk_std)
        self.data = UKFData.create_empty(accel_bias=True, gyro_bias=True, dim_u=dim_u, dim_z=dim_z)
        self.data = self.data.replace(
            covariance=jnp.eye(dim_x) * 0.01,
            sigmas_f=jnp.zeros((2 * dim_x + 1, dim_x)),
            sigmas_h=jnp.zeros((2 * dim_x + 1, dim_z)),
            dt=dt,
        )

    def step(self, ranges: Array, imu_accel: Array, imu_gyro: Array, gravity_vec: Array) -> UKFData:
        """Step the estimator with one set of range and IMU measurements."""
        self.data = uwb_imu_step(
            self.data,
            ranges,
            self.base_stations,
            imu_accel,
            imu_gyro,
            gravity_vec,
            self.range_std,
            self.accel_std,
            self.gyro_std,
            self.accel_bias_walk_std,
            self.gyro_bias_walk_std,
            jnp.asarray(True),
        )
        return self.data


def _uwb_imu_process(
    sigmas: Array, imu_accel: Array, imu_gyro: Array, gravity_vec: Array, dt: float
) -> Array:
    accel = jnp.broadcast_to(imu_accel[..., None, :], (*sigmas.shape[:-1], 3))
    gyro = jnp.broadcast_to(imu_gyro[..., None, :], (*sigmas.shape[:-1], 3))
    gravity = jnp.broadcast_to(gravity_vec[..., None, :], (*sigmas.shape[:-1], 3))
    flat_sigmas = sigmas.reshape((-1, sigmas.shape[-1]))
    flat_accel = accel.reshape((-1, 3))
    flat_gyro = gyro.reshape((-1, 3))
    flat_gravity = gravity.reshape((-1, 3))
    flat_next = jax.vmap(_uwb_imu_process_one, in_axes=(0, 0, 0, 0, None))(
        flat_sigmas, flat_accel, flat_gyro, flat_gravity, dt
    )
    return flat_next.reshape(sigmas.shape)


def _uwb_imu_process_one(
    x: Array, imu_accel: Array, imu_gyro: Array, gravity_vec: Array, dt: float
) -> Array:
    pos = x[0:3]
    quat = x[3:7]
    vel = x[7:10]

    if x.shape[-1] == 19:
        accel_bias = x[13:16]
        gyro_bias = x[16:19]
    else:
        accel_bias = jnp.zeros_like(x[0:3])
        gyro_bias = jnp.zeros_like(x[0:3])

    accel = imu_accel - accel_bias
    gyro = imu_gyro - gyro_bias
    rot = R.from_quat(quat)
    world_accel = rot.apply(accel) + gravity_vec
    next_pos = pos + vel * dt + 0.5 * world_accel * dt**2
    next_vel = vel + world_accel * dt
    next_quat = (rot * R.from_rotvec(gyro * dt)).as_quat()
    return jnp.concat((next_pos, next_quat, next_vel, gyro, x[13:]), axis=-1)


def _uwb_imu_process_noise(
    dt: float,
    dim_x: int,
    accel_std: Array,
    gyro_std: Array,
    accel_bias_walk_std: Array,
    gyro_bias_walk_std: Array,
) -> Array:
    accel_var = accel_std[..., 0] ** 2
    gyro_var = gyro_std[..., 0] ** 2
    diag = jnp.concat(
        (
            jnp.full((*accel_var.shape, 3), 0.25 * dt**4) * accel_var[..., None],
            jnp.full((*gyro_var.shape, 4), dt**2) * gyro_var[..., None],
            jnp.full((*accel_var.shape, 3), dt**2) * accel_var[..., None],
            jnp.full((*gyro_var.shape, 3), 1.0) * gyro_var[..., None],
        ),
        axis=-1,
    )
    if dim_x == 19:
        accel_bias_var = accel_bias_walk_std[..., 0] ** 2
        gyro_bias_var = gyro_bias_walk_std[..., 0] ** 2
        diag = jnp.concat(
            (
                diag,
                jnp.full((*accel_bias_var.shape, 3), dt) * accel_bias_var[..., None],
                jnp.full((*gyro_bias_var.shape, 3), dt) * gyro_bias_var[..., None],
            ),
            axis=-1,
        )
    return jnp.eye(dim_x, dtype=diag.dtype) * diag[..., None, :]


def _uwb_imu_covariance_predict(
    covariance: Array,
    x: Array,
    imu_accel: Array,
    imu_gyro: Array,
    gravity_vec: Array,
    dt: float,
    noise_cov: Array,
) -> Array:
    dim_x = x.shape[-1]
    flat_x = x.reshape((-1, dim_x))
    flat_accel = imu_accel.reshape((-1, 3))
    flat_gyro = imu_gyro.reshape((-1, 3))
    flat_gravity = gravity_vec.reshape((-1, 3))
    jacobian = jax.jacfwd(_uwb_imu_process_one, argnums=0)
    flat_transition = jax.vmap(jacobian, in_axes=(0, 0, 0, 0, None))(
        flat_x, flat_accel, flat_gyro, flat_gravity, dt
    )
    transition = flat_transition.reshape((*x.shape[:-1], dim_x, dim_x))

    return transition @ covariance @ jnp.swapaxes(transition, -1, -2) + noise_cov


def _range_noise_cov(range_std: Array) -> Array:
    n_ranges = range_std.shape[-1] if range_std.shape[-1] > 1 else 8
    var = range_std[..., 0] ** 2
    return jnp.eye(n_ranges, dtype=range_std.dtype) * var[..., None, None]


def _regularize_covariance(covariance: Array) -> Array:
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    diag = jnp.diagonal(covariance, axis1=-2, axis2=-1)
    diag_update = jnp.maximum(1e-9 - diag, 0.0) + 1e-9
    return (
        covariance
        + jnp.eye(covariance.shape[-1], dtype=covariance.dtype) * diag_update[..., None, :]
    )


def _set_imu_attitude(x: Array, quat: Array, gyro: Array, dt: float) -> Array:
    if x.shape[-1] == 19:
        gyro = gyro - x[..., 16:19]
    quat = (R.from_quat(quat) * R.from_rotvec(gyro * dt)).as_quat()
    return x.at[..., 3:7].set(quat).at[..., 10:13].set(gyro)
