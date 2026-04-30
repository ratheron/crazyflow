from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from jax import Array, Device

from crazyflow.control import Control
from crazyflow.control.mellinger import (
    MellingerAttitudeData,
    MellingerForceTorqueData,
    MellingerStateData,
)
from crazyflow.sim.physics import (
    FirstPrinciplesData,
    Physics,
    SoRpyData,
    SoRpyRotorData,
    SoRpyRotorDragData,
)

DEFAULT_UWB_BASE_STATIONS = (
    (-2.5, -2.5, 0.0),
    (-2.5, 2.5, 0.0),
    (2.5, -2.5, 0.0),
    (2.5, 2.5, 0.0),
    (-2.5, -2.5, 3.0),
    (-2.5, 2.5, 3.0),
    (2.5, -2.5, 3.0),
    (2.5, 2.5, 3.0),
)


@dataclass
class SimState:
    pos: Array  # (N, M, 3)
    """Position of the drone's center of mass."""
    quat: Array  # (N, M, 4)
    """Quaternion of the drone's orientation."""
    vel: Array  # (N, M, 3)
    """Velocity of the drone's center of mass in the world frame."""
    ang_vel: Array  # (N, M, 3)
    """Angular velocity of the drone's center of mass in the world frame."""
    force: Array  # (N, M, 3)  # CoM force
    """Force applied to the drone's center of mass in the world frame."""
    torque: Array  # (N, M, 3)  # CoM torque
    """Torque applied to the drone's center of mass in the world frame."""
    rotor_vel: Array  # (N, M, 4)  # Motor forces along body frame z axis
    """Motor forces along body frame z axis."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimState:
        """Create a default set of states for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        q_identity = jnp.zeros((n_worlds, n_drones, 4), device=device)
        q_identity = q_identity.at[..., -1].set(1.0)
        rotor_vel = jnp.zeros((n_worlds, n_drones, 4), device=device)
        return SimState(
            pos=zeros_3d,
            quat=q_identity,
            vel=zeros_3d,
            ang_vel=zeros_3d,
            force=zeros_3d,
            torque=zeros_3d,
            rotor_vel=rotor_vel,
        )


@dataclass
class SimEstimates:
    pos: Array  # (N, M, 3)
    """Estimated position of the drone's center of mass."""
    quat: Array  # (N, M, 4)
    """Estimated quaternion of the drone's orientation."""
    vel: Array  # (N, M, 3)
    """Estimated velocity of the drone's center of mass in the world frame."""
    ang_vel: Array  # (N, M, 3)
    """Estimated angular velocity of the drone in the world frame."""
    accel_bias: Array  # (N, M, 3)
    """Estimated accelerometer bias in m/s^2."""
    gyro_bias: Array  # (N, M, 3)
    """Estimated gyroscope bias in rad/s."""
    covariance: Array  # (N, M, 19, 19)
    """Full-state estimator covariance for [pos, quat, vel, ang_vel, accel_bias, gyro_bias]."""
    use_for_control: Array  # (N, 1, 1)
    """Whether the controller should consume estimated state for each world."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimEstimates:
        """Create a default state estimate for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        q_identity = jnp.zeros((n_worlds, n_drones, 4), device=device)
        q_identity = q_identity.at[..., -1].set(1.0)
        cov_diag = jnp.array(
            [
                2.5e-3,
                2.5e-3,
                2.5e-3,
                1e-6,
                1e-6,
                1e-6,
                1e-6,
                1e-2,
                1e-2,
                1e-2,
                1e-4,
                1e-4,
                1e-4,
                1e-4,
                1e-4,
                1e-4,
                1e-8,
                1e-8,
                1e-8,
            ],
            device=device,
        )
        covariance = jnp.tile(jnp.diag(cov_diag)[None, None, :, :], (n_worlds, n_drones, 1, 1))
        use_for_control = jnp.zeros((n_worlds, 1, 1), dtype=jnp.bool_, device=device)
        return SimEstimates(
            pos=zeros_3d,
            quat=q_identity,
            vel=zeros_3d,
            ang_vel=zeros_3d,
            accel_bias=zeros_3d,
            gyro_bias=zeros_3d,
            covariance=covariance,
            use_for_control=use_for_control,
        )


@dataclass
class UWBData:
    base_stations: Array  # (8, 3)
    """UWB base station positions in the world frame."""
    ranges: Array  # (N, M, 8)
    """Latest measured UWB ranges from each drone to each base station."""
    range_std: Array  # (N, M, 1)
    """Standard deviation of the range noise in meters."""
    estimator_range_std: Array  # (N, M, 1)
    """Standard deviation assumed by the estimator for UWB ranges."""
    steps: Array  # (N, 1)
    """Last simulation steps that UWB ranges were updated."""
    freq: int = field(pytree_node=False)
    """Frequency of UWB communication."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, device: Device, range_std: float = 0.05
    ) -> UWBData:
        """Create default UWB sensing data."""
        base_stations = jnp.array(DEFAULT_UWB_BASE_STATIONS, device=device)
        ranges = jnp.zeros((n_worlds, n_drones, base_stations.shape[0]), device=device)
        range_std = jnp.full((n_worlds, n_drones, 1), range_std, device=device)
        estimator_range_std = range_std
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        return UWBData(
            base_stations=base_stations,
            ranges=ranges,
            range_std=range_std,
            estimator_range_std=estimator_range_std,
            steps=steps,
            freq=freq,
        )


@dataclass
class IMUData:
    accel: Array  # (N, M, 3)
    """Latest accelerometer measurement as specific force in the body frame."""
    gyro: Array  # (N, M, 3)
    """Latest gyroscope measurement in the body frame."""
    prev_vel: Array  # (N, M, 3)
    """Previous ground-truth velocity kept for IMU models that finite-difference velocity."""
    accel_bias: Array  # (N, M, 3)
    """Current accelerometer bias in m/s^2."""
    gyro_bias: Array  # (N, M, 3)
    """Current gyroscope bias in rad/s."""
    accel_std: Array  # (N, M, 1)
    """Standard deviation of accelerometer noise in m/s^2."""
    gyro_std: Array  # (N, M, 1)
    """Standard deviation of gyroscope noise in rad/s."""
    accel_bias_walk_std: Array  # (N, M, 1)
    """Accelerometer bias random-walk standard deviation in m/s^2/sqrt(s)."""
    gyro_bias_walk_std: Array  # (N, M, 1)
    """Gyroscope bias random-walk standard deviation in rad/s/sqrt(s)."""

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        device: Device,
        accel_std: float = 0.05,
        gyro_std: float = 0.005,
        accel_bias_walk_std: float = 0.0,
        gyro_bias_walk_std: float = 0.0,
    ) -> IMUData:
        """Create default IMU sensing data."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        accel_std = jnp.full((n_worlds, n_drones, 1), accel_std, device=device)
        gyro_std = jnp.full((n_worlds, n_drones, 1), gyro_std, device=device)
        accel_bias_walk_std = jnp.full((n_worlds, n_drones, 1), accel_bias_walk_std, device=device)
        gyro_bias_walk_std = jnp.full((n_worlds, n_drones, 1), gyro_bias_walk_std, device=device)
        return IMUData(
            accel=zeros_3d,
            gyro=zeros_3d,
            prev_vel=zeros_3d,
            accel_bias=zeros_3d,
            gyro_bias=zeros_3d,
            accel_std=accel_std,
            gyro_std=gyro_std,
            accel_bias_walk_std=accel_bias_walk_std,
            gyro_bias_walk_std=gyro_bias_walk_std,
        )


@dataclass
class SimStateDeriv:
    vel: Array  # (N, M, 3)
    """Derivative of the position of the drone's center of mass."""
    ang_vel: Array  # (N, M, 3)
    """Derivative of the quaternion of the drone's orientation as angular velocity."""
    acc: Array  # (N, M, 3)
    """Derivative of the velocity of the drone's center of mass."""
    ang_acc: Array  # (N, M, 3)
    """Derivative of the angular velocity of the drone's center of mass."""
    rotor_acc: Array  # (N, M, 4)
    """Derivative of the rotor velocity."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, device: Device) -> SimStateDeriv:
        """Create a default set of state derivatives for the simulation."""
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        zeros_4d = jnp.zeros((n_worlds, n_drones, 4), device=device)
        return SimStateDeriv(
            vel=zeros_3d, ang_vel=zeros_3d, acc=zeros_3d, ang_acc=zeros_3d, rotor_acc=zeros_4d
        )


@typing.runtime_checkable
class ControlData(typing.Protocol):
    staged_cmd: Array  # (N, M, X)
    """Staged control command for the drone.

    The most recent control input gets staged here until the next controller tick and is then
    committed to the cmd field.
    """
    cmd: Array  # (N, M, X)
    """Control command for the drone."""
    staged_cmd: Array  # (N, M, X)
    """Staged control command for the drone."""
    steps: Array  # (N, 1)
    """Last simulation steps that the state control command was applied."""
    freq: int
    """Frequency of the state control command."""
    # Parameters for the controller
    params: tuple[typing.Any, ...]


@dataclass
class SimControls:
    mode: Control = field(pytree_node=False)
    """Control mode of the simulation."""
    state: ControlData | None
    """State control data."""
    attitude: ControlData | None
    """Attitude control data."""
    force_torque: ControlData | None
    """Force and torque control data."""
    rotor_vel: Array  # (N, M, 4)
    """Desired motor speed."""

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        control: Control,
        drone_model: str,
        state_freq: int | None,
        attitude_freq: int | None,
        force_torque_freq: int | None,
        device: Device,
    ) -> SimControls:
        """Create a default set of controls for the simulation."""
        rotor_vel = jnp.zeros((n_worlds, n_drones, 4), device=device)
        match control:
            case Control.state:
                state = MellingerStateData.create(
                    n_worlds, n_drones, state_freq, drone_model, device
                )
                attitude = MellingerAttitudeData.create(
                    n_worlds, n_drones, attitude_freq, drone_model, device
                )
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone_model, device
                )
                return SimControls(
                    mode=control,
                    state=state,
                    attitude=attitude,
                    force_torque=force_torque,
                    rotor_vel=rotor_vel,
                )
            case Control.attitude:
                attitude = attitude = MellingerAttitudeData.create(
                    n_worlds, n_drones, attitude_freq, drone_model, device
                )
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone_model, device
                )
                return SimControls(
                    mode=control,
                    state=None,
                    attitude=attitude,
                    force_torque=force_torque,
                    rotor_vel=rotor_vel,
                )
            case Control.force_torque:
                force_torque = MellingerForceTorqueData.create(
                    n_worlds, n_drones, force_torque_freq, drone_model, device
                )
                return SimControls(
                    mode=control,
                    state=None,
                    attitude=None,
                    force_torque=force_torque,
                    rotor_vel=rotor_vel,
                )
            case Control.rotor_vel:
                return SimControls(
                    mode=control, state=None, attitude=None, force_torque=None, rotor_vel=rotor_vel
                )
            case _:
                raise ValueError(f"Control mode {control} not implemented")


class SimParams(typing.Protocol):
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, physics: Physics, drone_model: str, device: Device
    ) -> SimParams:
        """Create a default set of parameters for the simulation."""
        match physics:
            case Physics.first_principles:
                return FirstPrinciplesData.create(n_worlds, n_drones, drone_model, device)
            case Physics.so_rpy:
                return SoRpyData.create(n_worlds, n_drones, drone_model, device)
            case Physics.so_rpy_rotor:
                return SoRpyRotorData.create(n_worlds, n_drones, drone_model, device)
            case Physics.so_rpy_rotor_drag:
                return SoRpyRotorDragData.create(n_worlds, n_drones, drone_model, device)
            case _:
                raise ValueError(f"Physics mode {physics} not implemented")


@dataclass
class SimCore:
    freq: int = field(pytree_node=False)
    """Frequency of the simulation."""
    device: Device = field(pytree_node=False)
    """Device of the simulation."""
    steps: Array  # (N, 1)
    """Simulation steps taken since the last reset."""
    n_worlds: int = field(pytree_node=False)
    """Number of worlds in the simulation."""
    n_drones: int = field(pytree_node=False)
    """Number of drones in the simulation."""
    drone_ids: Array  # (1, M)
    """MuJoCo IDs of the drones in the simulation."""
    rng_key: Array  # (N, 1)
    """Random number generator key for the simulation."""
    mjx_synced: Array  # (1,)
    """Whether the simulation data is synchronized with the MuJoCo model."""

    @staticmethod
    def create(
        freq: int,
        n_worlds: int,
        n_drones: int,
        drone_ids: Array,
        rng_key: int | Array,
        device: Device,
    ) -> SimCore:
        """Create a default set of core simulation parameters."""
        steps = jnp.zeros((n_worlds, 1), dtype=jnp.int32, device=device)
        if isinstance(rng_key, int):  # Only convert to an PRNG key if its not already one
            rng_key = jax.random.key(rng_key)
        rng_key = jax.device_put(rng_key, device)
        return SimCore(
            freq=freq,
            device=device,
            steps=steps,
            n_worlds=n_worlds,
            n_drones=n_drones,
            drone_ids=jnp.array(drone_ids, dtype=jnp.int32, device=device),
            rng_key=rng_key,
            mjx_synced=jnp.array(False, dtype=jnp.bool_, device=device),
        )


@dataclass
class SimData:
    states: SimState
    """State of the simulation."""
    states_deriv: SimStateDeriv
    """Derivative of the state of the simulation."""
    estimates: SimEstimates
    """Estimated state of the simulation."""
    uwb: UWBData
    """UWB sensing data."""
    imu: IMUData
    """IMU sensing data."""
    controls: SimControls
    """Drone controller data."""
    params: SimParams
    """Drone parameters."""
    core: SimCore
    """Core parameters of the simulation."""


import numpy as np


@dataclass
class UKFData:
    """TODO."""

    pos: Array
    quat: Array
    vel: Array
    ang_vel: Array
    rotor_vel: Array | None
    dist_f: Array | None
    dist_t: Array | None
    covariance: Array  # Covariance matrix

    sigmas_f: Array
    sigmas_h: Array

    u: Array  # input
    z: Array  # measurement
    dt: float
    accel_bias: Array | None = None
    gyro_bias: Array | None = None

    @classmethod
    def create_empty(
        cls,
        rotor_vel: bool = False,
        dist_f: bool = False,
        dist_t: bool = False,
        accel_bias: bool = False,
        gyro_bias: bool = False,
        dim_u: int = 4,
        dim_z: int = 7,
    ) -> UKFData:
        """TODO."""
        pos = np.zeros(3)
        quat = np.array([0, 0, 0, 1])
        vel = np.zeros(3)
        ang_vel = np.zeros(3)
        dim_x = 13
        if rotor_vel:
            rotor_vel = np.zeros(4)
            dim_x = dim_x + 4
        else:
            rotor_vel = None
        if dist_f:
            dist_f = np.zeros(3)
            dim_x = dim_x + 3
        else:
            dist_f = None
        if dist_t:
            dist_t = np.zeros(3)
            dim_x = dim_x + 3
        else:
            dist_t = None
        if accel_bias:
            accel_bias = np.zeros(3)
            dim_x = dim_x + 3
        else:
            accel_bias = None
        if gyro_bias:
            gyro_bias = np.zeros(3)
            dim_x = dim_x + 3
        else:
            gyro_bias = None

        covariance = np.eye(dim_x)

        sigmas_f = np.zeros((2 * dim_x + 1, dim_x))
        sigmas_h = np.zeros((2 * dim_x + 1, dim_z))

        u = np.zeros(dim_u)  # input
        z = np.zeros(dim_z)  # measurement
        dt = 1

        return cls(
            pos,
            quat,
            vel,
            ang_vel,
            rotor_vel,
            dist_f,
            dist_t,
            covariance,
            sigmas_f,
            sigmas_h,
            u,
            z,
            dt,
            accel_bias,
            gyro_bias,
        )

    @classmethod
    def create(
        cls,
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array,
        rotor_vel: Array | None = None,
        dist_f: Array | None = None,
        dist_t: Array | None = None,
        accel_bias: Array | None = None,
        gyro_bias: Array | None = None,
    ) -> UKFData:
        """TODO."""
        dim_x = 13
        if rotor_vel is not None:
            dim_x = dim_x + 4
        if dist_f is not None:
            dim_x = dim_x + 3
        if dist_t is not None:
            dim_x = dim_x + 3
        if accel_bias is not None:
            dim_x = dim_x + 3
        if gyro_bias is not None:
            dim_x = dim_x + 3

        covariance = np.eye(dim_x)

        sigmas_f = np.zeros((2 * dim_x + 1, dim_x))
        sigmas_h = np.zeros((2 * dim_x + 1, 7))

        u = np.zeros(4)  # input
        z = np.zeros(7)  # measurement
        dt = 1

        return cls(
            pos,
            quat,
            vel,
            ang_vel,
            rotor_vel,
            dist_f,
            dist_t,
            covariance,
            sigmas_f,
            sigmas_h,
            u,
            z,
            dt,
            accel_bias,
            gyro_bias,
        )

    @classmethod
    def as_state_array(cls, data: UKFData) -> Array:
        """Returns the state as an array."""
        xp = data.pos.__array_namespace__()
        x = xp.concat((data.pos, data.quat, data.vel, data.ang_vel), axis=-1)
        if data.rotor_vel is not None:
            x = xp.concat((x, data.rotor_vel), axis=-1)
        if data.dist_f is not None:
            x = xp.concat((x, data.dist_f), axis=-1)
        if data.dist_t is not None:
            x = xp.concat((x, data.dist_t), axis=-1)
        if data.accel_bias is not None:
            x = xp.concat((x, data.accel_bias), axis=-1)
        if data.gyro_bias is not None:
            x = xp.concat((x, data.gyro_bias), axis=-1)
        return x

    @classmethod
    def from_state_array(cls, data: UKFData, array: Array) -> UKFData:
        """Updates data in the given structure based on a given array."""
        pos = array[..., 0:3]
        quat = array[..., 3:7]
        vel = array[..., 7:10]
        ang_vel = array[..., 10:13]
        idx = 13
        if data.rotor_vel is not None:
            rotor_vel = array[..., idx : idx + 4]
            idx = idx + 4
        else:
            rotor_vel = None
        if data.dist_f is not None:
            dist_f = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            dist_f = None
        if data.dist_t is not None:
            dist_t = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            dist_t = None
        if data.accel_bias is not None:
            accel_bias = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            accel_bias = None
        if data.gyro_bias is not None:
            gyro_bias = array[..., idx : idx + 3]
        else:
            gyro_bias = None

        return data.replace(
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            rotor_vel=rotor_vel,
            dist_f=dist_f,
            dist_t=dist_t,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        )

    @classmethod
    def get_state_dim(cls, data: UKFData) -> int:
        """Returns the dimension of the state."""
        dim_x = 13
        if data.rotor_vel is not None:
            dim_x = dim_x + 4
        if data.dist_f is not None:
            dim_x = dim_x + 3
        if data.dist_t is not None:
            dim_x = dim_x + 3
        if data.accel_bias is not None:
            dim_x = dim_x + 3
        if data.gyro_bias is not None:
            dim_x = dim_x + 3
        return dim_x
