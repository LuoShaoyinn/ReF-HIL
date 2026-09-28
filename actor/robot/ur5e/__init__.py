"""Lazy UR5e hardware exports.

Learner-only machines can import the actor package without installing camera
or RTDE drivers. Hardware modules are loaded only when a robot export is used.
"""

from importlib import import_module

__all__: list[str] = [
    "RealSenseCamera",
    "RealSenseCameraConfig",
    "RobotiqGripperConfig",
    "URImpedanceControlConfig",
    "RotationRecoveryConfig",
    "UR5eBaseConfig",
    "UR5eBase",
]


_EXPORT_MODULE = {
    "RealSenseCamera": ".realsense_camera",
    "RealSenseCameraConfig": ".realsense_camera",
    "RobotiqGripperConfig": ".robotiq_gripper",
    "URImpedanceControlConfig": ".ur_impedance_control",
    "RotationRecoveryConfig": ".ur_impedance_control",
    "UR5eBaseConfig": ".ur5e_base",
    "UR5eBase": ".ur5e_base",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
