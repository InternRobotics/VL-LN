from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SemObservations:
    # (x, y) where positive x is forward, positive y is translation to left in meters
    gps: np.ndarray  # (3,)
    # positive theta is rotation to left in radians - consistent with robot
    compass: np.ndarray  # (1,)

    # Camera
    # rgb image
    rgb: np.ndarray  # (H, W, 3) in [0, 255]
    # depth image
    depth: np.ndarray  # (H, W) in meters
    # points in camera coordinates
    xyz: Optional[np.ndarray] = None  # (H, W, 3) in camera coordinates
    # semantic mask
    semantic: Optional[
        np.array
    ] = None  # (H, W, num_categories) in [0, num_sem_categories - 1]
    # instance ids mask
    instance: Optional[np.array] = None  # (H, W) in [0, max_int]

    # optional third-person view from simulation
    third_person_image: Optional[np.array] = None

    # pose of the camera in world coordinates
    camera_pose: Optional[np.array] = None
    camera_K: Optional[np.array] = None

    # Proprioreception
    joint: Optional[np.array] = None  # joint positions of the robot
    relative_resting_position: Optional[
        np.array
    ] = None  # end-effector position relative to the desired resting position
    is_holding: Optional[np.array] = None  # whether the agent is holding the object

    # --------------------------------------------------------
    # Untyped task-specific observations
    # --------------------------------------------------------

    task_observations: Optional[Dict[str, Any]] = None  # object_goal, goal_name
