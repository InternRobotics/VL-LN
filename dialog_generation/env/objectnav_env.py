from typing import Optional

import gym
import habitat
import numpy as np
import quaternion
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode
from habitat.core.env import Env
from habitat.core.simulator import Observations
from omegaconf import DictConfig

from .interfaces import SemObservations
from .mp3d_gt import MP3DGTPerception


class ObjectNavEnv(Env):
    """Habitat-style Object Navigation environment wrapper that preprocesses simulator observations into a semantic 
    observation bundle. 

    Args:
        config: OmegaConf DictConfig containing simulator/task configuration.
        dataset: Optional dataset providing episodes for the environment.
        exp_config: Optional experiment-specific configuration object stored for downstream use.
    """
    def __init__(
        self,
        config: "DictConfig",
        dataset: Optional[Dataset[Episode]] = None,
        exp_config=None,
    ):
        super().__init__(config, dataset)
        agent_config = get_agent_config(config.simulator)
        self.min_depth = agent_config.sim_sensors.depth_sensor.min_depth
        self.max_depth = agent_config.sim_sensors.depth_sensor.max_depth
        self._camera_fov = np.deg2rad(agent_config.sim_sensors.depth_sensor.hfov)
        self._fx = self._fy = agent_config.sim_sensors.depth_sensor.width / (
            2 * np.tan(self._camera_fov / 2)
        )
        self._camera_height = agent_config.sim_sensors.rgb_sensor.position[1]

        self.segmentation = MP3DGTPerception(
            self.max_depth, self.min_depth, self._fx, self._fy
        )

        self.config = config
        self.exp_config = exp_config

    def reset(self):
        obs = super().reset()
        self._last_obs = self._preprocess_obs(obs)
        return self._last_obs

    def get_tf_episodic_to_global(self):
        agent_state = self.sim.get_agent_state()
        rotation = agent_state.rotation
        translation = agent_state.position
        rotation_matrix = quaternion.as_rotation_matrix(rotation)
        tf_episodic_to_global = np.eye(4)
        tf_episodic_to_global[:3, :3] = rotation_matrix
        tf_episodic_to_global[:3, 3] = translation
        return tf_episodic_to_global

    def apply_action(
        self,
        action,
    ):
        obs = self.step(action)
        self._last_obs = self._preprocess_obs(obs)
        metric = self.get_metrics()
        return self._last_obs, self.episode_over, metric

    def _preprocess_obs(self, obs: Observations):
        goal_name = (
            self.current_episode.instruction.instance_id[0].split("/")[-1].split("_")[0]
        )
        task_observations = {"goal_name": goal_name}
        sem_obs = SemObservations(
            rgb=obs["rgb"],
            depth=obs["depth"],
            compass=obs["compass"],
            gps=obs["gps"],
            task_observations=task_observations,
            camera_pose=None,
            third_person_image=None,
        )
        targets = [
            self.current_episode.goals[idx].bbox
            for idx, _ in enumerate(self.current_episode.instruction.instance_id)
        ]
        targets = np.array(
            [
                [
                    target[0],
                    min(-target[2], -target[5]),
                    target[1],
                    target[3],
                    max(-target[5], -target[2]),
                    target[4],
                ]
                for target in targets
            ]
        )
        depth = filter_depth(
            sem_obs.depth.reshape(sem_obs.depth.shape[:2]), blur_type=None
        )
        tf_camera_to_global = self.get_tf_episodic_to_global()
        tf_camera_to_global[1, 3] = (
            self._camera_height + self.sim.get_agent_state().position[1]
        )
        tf_camera_to_ply = np.dot(
            np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
            tf_camera_to_global,
        )
        sem_obs.semantic = self.segmentation.predict(depth, targets, tf_camera_to_ply)
        return sem_obs

    @property
    def original_action_space(self):
        return self.action_space
