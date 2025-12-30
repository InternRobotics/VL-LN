import json
import os
import random
import re
import time

import torch
from torch import Tensor

random.seed(2025)
import argparse

import numpy as np

np.random.seed(2025)
from collections import defaultdict
from typing import Tuple

import habitat
import habitat_sim
import matplotlib.pyplot as plt
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import append_text_underneath_image
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from dialog_generation.dialog_utils import (
    generate_dialog,
    get_navigable_path,
    get_shortest_path,
)
from dialog_generation.env.interfaces import SemObservations
from dialog_generation.env.modules.dialog_episodes import (
    AgentPosition,
    DialogEpisode,
    DialogGoal,
    DialogViewLocation,
)
from dialog_generation.env.objectnav_env import ObjectNavEnv
from dialog_generation.other_utils import (
    filter_depth,
    get_axis_align_matrix,
    get_config,
    get_intrinsic_matrix,
    prepare_dirs,
    xyz_yaw_pitch_to_tf_matrix,
    xyz_yaw_to_tf_matrix,
)
from dist import init_distributed_mode
from vlfm.mapping.voro_obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import closest_point_within_threshold, rho_theta
from vlfm.utils.img_utils import reorient_rescale_map

DEFAULT_IMAGE_TOKEN = "<image>"


class LlavaAgent(habitat.Agent):
    def __init__(self, config, best_p=0.5):
        # constants
        self._max_steps = config.max_steps
        self._pointnav_stop_radius = config.pointnav_stop_radius

        self.waypoint_ids = {}
        for i in range(1, 10):
            self.waypoint_ids[f"<wp{i:01}>"] = i

        # initialization
        self.tf_episodic_to_global = np.eye(4)
        self._acyclic_enforcer = AcyclicEnforcer()
        self._done_initializing = False
        self._memory = []
        self._object_masks = None

        # obstacle map
        self._obstacle_map = ObstacleMap(
            min_height=0.61,
            max_height=0.88,
            agent_radius=0.18,
            area_thresh=1.5,
            hole_area_thresh=100000,
            size=1000,
        )

        self.best_p = best_p

        # pointnav policy
        self._pointnav_policy = WrappedPointNavResNetPolicy(args.pointnav_policy_path)
        self._pointnav_depth_image_shape = (224, 224)

    def reset(
        self, start_pos: np.ndarray, start_rotation: np.ndarray, episode: DialogEpisode
    ):
        self._episode = episode
        self._num_steps = 0
        self._start_pos = start_pos
        self._start_rotation = start_rotation
        self._start_matrix = xyz_yaw_to_tf_matrix(
            self._start_pos,
            2 * np.arctan2(self._start_rotation.y, self._start_rotation.w),
        )

        self._called_stop = True
        self._obstacle_map.reset()
        self._last_frontier = np.zeros(2)
        self._last_goal = np.zeros(2)
        self.update_voronois = self.update_frontiers = True

        self.look_around_actions = []
        self.transformation_matrix = np.eye(
            4
        )  # transformation matrix from the episodic frame to the simulation frame

        self.vis_frames = []
        self._done_initializing = False

        self.move_best = False

    def parse_waypoint(self, input_str: str) -> int:
        waypoint = re.findall(
            r"<wp[1-9]>", input_str
        )  # define the regex pattern to extract all <wp1> to <wp9>
        waypoint = list(set(waypoint))  # create a list to store the extracted waypoints

        if not waypoint or len(waypoint) != 1:
            return None
        if waypoint[0] not in self.waypoint_ids:
            return None
        waypoint_id = self.waypoint_ids[waypoint[0]]
        return waypoint_id

    def _cache_observations(self, **kwargs):
        # cache the observations
        self._observations_cache = kwargs

        # update frontiers with the latest obstacle map
        self._obstacle_map.update_map(
            *kwargs["obstacle_map_rgbd"][0],
            update_voronois=False,
            update_frontiers=self.update_frontiers,
        )
        self._observations_cache.update(
            {
                "frontier_sensor": self._obstacle_map.frontiers,
            }
        )
        self._obstacle_map.update_agent_traj(
            kwargs["robot_xy"], kwargs["robot_heading"]
        )

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor(
            [[rho, theta]], device="cuda", dtype=torch.float32
        )
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (
                    self._pointnav_depth_image_shape[0],
                    self._pointnav_depth_image_shape[1],
                ),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        if rho < self._pointnav_stop_radius:
            self._called_stop = True
        else:
            self._called_stop = False

        if self._called_stop and stop:
            return 0
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def get_geo_distance(
        self, start_position, target_positions: list, object_info: dict
    ):
        start_position = [float(i) for i in start_position]
        target_positions = sorted(
            target_positions,
            key=lambda x: np.linalg.norm(
                np.array(x["agent_state"]["position"])
                - np.array(object_info["position"])
            ),
        )
        success = False
        while not success and len(target_positions) > 0:
            target_position = target_positions.pop(0)
            shortest_path = habitat_sim.ShortestPath()
            shortest_path.requested_start = start_position
            shortest_path.requested_end = target_position["agent_state"]["position"]

            success = self.sim.pathfinder.find_path(shortest_path)
        if success:
            return True, shortest_path
        else:
            return False, None

    def get_frontier_in_global(self, frontiers: np.ndarray) -> np.ndarray:
        tmp = (
            self._start_matrix[:3, :3]
            @ (
                np.hstack(
                    (
                        frontiers.reshape(-1, 2),
                        np.zeros(frontiers.reshape(-1, 2).shape[0]).reshape(-1, 1),
                    )
                )
            ).T
        ).T
        frontiers_global = np.zeros_like(tmp)
        frontiers_global[:, 0] = self._start_pos[0] - tmp[:, 1]
        frontiers_global[:, 2] = self._start_pos[2] - tmp[:, 0]
        frontiers_global[:, 1] = self.sim.get_agent_state().position[1]
        return frontiers_global

    def best_point(self, frontiers: np.ndarray) -> int:
        frontiers_global = self.get_frontier_in_global(frontiers)
        # get nearest goal
        best_path = None
        best_idx = -1
        unreachable_frontiers = defaultdict(int)
        for goal in self._episode.goals:
            for frontier_idx, frontier in enumerate(frontiers_global):
                success, shortest_path = self.get_geo_distance(
                    frontier,
                    [
                        {"agent_state": {"position": vp.agent_state.position}}
                        for vp in goal.view_points
                    ],
                    {"position": goal.position},
                )
                if not success:
                    unreachable_frontiers[frontier_idx] += 1
                    continue
                if (
                    best_path is None
                    or shortest_path.geodesic_distance < best_path.geodesic_distance
                ):
                    best_path = shortest_path
                    best_idx = frontier_idx
        unreachable_frontiers = [
            frontier_idx
            for frontier_idx, count in unreachable_frontiers.items()
            if count - len(self._episode.goals) == 0
        ]
        if best_path is None:
            return -1, [i for i in range(len(frontiers_global))]
        return best_idx, unreachable_frontiers

    def get_frontier(self, frontiers: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        self._last_frontier = self._last_goal
        closest_index = closest_point_within_threshold(
            frontiers, self._last_frontier, threshold=np.inf
        )
        best_index, unreachable_frontiers = self.best_point(frontiers)

        if np.random.rand() < self.best_p and best_index >= 0:
            self.move_best = True
            curr_index = best_index
        else:
            self.move_best = False
            curr_index = np.random.choice(unreachable_frontiers + [closest_index])

        self._last_frontier = frontiers[curr_index]

        return frontiers[curr_index]

    def _initialize(self):
        self._done_initializing = not self._num_steps < 11
        return 2  # turn left

    def _explore(self):
        frontiers = self._obstacle_map.frontiers
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found.")
            return 0
        if self._called_stop or np.random.rand() < 0.1:
            goal = self.get_frontier(frontiers)
        else:
            goal = self._last_goal
        pointnav_action = self._pointnav(goal, stop=False)

        return pointnav_action

    def act(self, sim: HabitatSim) -> Tuple[int, str]:
        self.sim = sim
        goal = None

        if not self._done_initializing:
            action = self._initialize()  # turn around
        elif goal is None:
            action = self._explore()
        else:
            action = self._pointnav(goal, stop=True)

        action = (
            action.detach().cpu().numpy()[0]
            if isinstance(action, torch.Tensor)
            else action
        )
        action = action[0] if hasattr(action, "__len__") else action

        self._num_steps += 1
        self._last_goal = goal if goal is not None else self._last_goal

        return action


def create_frame(
    agent,
    metrics,
    mode: str,
    step: int,
):
    goal = agent._last_goal
    markers = []
    frontiers = agent._observations_cache["frontier_sensor"]
    for idx, frontier in enumerate(frontiers):
        marker_kwargs = {
            "radius": 5,  # self._circle_marker_radius,
            "thickness": 2,  # self._circle_marker_thickness,
            "color": (0, 0, 255),  # self._frontier_color,
            "idx": idx,
        }
        markers.append((frontier[:2], marker_kwargs))
    if not np.array_equal(agent._last_goal, np.zeros(2)):
        # Draw the pointnav goal on to the cost map
        if any(np.array_equal(agent._last_goal, frontier) for frontier in frontiers):
            color = (0, 255, 255)  # self._selected__frontier_color
        else:
            color = (0, 255, 0)  # self._target_object_color
        marker_kwargs = {"radius": 5, "thickness": 2, "color": color, "idx": -1}
        markers.append((agent._last_goal, marker_kwargs))

    combined_height = agent._observations_cache["rgb"].shape[0]

    # obstacle map
    vis_obs_map = agent._obstacle_map.visualize(None, None, goal)
    for pos, marker_kwargs in markers:
        vis_obs_map = agent._obstacle_map._traj_vis.draw_circle(
            vis_obs_map, pos, **marker_kwargs
        )
    vis_obs_map = reorient_rescale_map(vis_obs_map)
    vis_obs_map = np.flipud(vis_obs_map)
    vis_obs_map = np.fliplr(vis_obs_map)

    # topdown map
    vis_topdown_map = maps.colorize_draw_agent_and_fit_to_height(
        metrics["top_down_map"], combined_height
    )

    imgs = [vis_topdown_map, vis_obs_map, agent._observations_cache["rgb"]]
    resized_imgs = []
    for img in imgs:
        img = Image.fromarray(img)
        img = img.resize(
            (int(img.width * combined_height / img.height), combined_height)
        )
        img = np.array(img)
        resized_imgs.append(img)

    combined_width = sum([img.shape[1] for img in resized_imgs])
    combined_pil = Image.new("RGB", (combined_width, combined_height))

    x_offset = 0
    for img in resized_imgs:
        combined_pil.paste(Image.fromarray(img), (x_offset, 0))
        x_offset += img.shape[1]

    frame = np.array(combined_pil)

    # append text to the frame
    frame = append_text_underneath_image(frame, f"step: {step}")
    frame = append_text_underneath_image(
        frame, f"robot_xy: {agent._observations_cache['robot_xy']}"
    )
    frame = append_text_underneath_image(frame, f"mode: {mode}")

    return frame


def get_save_observations(
    env: ObjectNavEnv,
    observations: SemObservations,
    step: int,
    params: dict,
    is_down: bool = False,
):
    idx = params["idx"]
    scene_id = params["scene_id"]
    ep_id = params["ep_id"]
    initial_height = params["initial_height"]

    rgb = observations.rgb
    depth = observations.depth
    x, y = observations.gps
    camera_yaw = observations.compass[0]
    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
    depth = (
        depth * (params["max_depth"] - params["min_depth"]) + params["min_depth"]
    )  # in m

    depth = depth * 1000
    rgb_path = os.path.join(
        params["output_path"] + "_30down" if is_down else params["output_path"],
        f"{scene_id}_{idx}_{ep_id:04d}",
        "rgb_images",
        f"{step:03d}.jpg",
    )
    depth_path = os.path.join(
        params["output_path"] + "_30down" if is_down else params["output_path"],
        f"{scene_id}_{idx}_{ep_id:04d}",
        "depth_images",
        f"{step:03d}.png",
    )

    agent_position = env.sim.get_agent_state().position
    height = agent_position[1] - initial_height
    camera_pitch = np.deg2rad(30) if is_down else np.deg2rad(0)
    camera_position = np.array([x, -y, params["camera_height"] + height])
    pose_episodic = (
        xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, camera_pitch)
        @ get_axis_align_matrix()
    )
    rgb_key = "/".join(rgb_path.split("/")[-4:])
    tf_camera_to_episodic = (
        xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
        @ get_axis_align_matrix()
        if is_down
        else xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(0))
        @ get_axis_align_matrix()
    )

    tf_camera_to_episodic_agent = (
        xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
        if is_down
        else xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(0))
    )

    return (
        rgb,
        depth,
        rgb_path,
        depth_path,
        pose_episodic,
        [x, -y, height, camera_yaw],
        rgb_key,
        camera_position,
        tf_camera_to_episodic,
        tf_camera_to_episodic_agent,
    )


def find_stairs(height):
    height_list = []
    for i, h in enumerate(height):
        if i == 0:
            height_list.append((i, h, 1))
            continue
        if abs(h - height_list[-1][1]) < 0.1:
            height_list[-1] = (
                height_list[-1][0],
                height_list[-1][1],
                height_list[-1][2] + 1,
            )
        else:
            height_list.append((i, h, 1))
    for i, t in enumerate(height_list):
        if i == 0:
            continue
        assert height_list[i - 1][0] + height_list[i - 1][-1] == height_list[i][0]
    return height_list


def calculate_path_length_noheight(path):
    accumulated_length = [0]
    for i, p in enumerate(path[1:]):
        accumulated_length.append(
            accumulated_length[i]
            + np.linalg.norm(
                np.array([p[0], p[2]]) - np.array([path[i][0], path[i][2]])
            )
        )
    return accumulated_length


def make_dialog(
    env: ObjectNavEnv,
    question_type: str,
    ask_num: int,
    step: int,
    path_node_to_frame: list,
    history: list,
    nav_camera: list,
    object_dict: dict = None,
    region_dict: dict = None,
    height_list: list = None,
    is_goal: bool = False,
    questioned_path: list = None,
    question_yaw: float = None,
):
    instance = env.current_episode.instruction.instance_id[0]
    if f"{step:03d}" not in path_node_to_frame:
        path_node_to_frame.append(f"{step:03d}")
    nav_idx = path_node_to_frame.index(f"{step:03d}")
    if question_type == "information":
        attributes = {
            a: i
            for a, i in object_dict[instance]["unique_description"].items()
            if a in ["color", "texture", "material", "shape", "placement"]
            and len(i) > 0
        }
        attributes["room"] = object_dict[instance]["room"]
        nearby_objects = [
            object_dict[obj]["unique_description"]["fine grained category"].lower()
            for obj, _ in object_dict[instance]["nearby_objects"].items()
            if isinstance(object_dict[obj]["unique_description"], dict)
        ]
        if len(nearby_objects) > 0:
            attributes["nearby objects"] = ", ".join(nearby_objects)
        unknown_attributes = set(attributes.keys()) - set(
            env.current_episode.instruction.instruction_info
        )
        if len(unknown_attributes) == 0:
            return ask_num, history, nav_camera

        attribute_to_ask = (
            "room"
            if "room" in unknown_attributes
            else np.random.choice(list(unknown_attributes))
        )
        attribute_to_ask = {
            "attribute": attribute_to_ask,
            "content": attributes[attribute_to_ask],
        }
        question, answer = generate_dialog(
            env,
            "information",
            object_dict=object_dict,
            attribute_to_ask=attribute_to_ask,
        )
    elif question_type == "local":
        question, answer = generate_dialog(
            env,
            "local",
            object_dict,
            region_dict,
            questioned_path,
            yaw=question_yaw,
            height_list=height_list,
        )
    elif question_type == "stairs":
        question, answer = generate_dialog(env, "stairs", height_list=height_list)
    elif question_type == "disambiguation":
        question, answer = generate_dialog(env, "disambiguation", is_goal=is_goal)
    else:
        raise Exception(f"Invalid question type: {question_type}")

    ask_num += 1
    history.extend(
        [
            {
                "role": "navigator",
                "message": question,
                "nav_idx": nav_idx,
                "true_idx": step,
            },
            {"role": "oracle", "message": answer, "nav_idx": nav_idx, "true_idx": step},
        ]
    )
    rotation = env.sim.get_agent_state().rotation
    position = env.sim.get_agent_state().position
    nav_camera.append(
        {
            "nav_idx": nav_idx,
            "true_idx": step,
            "dialog_idx": len(history) / 2 - 1,
            "message": {
                "rotation": [rotation.w, rotation.x, rotation.y, rotation.z],
                "position": [float(i) for i in position],
            },
        }
    )

    return ask_num, history, nav_camera


def qwen_move_to_target(  # noqa: C901
    env: ObjectNavEnv,
    episode: DialogEpisode,
    agent: LlavaAgent,
    params: dict,
    object_dict: dict = None,
    region_dict: dict = None,
    normal_category: list = None,
):
    # revise episode
    instance = episode.instruction.instance_id[0]
    if episode.object_category in normal_category:
        candidates = [
            k
            for k, v in object_dict.items()
            if v["category"].lower() == episode.object_category and k != instance
        ]
        for candidate in candidates:
            candidate_info = object_dict[candidate]
            min_x, min_y, min_z = (
                min(candidate_info["min_points"][0], candidate_info["max_points"][0]),
                min(candidate_info["min_points"][1], candidate_info["max_points"][1]),
                min(candidate_info["min_points"][2], candidate_info["max_points"][2]),
            )
            max_x, max_y, max_z = (
                max(candidate_info["min_points"][0], candidate_info["max_points"][0]),
                max(candidate_info["min_points"][1], candidate_info["max_points"][1]),
                max(candidate_info["min_points"][2], candidate_info["max_points"][2]),
            )
            goal = {
                "position": candidate_info["position"],
                "radius": np.linalg.norm(
                    np.array([min_x, min_z]) - np.array([max_x, max_z])
                )
                / 2,
                "bbox": [min_x, min_y, min_z, max_x, max_y, max_z],
            }
            view_points = [
                DialogViewLocation(
                    **{
                        "agent_state": AgentPosition(
                            **{"position": candidate_info["position"]}
                        )
                    }
                )
            ]
            goal["view_points"] = view_points
            episode.goals.append(DialogGoal(**goal))
            episode.instruction.instance_id.append(candidate)
    env.current_episode = episode
    observations = env.reset()
    if "instance" in params["task"]:
        check_path, success = get_navigable_path(
            env,
            env.sim.get_agent_state().position,
            [
                {"agent_state": {"position": vp.agent_state.position}}
                for vp in env.current_episode.goals[0].view_points
            ],
            {"position": env.current_episode.goals[0].position},
        )
        assert success, "No path from start to end."
        height_list = find_stairs([i[1] for i in check_path])
        begin_length = calculate_path_length_noheight(
            check_path[: height_list[0][2] + 1]
        )[-1]
        middle_length = calculate_path_length_noheight(
            check_path[height_list[0][2] : height_list[-1][0]]
        )[-1]
        end_length = calculate_path_length_noheight(
            check_path[height_list[-1][0] - 1 :]
        )[-1]
        if abs(height_list[0][1] - height_list[-1][1]) <= 1:
            diff_stair = False
        elif (
            begin_length > 3
            and end_length > 3
            and begin_length + end_length > middle_length
        ):
            diff_stair = True
            trans_start = check_path[height_list[1][0] - 1]
            trans_end = check_path[height_list[-1][0]]
            _, success = get_shortest_path(env, trans_start, trans_end)
            assert success, "No path from trans start to end."
            trans = 0
        else:
            raise (Exception("Episode cross floors and may have bugs"))
    else:
        raise (Exception("Only instance task is supported!"))
    move_best = False

    agent.reset(
        env.sim.get_agent_state().position,
        env.sim.get_agent_state().rotation,
        env.current_episode,
    )
    agent.transformation_matrix = env.get_tf_episodic_to_global()
    agent._obstacle_map._min_height = 0.61
    agent._obstacle_map._max_height = 0.88
    agent._last_value = float("-inf")
    agent._last_frontier = np.zeros(2)
    agent._last_goal = np.zeros(2)

    # reset camera parameters
    sim_sensors_cfg = env.config.simulator.agents.main_agent.sim_sensors
    min_depth = sim_sensors_cfg.depth_sensor.min_depth
    max_depth = sim_sensors_cfg.depth_sensor.max_depth
    camera_fov = np.deg2rad(sim_sensors_cfg.depth_sensor.hfov)
    fx = fy = sim_sensors_cfg.depth_sensor.width / (2 * np.tan(camera_fov / 2))

    # reset episode
    step = 0
    angle_deviation = np.inf
    target_detected = -1
    video_frames = []
    # init save objects
    action = None
    metrics = {"success": 0}
    idx = params["idx"]
    scene_id = params["scene_id"]
    ep_id = params["ep_id"]
    params["initial_height"] = env.sim.get_agent_state().position[1]
    instructions = (
        episode.object_category
        if params["task"] == "object"
        else episode.instruction.instruction_text
    )
    prepare_dirs(params)
    ask_num, history, nav_camera = 0, [], []
    path_node_to_frame = []
    info, info_down = {}, {}
    (
        rgb_list,
        depth_list,
        rgb_down_list,
        depth_down_list,
        pose_list,
        pose_down_list,
        actions,
        shortest_path,
        ref_path,
    ) = ({}, {}, {}, {}, [], [], [int(-1)], [], [])

    os.makedirs("./check_sim", exist_ok=True)
    Image.fromarray(observations.rgb).save(
        os.path.join(".", "check_sim", f"rgb_{idx}.jpg")
    )
    agent_end = ShortestPathFollower(env.sim, params["threshold"], False)

    image_height, image_width = observations.semantic.shape[1:]
    h_min, h_max, w_min, w_max = (
        image_height // 2 - image_height // 4,
        image_height // 2 + image_height // 4,
        image_width // 2 - image_width // 4,
        image_width // 2 + image_width // 4,
    )
    ask_num, history, nav_camera = make_dialog(
        env,
        "information",
        ask_num,
        step,
        path_node_to_frame,
        history,
        nav_camera,
        object_dict=object_dict,
    )

    while not env.episode_over and step <= 500:
        if action != 0:
            (
                rgb,
                depth,
                rgb_path,
                depth_path,
                pose_episodic,
                point,
                rgb_key,
                camera_position,
                tf_camera_to_episodic,
                tf_camera_to_episodic_agent,
            ) = get_save_observations(env, observations, step, params)
            rgb_list[rgb_path] = Image.fromarray(rgb).convert("RGB")
            depth_list[depth_path] = Image.fromarray(
                depth.astype(np.uint16), mode="I;16"
            )
            pose_list.append(pose_episodic)
            shortest_path.append(np.array(point, dtype=float).tolist())
            info[rgb_key] = {
                "pose": pose_episodic.tolist(),
                "depth": f"{rgb_key}".replace("rgb", "depth").replace(".jpg", ".png"),
            }

            if step == 11:
                agent.update_voronois = agent.update_frontiers = True

            agent._cache_observations(
                rgb=observations.rgb,  # (H, W, 3)
                nav_depth=torch.from_numpy(observations.depth)
                .unsqueeze(0)
                .cuda(),  # (1, H, W) for pointnav
                robot_xy=camera_position[:2],
                robot_z=camera_position[2],
                robot_heading=point[-1],
                obstacle_map_rgbd=[
                    (
                        filter_depth(
                            (observations.depth).reshape(observations.depth.shape[:2]),
                            blur_type=None,
                        ),
                        tf_camera_to_episodic_agent,
                        min_depth,
                        max_depth,
                        fx,
                        fy,
                        camera_fov,
                    )
                ],
            )
            # look down
            observations, _, metrics = env.apply_action(5)
            (
                rgb,
                depth,
                rgb_path,
                depth_path,
                pose_episodic,
                point,
                rgb_key,
                camera_position,
                tf_camera_to_episodic,
                tf_camera_to_episodic_agent,
            ) = get_save_observations(env, observations, step, params, is_down=True)
            rgb_down_list[rgb_path] = Image.fromarray(rgb).convert("RGB")
            depth_down_list[depth_path] = Image.fromarray(
                depth.astype(np.uint16), mode="I;16"
            )
            pose_down_list.append(pose_episodic)
            info_down[rgb_key] = {
                "pose": pose_episodic.tolist(),
                "depth": f"{rgb_key}".replace("rgb", "depth").replace(".jpg", ".png"),
            }

            # look up
            observations, _, metrics = env.apply_action(4)
            if (
                diff_stair
                and np.linalg.norm(trans_start - env.sim.get_agent_state().position) < 1
                and trans != 1
            ):
                assert trans == 0, f"something wrong! trans={trans}"
                trans += 1
                agent.move_best
                ask_num, history, nav_camera = make_dialog(
                    env,
                    "stairs",
                    ask_num,
                    step,
                    path_node_to_frame,
                    history,
                    nav_camera,
                    height_list=[trans_start[1], trans_end[1]],
                )

        if target_detected < 0 and (not diff_stair or trans != 1):
            action = agent.act(env.sim)
            if observations.semantic[1:, h_min:h_max, w_min:w_max].sum() > 0:
                candidate_detected = (
                    observations.semantic[1:, h_min:h_max, w_min:w_max]
                    .sum(axis=(1, 2))
                    .argmax()
                    + 1
                )
                distance = np.linalg.norm(
                    env.sim.get_agent_state().position
                    - env.current_episode.goals[candidate_detected].position
                )
                if distance < 3:
                    ask_num, history, nav_camera = make_dialog(
                        env,
                        "disambiguation",
                        ask_num,
                        step,
                        path_node_to_frame,
                        history,
                        nav_camera,
                        is_goal=False,
                    )
                    remove_index = (
                        np.where(observations.semantic[1:].sum(axis=(1, 2)) > 0)[0] + 1
                    ).tolist()
                    episode.goals = [
                        g for i, g in enumerate(episode.goals) if i not in remove_index
                    ]
                    episode.instruction.instance_id = [
                        ins
                        for i, ins in enumerate(episode.instruction.instance_id)
                        if i not in remove_index
                    ]
            if observations.semantic[0].sum() > 0 and (not diff_stair or trans == 2):
                target_detected = 0
                goal_path, success = get_navigable_path(
                    env,
                    env.sim.get_agent_state().position,
                    [
                        {"agent_state": {"position": vp.agent_state.position}}
                        for vp in env.current_episode.goals[target_detected].view_points
                    ],
                    {"position": env.current_episode.goals[target_detected].position},
                )
                agent.move_best = False
                if not success:
                    raise Exception("Unreachable goal!")
                if (
                    calculate_path_length_noheight(goal_path)[-1] < 6
                    and observations.semantic[0, h_min:h_max, w_min:w_max].sum() > 0
                ):
                    ask_num, history, nav_camera = make_dialog(
                        env,
                        "disambiguation",
                        ask_num,
                        step,
                        path_node_to_frame,
                        history,
                        nav_camera,
                        is_goal=True,
                    )
        elif diff_stair and trans == 1:
            assert target_detected < 0, "something wrong! target_detected>=0"
            action = agent_end.get_next_action(trans_end)

            if action == 0:
                trans += 1
                agent._called_stop = True
                agent._obstacle_map.reset()
                agent._last_frontier = np.zeros(2)
                agent._last_goal = np.zeros(2)
                agent._done_initializing = False
                agent._obstacle_map._min_height += (
                    env.sim.get_agent_state().position[1] - params["initial_height"]
                )
                agent._obstacle_map._max_height += (
                    env.sim.get_agent_state().position[1] - params["initial_height"]
                )
                continue
        else:
            action = agent_end.get_next_action(goal_path[-1])
            if (
                action == 0
                and np.linalg.norm(
                    (env.sim.get_agent_state().position - goal_path[-1])[[0, 2]]
                )
                > params["threshold"]
            ):
                raise Exception(
                    "Shortest path follower failed to navigate to the goal!"
                )

            if action == 0 and abs(angle_deviation) > np.pi / 6:
                current_pos = env.sim.get_agent_state().position
                current_yaw = 2 * np.arctan2(
                    env.sim.get_agent_state().rotation.y,
                    env.sim.get_agent_state().rotation.w,
                )
                dx, _, dz = (
                    env.current_episode.goals[target_detected].position - current_pos
                )
                yaw = np.arctan2(-dz, dx) - np.pi / 2
                angle_deviation = yaw - current_yaw
                angle_deviation = np.arctan2(
                    np.sin(angle_deviation), np.cos(angle_deviation)
                )
                action = 2 if angle_deviation > 0 else 3

        if not move_best and agent.move_best:
            if f"{step:03d}" not in path_node_to_frame:
                path_node_to_frame.append(f"{step:03d}")
            begin_idx = step
            question_yaw = 2 * np.arctan2(
                env.sim.get_agent_state().rotation.y,
                env.sim.get_agent_state().rotation.w,
            )
            move_best = agent.move_best

        if move_best and not agent.move_best:
            move_best = agent.move_best
            begin_path, success_begin = get_navigable_path(
                env,
                np.array(ref_path[begin_idx]),
                [
                    {"agent_state": {"position": vp.agent_state.position}}
                    for vp in env.current_episode.goals[0].view_points
                ],
                {"position": env.current_episode.goals[0].position},
            )
            end_path, success_end = get_navigable_path(
                env,
                env.sim.get_agent_state().position,
                [
                    {"agent_state": {"position": vp.agent_state.position}}
                    for vp in env.current_episode.goals[0].view_points
                ],
                {"position": env.current_episode.goals[0].position},
            )
            if (
                success_begin
                and success_end
                and calculate_path_length_noheight(begin_path)[-1]
                > calculate_path_length_noheight(end_path)[-1]
            ):
                questioned_path = ref_path[begin_idx:]
                _, idx_sorted = np.unique(questioned_path, axis=0, return_index=True)
                idx_sorted = np.sort(idx_sorted)
                questioned_path = list(np.array(questioned_path)[idx_sorted])
                if calculate_path_length_noheight(questioned_path)[-1] > 1:
                    ask_num, history, nav_camera = make_dialog(
                        env,
                        "local",
                        ask_num,
                        begin_idx,
                        path_node_to_frame,
                        history,
                        nav_camera,
                        object_dict=object_dict,
                        region_dict=region_dict,
                        height_list=[env.sim.get_agent_state().position[1]]
                        * len(questioned_path),
                        questioned_path=questioned_path,
                        question_yaw=question_yaw,
                    )
        actions.append(int(action))
        ref_path.append([float(i) for i in env.sim.get_agent_state().position])
        observations, _, metrics = env.apply_action(action)
        if (
            np.linalg.norm((env.sim.get_agent_state().position - ref_path[-1])[[0, 2]])
            < 0.05
            and action == 1
        ):
            raise Exception("Get Stuck!")
        step += 1
        print(step, flush=True)

        if params["save_image"] and metrics["top_down_map"] is not None:
            frame = create_frame(
                agent, metrics, "explore" if target_detected < 0 else "navigate", step
            )
            video_frames.append(frame)
            plt.imsave(f"check_sim/frame_{idx}.jpg", video_frames[-1])
            plt.imsave(
                f"check_sim/semantic_{idx}.jpg", observations.semantic[target_detected]
            )
    instruct = {"id": ep_id}
    instruct["video"] = (
        os.path.basename(params["output_path"]) + f"/{scene_id}_{idx}_{ep_id:04d}"
    )
    prompt = "<video>\nYou are an autonomous navigation assistant. Your task is to find a specific <instruction> in current scene. Where should you go next to stay on track?"
    answer = "<trajectory>"
    conversation = [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": answer},
    ]
    instruct["conversations"] = conversation
    instruct["target"] = {
        "instructions": instructions,
        "shortest_path": shortest_path,
        "actions": actions[:-1],
        "r2r_path": ref_path,
        "ref_path": path_node_to_frame,
    }
    instruct["original_episode_idx"] = env.current_episode.episode_id
    instruct["target"]["dialog_history"] = history
    instruct["target"]["nav_camera"] = nav_camera
    assert ask_num == len(history) / 2, "ask num not match with history!"
    assert ask_num == len(nav_camera), "ask num not match with nav_camera!"
    for i in history:
        true_idx = i["true_idx"]
        assert (
            path_node_to_frame[i["nav_idx"]] == f"{true_idx:03d}"
        ), "Invalid dialog history!"
    assert (
        (
            len(instruct["target"]["shortest_path"])
            == len(instruct["target"]["r2r_path"])
        )
        and (
            len(instruct["target"]["shortest_path"])
            == len(instruct["target"]["actions"])
        )
        and (len(instruct["target"]["shortest_path"]) == len(info))
        and (len(instruct["target"]["shortest_path"]) == len(info_down))
        and (len(instruct["target"]["shortest_path"]) == len(rgb_list))
        and (len(instruct["target"]["shortest_path"]) == len(rgb_down_list))
    ), "Invalid trajectory!"
    assert step < 500, "times out!"
    assert target_detected >= 0, "target is not detected!"
    print("-" * 100)
    print(f"save images to {idx}")
    print("-" * 100)
    frame = create_frame(
        agent, metrics, "explore" if target_detected < 0 else "navigate", step
    )
    plt.imsave(
        os.path.join(
            params["output_path"],
            f'{scene_id}_{params["idx"]}_{ep_id:04d}',
            "frame.jpg",
        ),
        frame,
    )
    for path, rgb in rgb_list.items():
        rgb.save(path)
    for path, depth in depth_list.items():
        depth.save(path)
    for path, rgb in rgb_down_list.items():
        rgb.save(path)
    for path, depth in depth_down_list.items():
        depth.save(path)
    return info, instruct, info_down, step


def get_worker_info():
    idx = int(os.environ.get("SLURM_PROCID", 0))
    num_workers = int(os.environ.get("SLURM_NTASKS", 1))
    return idx, num_workers


if __name__ == "__main__":
    global local_rank
    # habitat params
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes_file_path",
        type=str,
        default="data/generated_dialog/instance_goal/instance_episodes_meta_dialog_train_mini.json.gz",
        help="Path to episodes file",
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="habitat/vlfm/config/tasks/dialog_mp3d.yaml",
        help="Path to habitat config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="habitat/vlfm/config/expertiments/gen_videos.yaml",
    )
    parser.add_argument("--split", type=str, default="train", help="Split")
    parser.add_argument("--scene_ids", type=str, default="ur6pFq6Qu1A", help="Scene id")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    # dialog params
    parser.add_argument("--task", type=str, default="object", help="Task")
    parser.add_argument("--vocabulary", type=str, default="hm3d", help="Vocabulary")
    parser.add_argument(
        "--pointnav_policy_path",
        type=str,
        default="data/debug_data/pointnav_weights.pth",
        help="Path to pointnav policy",
    )
    parser.add_argument(
        "--scene_summary_path",
        type=str,
        default="data/scene_summary",
        help="Path to object dict and region dict",
    )
    parser.add_argument(
        "--normal_category_path",
        type=str,
        default="habitat/generate_exploration/normal_category.json",
        help="Path to normal category",
    )

    # collect data params
    parser.add_argument("--target_detected_threshold", type=int, default=10)
    parser.add_argument("--shortest_path_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="data/exploration/qwen")
    parser.add_argument("--save_image", action="store_true", help="Save image")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--rank", default=0, type=int, help="rank")
    parser.add_argument("--gpu", default=0, type=int, help="gpu")
    parser.add_argument("--port", default="2333")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    print("-" * 100)
    print("Arguments:")
    args = parser.parse_args()
    init_distributed_mode(args)

    with open(args.normal_category_path, "r") as f:
        normal_category = json.load(f)
    # load environment
    config = get_config(args.habitat_config_path, args.baseline_config_path, args.opts)
    with read_write(config):
        config.exp.task = args.task
        config.exp.vocabulary = args.vocabulary
        if config.exp.task == "instance":
            if "all" == args.scene_ids:
                config.habitat.dataset.data_path = args.episodes_file_path
            else:
                config.habitat.dataset.data_path = os.path.join(
                    os.path.dirname(args.episodes_file_path),
                    args.split,
                    args.scene_ids,
                    os.path.basename(args.episodes_file_path),
                )
        config.habitat.dataset.split = args.split
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    print(f"conifg: {OmegaConf.to_yaml(config)}")
    env = ObjectNavEnv(config=config.habitat, exp_config=config.exp)
    # init agent
    agent = LlavaAgent(config.agent, args.shortest_path_threshold)
    # initial_save_objects
    intrinsic_matrix = get_intrinsic_matrix(
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    )
    axis_align_matrix = get_axis_align_matrix()
    sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
    camera_height = sim_sensors_cfg.rgb_sensor.position[1]  # 高度
    min_depth = sim_sensors_cfg.depth_sensor.min_depth
    max_depth = sim_sensors_cfg.depth_sensor.max_depth

    # generate episodes dialog for each scene
    scene_episode_dict = defaultdict(list)
    for episode in env.episodes:
        scene_episode_dict[episode.scene_id.split("/")[-2]].append(episode)

    for scene, episodes in scene_episode_dict.items():
        idx, num_workers = get_worker_info()
        os.makedirs(os.path.join(args.output_dir, scene), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir + "_30down"), exist_ok=True)
        episode_state = {}
        for file in [
            i
            for i in os.listdir(os.path.join(args.output_dir, scene))
            if i.endswith(".json")
        ]:
            with open(os.path.join(args.output_dir, scene, file), "r") as f:
                for line in f.readlines():
                    r = json.loads(line.strip())
                    episode_state[r["episode_name"]] = r["episode_content"]

        # load some files
        with open(
            os.path.join(args.scene_summary_path, f"{scene}/object_dict.json"), "r"
        ) as f:
            object_dict = json.load(f)
        with open(
            os.path.join(args.scene_summary_path, f"{scene}/region_dict.json"), "r"
        ) as f:
            region_dict = json.load(f)
        # initial_save_objects
        if os.path.exists(
            os.path.join(args.output_dir, f"instance_goal_{args.split}_{idx}.json")
        ):
            with open(
                os.path.join(args.output_dir, f"instance_goal_{args.split}_{idx}.json"),
                "r",
            ) as json_file:
                anno = json.load(json_file)
            with open(
                os.path.join(args.output_dir, f"infos_{args.split}_{idx}.json"), "r"
            ) as json_file:
                infos = json.load(json_file)
            with open(
                os.path.join(
                    args.output_dir + "_30down", f"infos_30down_{args.split}_{idx}.json"
                ),
                "r",
            ) as json_file:
                infos_down = json.load(json_file)
        else:
            infos, anno, infos_down = {}, [], {}
        print(f"totally {len(episodes)} in {scene} to be generated")
        print(f"for {idx} {len(episodes[idx::num_workers])} to be generated")
        ep_id = 0
        for episode in tqdm(episodes[idx::num_workers]):
            if f"{scene}_{episode.episode_id}" in episode_state:
                ep_id += 1
                continue
            error_message = "DONE"
            start_time = time.time()
            params = {
                "task": config.exp.task,
                "idx": idx,
                "scene_id": scene,
                "ep_id": ep_id,
                "split": args.split,
                "output_path": args.output_dir,
                "max_depth": max_depth,
                "min_depth": min_depth,
                "camera_height": camera_height,
                "threshold": 0.25,
                "save_image": args.save_image,
                "intrinsic_matrix": intrinsic_matrix,
                "target_detected_threshold": args.target_detected_threshold,
                "shortest_path_threshold": args.shortest_path_threshold,
            }
            try:
                info, instruct, info_down, step = qwen_move_to_target(
                    env,
                    episode,
                    agent,
                    params,
                    object_dict,
                    region_dict,
                    normal_category,
                )
                infos[f"instance_goal_{args.split}/{scene}_{idx}_{ep_id:04d}"] = info
                infos_down[
                    f"instance_goal_{args.split}/{scene}_{idx}_{ep_id:04d}"
                ] = info_down
                anno.append(instruct)
            except Exception as e:
                print(e)
                error_message = str(e)
                step = -1
            ep_id += 1
            end_time = time.time()
            print("-" * 100)
            print(f"time cost: {end_time - start_time} seconds")
            print("-" * 100)
            infos["intrinsic"] = intrinsic_matrix.tolist()
            infos["axis_align_matrix"] = axis_align_matrix.tolist()
            infos_down["intrinsic"] = intrinsic_matrix.tolist()
            infos_down["axis_align_matrix"] = axis_align_matrix.tolist()
            with open(
                os.path.join(
                    args.output_dir, f"instance_goal_{args.split}_{idx}_{scene}.json"
                ),
                "w",
            ) as json_file:
                json_file.write(json.dumps(anno, indent=4))
            with open(
                os.path.join(args.output_dir, f"infos_{args.split}_{idx}_{scene}.json"),
                "w",
            ) as json_file:
                json_file.write(json.dumps(infos, indent=4))
            with open(
                os.path.join(
                    args.output_dir + "_30down",
                    f"infos_30down_{args.split}_{idx}_{scene}.json",
                ),
                "w",
            ) as json_file:
                json_file.write(json.dumps(infos_down, indent=4))
            with open(
                os.path.join(
                    args.output_dir, scene, f"{args.split}_{idx}_episode_state.json"
                ),
                "a",
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "episode_name": f"{scene}_{episode.episode_id}",
                            "episode_content": {
                                "error": error_message,
                                "instruction": episode.object_category
                                if config.exp.task == "object"
                                else episode.instruction.instruction_text,
                                "step": step,
                                "name": f"{idx}_{scene}_{ep_id-1}",
                                "time_cost": end_time - start_time,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    env.close()
