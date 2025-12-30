import argparse
from collections import defaultdict

import habitat_sim
import numpy as np
import quaternion

from dialog_generation.env.objectnav_env import ObjectNavEnv
from dialog_generation.get_description import (
    get_path_description,
    get_path_description_without_additional_info,
    get_room_name,
)
from dialog_generation.other_utils import sparse_path_by_distance
from dialog_generation.template import (
    INFORMATION_TEMPLATE,
    QUESTION_TEMPLATE,
    STAIRS_TEMPLATE,
)


# information
def sift_candidates(env: ObjectNavEnv, object_dict: dict, missed_attributes: dict):
    """Sift candidate object instances that match the current goal object's known attributes, and (if needed) choose 
    one missed attribute that would best reduce ambiguity among remaining candidates.

    Args:
        env (ObjectNavEnv): ObjectNavEnv.
        object_dict (Dict[Hashable, Dict[str, Any]]): Mapping from instance_id to object metadata.
        missed_attributes (MutableMapping[str, Any]): Mapping of attribute name.

    Returns:
        Set[Hashable]: A set of remaining candidate instance_ids after filtering by known attributes.
        Optional[str]: The selected attribute name that would remove the most candidates, or None.
        bool: Boolean indicating whether an information question should be asked.
    """
    instance_id = env.current_episode.instruction.instance_id[0]
    obj_info = object_dict[instance_id]
    obj_common = obj_info["common"]
    known_attributes = env.current_episode.instruction.instruction_info
    known_attributes = list(set(known_attributes) - set(["texture", "nearby objects"]))
    if "nearby objects" in missed_attributes:
        missed_attributes.pop("nearby objects")

    init_candidates = set(
        [
            obj
            for obj in object_dict
            if object_dict[obj]["category"] == obj_info["category"]
        ]
    )
    for a in known_attributes:
        if a == "room":
            new_candidates = set(
                [c for c in init_candidates if object_dict[c][a] == obj_info[a]]
            )
        else:
            new_candidates = set(
                [
                    c
                    for c in init_candidates
                    if "common" not in object_dict[c]
                    or object_dict[c]["common"][a] == obj_common[a]
                ]
            )
        init_candidates = init_candidates & new_candidates

    if len(init_candidates) == 1 or len(missed_attributes) == 0:
        return init_candidates, None, False

    sift_num = {}
    for a in missed_attributes:
        if a == "room":
            new_candidates = set(
                [c for c in init_candidates if object_dict[c][a] == obj_info[a]]
            )
        else:
            new_candidates = set(
                [
                    c
                    for c in init_candidates
                    if "common" not in object_dict[c]
                    or object_dict[c]["common"][a] == obj_common[a]
                ]
            )
        sift_num[a] = {
            "sifted_candidates": list(new_candidates),
            "sift_num": len(init_candidates) - len(new_candidates),
        }
    sift_num = sorted(sift_num.items(), key=lambda x: x[1]["sift_num"], reverse=True)
    return init_candidates, sift_num[0][0], True


def check_missed_attributes(env: ObjectNavEnv, object_dict: dict):
    object_info = object_dict[env.current_episode.instruction.instance_id[0]]
    assert isinstance(object_info["unique_description"], dict), "object_info must exist"
    attributes = env.current_episode.instruction.instruction_info
    all_attributes = {"room": get_room_name(object_info["room"])}
    all_attributes.update(
        {
            a.lower(): i.lower()
            for a, i in object_info["unique_description"].items()
            if a in ["color", "material", "shape", "placement"] and len(i) > 0
        }
    )
    nearby_objects = list(
        set(
            [
                object_dict[obj]["unique_description"]["fine grained category"].lower()
                for obj, _ in object_info["nearby_objects"].items()
                if isinstance(object_dict[obj]["unique_description"], dict)
            ]
        )
    )
    if len(nearby_objects) > 0:
        all_attributes["nearby objects"] = ", ".join(nearby_objects)
    missed_attributes = list(set(list(all_attributes.keys())) - set(attributes))
    return {ma: all_attributes[ma] for ma in missed_attributes}


def pixel_to_world(index, x_min, z_min, mpp):
    i, j = index[0], index[1]
    # get the pixel center (i+0.5, j+0.5)
    x = x_min + (j + 0.5) * mpp
    z = z_min + (i + 0.5) * mpp
    return (x, z)


def cut_local_question_path(
    questioned_path: list,
    semi_reference_path: list,
    max_length: int = 5,
    angle_gap: float = np.pi / 6,
):
    gap = int(2 * np.pi / angle_gap)
    path_length = calculate_path_length(questioned_path)
    if path_length[-1] > max_length:
        question_ask_indices = min(
            [i for i, c in enumerate(path_length) if c > max_length]
        )
    else:
        question_ask_indices = len(questioned_path) - 1
    if len(semi_reference_path) <= 1:
        yaw = 0
    else:
        dx, _, dz = np.array(semi_reference_path[-1]) - np.array(
            semi_reference_path[-2]
        )
        yaw = np.arctan2(-dz, dx) - np.pi / 2
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        yaw_list = np.linspace(-np.pi, np.pi, gap, endpoint=False)
        index = np.argmin(np.abs(yaw_list - yaw))
        yaw = yaw_list[index]
    return question_ask_indices, yaw


def world_to_pixel(x, z, x_min, z_min, mpp):
    j = int((x - x_min) / mpp)
    i = int((z - z_min) / mpp)
    return (i, j)


def find_nearest_ponit_index(position, exploration_path: list):
    return np.argmin(
        np.linalg.norm(np.array(position) - np.array(exploration_path), axis=1)
    )


def find_best_exploration_path(
    exploration_path, agent_position, agent_position_px, global_map
):
    """Find the index of the closest point in an exploration path to a given 3D position.

    Args:
        exploration_path (MutableMapping[float, List[Sequence[Tuple[int, int]]]]): Mapping from height -> list of 2D 
            grid paths. Will be mutated via pop().
        agent_position (Union[Sequence[float], np.ndarray]): Agent 3D position, uses agent_position[1] as height.
        agent_position_px (Union[Sequence[float], np.ndarray]): Agent 2D position in map pixel/grid coordinates.
        global_map (np.ndarray): 2D occupancy grid; indexed as global_map[x, y] with boolean semantics.

    Returns:
        bool: found flag.
        Optional[float]]: selected_height.
    """
    all_heights = sorted(
        exploration_path.keys(), key=lambda h: abs(h - agent_position[1])
    )
    all_h_result = []
    for h in all_heights:
        one_h_result = []
        for path in exploration_path[h]:
            total_nodes = len(path)
            occ_nodes = 0
            min_agent_dist = float("inf")
            for x, y in path:
                if global_map[x, y] is True:
                    occ_nodes += 1
                dist = np.linalg.norm(
                    np.array([x, y])
                    - np.array([agent_position_px[0], agent_position_px[1]])
                )
                if dist < min_agent_dist:
                    min_agent_dist = dist
            ratio = occ_nodes / total_nodes if total_nodes > 0 else 0
            one_h_result.append(
                {
                    "height": h,
                    "path": path,
                    "occupied_ratio": ratio,
                    "min_agent_dist": min_agent_dist,
                }
            )

        high_occ_paths = [p for p in one_h_result if p["occupied_ratio"] > 0.95]
        if high_occ_paths:
            # Select the path with the smallest min_agent_dist
            best_high_occ_path = min(high_occ_paths, key=lambda p: p["min_agent_dist"])
            exploration_path.pop(h)
            return True, best_high_occ_path["height"]

        # If no path has occ_ratio > 0.95, keep the one with the highest occupancy ratio at this height
        if one_h_result:
            best_one_h = max(
                one_h_result, key=lambda p: (p["occupied_ratio"], -p["min_agent_dist"])
            )
            all_h_result.append(best_one_h)

    if all_h_result:
        best_overall = max(
            all_h_result, key=lambda p: (p["occupied_ratio"], -p["min_agent_dist"])
        )
        if best_overall["occupied_ratio"] > 0.8:
            exploration_path.pop(best_overall["height"])
            return True, best_overall["height"]

    return False, None


def build_path(
    start_index: int, end_index: int, path: list, initial_start_index: int = None
):
    """Construct a contiguous segment on a cyclic path from start_index to end_index, optionally taking into account 
    an initial_start_index to decide direction and wrap-around behavior.

    Args:
        start_index (int): Start index on the path.
        end_index (int): End index on the path (inclusive).
        path (Sequence[T]): The cyclic path as a list of points.
        initial_start_index (Optional[int]): Optional initial anchor index used to resolve traversal direction.

    Returns:
        List[T]: A list of points representing the constructed traversal segment.
    """
    next_exploration_path = []
    if initial_start_index is None:
        if start_index > end_index:
            next_exploration_path += path[start_index:] + path[: end_index + 1]
        elif start_index <= end_index:
            next_exploration_path += path[start_index : end_index + 1]
        else:
            raise Exception("Something wrong in build_path!")
    else:
        if initial_start_index > end_index:
            if start_index > initial_start_index:
                next_exploration_path += path[start_index:] + path[: end_index + 1]
            elif start_index < end_index:
                next_exploration_path += path[start_index : end_index + 1]
            elif start_index <= initial_start_index and start_index >= end_index:
                next_exploration_path += list(
                    reversed(path[end_index : start_index + 1])
                )
            else:
                raise Exception("Something wrong in build_path!")
        elif initial_start_index < end_index:
            if start_index >= initial_start_index and start_index <= end_index:
                next_exploration_path += path[start_index : end_index + 1]
            elif start_index < initial_start_index:
                next_exploration_path += list(reversed(path[: start_index + 1])) + list(
                    reversed(path[end_index:])
                )
            elif start_index > end_index:
                next_exploration_path += list(
                    reversed(path[end_index : start_index + 1])
                )
            else:
                raise Exception("Something wrong in build_path!")
        else:
            raise Exception("Something wrong in out loop!")
    return next_exploration_path


def get_next_exploration_path(
    env: ObjectNavEnv,
    start_point: list,
    end_point: list,
    exploration_paths: dict,
    initial_path: dict,
):
    """Generate the next exploration trajectory in world coordinates by selecting suitable exploration paths.

    Args:
        env (ObjectNavEnv): ObjectNavEnv used to query scene bounds and top-down maps from the simulator pathfinder.
        start_point (Union[Sequence[float], np.ndarray]): Start 3D position (x, y, z).
        end_point (Union[Sequence[float], np.ndarray]): End 3D position (x, y, z).
        exploration_paths (Dict[float, List[Sequence[Tuple[int, int]]]]): Dict mapping height -> list of pixel paths, 
            where each pixel path is [(i, j), ...].
        initial_path (Optional[Dict[str, Any]]): State dict controlling deterministic reuse of previously chosen 
            exploration paths; may be None.

    Returns:
        List[List[float]]: List of 3D waypoints (x, y, z) for the next exploration segment.
        Dict[str, Any]: Updated state dict for subsequent calls.
    """
    next_exploration_path = []
    meters_per_pixel = 0.05
    min_bound, max_bound = env.sim.pathfinder.get_bounds()
    x_min, _, z_min = min_bound
    x_max, _, z_max = max_bound

    start_px = world_to_pixel(
        start_point[0], start_point[2], x_min, z_min, mpp=meters_per_pixel
    )
    end_px = world_to_pixel(
        end_point[0], end_point[2], x_min, z_min, mpp=meters_per_pixel
    )
    start_success, start_height = find_best_exploration_path(
        exploration_paths.copy(),
        start_point,
        start_px,
        env.sim.pathfinder.get_topdown_view(meters_per_pixel, start_point[1]),
    )  # (H, W)
    end_success, end_height = find_best_exploration_path(
        exploration_paths.copy(),
        end_point,
        end_px,
        env.sim.pathfinder.get_topdown_view(meters_per_pixel, end_point[1]),
    )  # (H, W)
    if not start_success:
        if end_success:
            start_height, _ = get_heights(
                list(exploration_paths.keys()),
                end_height=end_height,
                start_position=start_point[1],
            )
        else:
            start_height, end_height = get_heights(
                list(exploration_paths.keys()),
                start_position=start_point[1],
                end_position=end_point[1],
            )
    elif not end_success:
        _, end_height = get_heights(
            list(exploration_paths.keys()),
            start_height=start_height,
            end_position=end_point[1],
        )

    # transform px exploration path to world coordinates
    exploration_paths_px = defaultdict(list)
    for height, paths in exploration_paths.items():
        for path in paths:
            new_path = []
            for idx, point in enumerate(path):
                if (
                    idx > 0
                    and point[0] == path[idx - 1][0]
                    and point[1] == path[idx - 1][1]
                ):
                    continue
                x, z = pixel_to_world(point, x_min, z_min, mpp=meters_per_pixel)
                new_path.append([x, height, z])
            exploration_paths_px[height].append(new_path)

    if initial_path is None:
        start_exploration_path = exploration_paths_px[start_height][
            np.random.randint(0, len(exploration_paths_px[start_height]))
        ]
        start_point_index = find_nearest_ponit_index(
            start_point, start_exploration_path
        )
        if start_height == end_height:
            end_point_index = find_nearest_ponit_index(
                end_point, start_exploration_path
            )
            next_exploration_path += build_path(
                start_point_index, end_point_index, start_exploration_path
            )
            initial_path = {
                "start": {
                    "height": start_height,
                    "path": start_exploration_path,
                    "index": start_point_index,
                },
                "end": {
                    "height": end_height,
                    "path": start_exploration_path,
                    "index": end_point_index,
                },
            }
        else:
            end_exploration_path = exploration_paths_px[end_height][
                np.random.randint(0, len(exploration_paths_px[end_height]))
            ]
            end_point_index = find_nearest_ponit_index(end_point, end_exploration_path)
            next_exploration_path += (
                start_exploration_path[start_point_index:]
                + start_exploration_path[:start_point_index]
            )
            arbitary_index = np.random.randint(0, len(end_exploration_path))
            while arbitary_index == end_point_index:
                arbitary_index = np.random.randint(0, len(end_exploration_path))
            next_exploration_path += build_path(
                arbitary_index, end_point_index, end_exploration_path
            )
            initial_path = {
                "start": {
                    "height": start_height,
                    "path": start_exploration_path,
                    "index": start_point_index,
                },
                "end": {
                    "height": end_height,
                    "path": end_exploration_path,
                    "index": end_point_index,
                },
                "arbitary_index": arbitary_index,
            }
    else:
        start_exploration_path = initial_path["start"]["path"]
        end_exploration_path = initial_path["end"]["path"]
        end_point_index = initial_path["end"]["index"]
        if initial_path["end"]["height"] == initial_path["start"]["height"]:
            start_point_index = find_nearest_ponit_index(
                start_point, start_exploration_path
            )
            initial_start_index = initial_path["start"]["index"]
            next_exploration_path += build_path(
                start_point_index,
                end_point_index,
                start_exploration_path,
                initial_start_index=initial_start_index,
            )
        elif start_height == initial_path["start"]["height"]:
            start_point_index = find_nearest_ponit_index(
                start_point, start_exploration_path
            )
            initial_start_index = initial_path["arbitary_index"]
            next_exploration_path += [start_point] + build_path(
                initial_start_index, end_point_index, end_exploration_path
            )
        elif start_height == initial_path["end"]["height"]:
            start_point_index = find_nearest_ponit_index(
                start_point, end_exploration_path
            )
            initial_start_index = initial_path["arbitary_index"]
            next_exploration_path += build_path(
                start_point_index,
                end_point_index,
                end_exploration_path,
                initial_start_index=initial_start_index,
            )
        else:
            raise Exception("start height is not in initial path!")
    return next_exploration_path, initial_path


def get_heights(
    heights: list,
    start_height: float = None,
    end_height: float = None,
    start_position: float = None,
    end_position: float = None,
):
    """Determine appropriate discrete start/end heights from a list of available heights given either explicit heights 
    or continuous y-positions, ensuring selected heights are consistent with movement direction.

    Args:
        heights (Sequence[float]): List of discrete height values available in the environment.
        start_height (Optional[float]):  Optional explicit start height (discrete).
        end_height (Optional[float]): Optional explicit end height (discrete).
        start_position (Optional[float]): Optional continuous start y-position used to infer start_height.
        end_position (Optional[float]): Optional continuous end y-position used to infer end_height.

    Returns:
        float: start height.
        float: end_height.
    """
    assert (
        start_height is not None or start_position is not None
    ), "start_height or start_position must be provided!"
    assert (
        end_height is not None or end_position is not None
    ), "end_height or end_position must be provided!"

    if start_height is None and end_height is None:
        if start_position < end_position:
            candidate_heights = sorted(
                [h for h in heights if h >= start_position and h <= end_position]
            )
            if len(candidate_heights) == 0:
                return (
                    heights[np.argmin([abs(start_position - i) for i in heights])],
                    heights[np.argmin([abs(end_position - i) for i in heights])],
                )
            return candidate_heights[0], candidate_heights[-1]
        else:
            candidate_heights = sorted(
                [h for h in heights if h <= start_position and h >= end_position]
            )
            if len(candidate_heights) == 0:
                return (
                    heights[np.argmin([abs(start_position - i) for i in heights])],
                    heights[np.argmin([abs(end_position - i) for i in heights])],
                )
            return candidate_heights[-1], candidate_heights[0]

    if start_height is None:
        min_height = min(heights)
        max_height = max(heights)
        for h in heights:
            if h >= start_position and h < max_height:
                max_height = h
            if h <= start_position and h > min_height:
                min_height = h
        if end_height >= max_height:
            return max_height, end_height
        elif end_height <= min_height:
            return min_height, end_height
        else:
            raise Exception("min height and max height must near each other!")

    if end_height is None:
        min_height = min(heights)
        max_height = max(heights)
        for h in heights:
            if h >= end_position and h < max_height:
                max_height = h
            if h <= end_position and h > min_height:
                min_height = h
        if start_height >= max_height:
            return start_height, max_height
        elif start_height <= min_height:
            return start_height, min_height
        else:
            raise Exception("min height and max height must near each other!")
    return start_height, end_height


# shared
def calculate_path_length(path):
    accumulated_length = [0]
    for i, p in enumerate(path[1:]):
        accumulated_length.append(
            accumulated_length[i] + np.linalg.norm(np.array(p) - np.array(path[i]))
        )
    return accumulated_length


def get_navigable_path(
    env: ObjectNavEnv, start_position, target_positions: list, object_info: dict
):
    start_position = [float(i) for i in start_position]
    target_positions = sorted(
        target_positions,
        key=lambda x: np.linalg.norm(
            np.array(x["agent_state"]["position"]) - np.array(object_info["position"])
        ),
    )
    success = False
    while not success and len(target_positions) > 0:
        target_position = target_positions.pop(0)
        shortest_path, success = get_shortest_path(
            env, start_position, target_position["agent_state"]["position"]
        )
    if success:
        return shortest_path, True
    else:
        return [], False


def get_shortest_path(env, start_position, target_position):
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_position
    shortest_path.requested_end = target_position

    success = env.sim.pathfinder.find_path(shortest_path)
    return shortest_path.points, success


def generate_dialog(
    env: ObjectNavEnv,
    question_type: str,
    object_dict: dict = None,
    region_dict: dict = None,
    questioned_path: list = None,
    attribute_to_ask: dict = None,
    yaw: float = None,
    is_goal: bool = None,
    height_list: list = None,
):
    """Generate a (question, answer) pair for a specified dialog question type.

    Args:
        env (ObjectNavEnv): ObjectNavEnv.
        question_type (Literal["local","disambiguation","information","stairs"]): One of {"local", "disambiguation", 
            "information", "stairs"}.
        object_dict (Optional[Dict[Hashable, Dict[str, Any]]]): Optional object metadata dict required for some 
            question types.
        region_dict (Optional[Dict[str, Any]]): Optional region metadata dict used by path description generation.
        questioned_path (Optional[Sequence[Sequence[float]]]): Optional list of 3D points used to generate local path 
            descriptions.
        attribute_to_ask (Optional[Dict[str, Any]]): Optional dict with keys {"attribute", "content"} used for 
            information questions.
        yaw (Optional[float]): Optional yaw (radians) used as the reference orientation for local path description.
        is_goal (Optional[bool]): Optional boolean for disambiguation questions indicating whether the confirmed object
             is the goal.
        height_list (Optional[Sequence[float]]): Optional list of discrete heights aligned with questioned_path for 
            floor-change descriptions.

    Returns:
        str: question.
        str: answer.
    """
    if question_type == "local":
        # generate question
        question = np.random.choice(QUESTION_TEMPLATE["local_question"]["question"])
        # generate answer
        _, idx = np.unique(questioned_path, axis=0, return_index=True)
        idx_sorted = np.sort(idx)
        questioned_path = list(np.array(questioned_path)[idx_sorted])
        try:
            answer, _ = get_path_description(
                quaternion.from_euler_angles([0, yaw, 0]),
                questioned_path,
                object_dict,
                region_dict,
                return_finish=False,
                height_list=height_list,
            )
        except Exception as e:
            print(e)
            answer, _ = get_path_description_without_additional_info(
                quaternion.from_euler_angles([0, yaw, 0]),
                questioned_path,
                height_list=height_list,
            )
    elif question_type == "disambiguation":
        question = np.random.choice(
            QUESTION_TEMPLATE["object_confirm_question"]["question"]
        )
        if is_goal:
            answer = np.random.choice(
                QUESTION_TEMPLATE["object_confirm_question"]["answer"]["yes"]
            )
        else:
            answer = np.random.choice(
                QUESTION_TEMPLATE["object_confirm_question"]["answer"]["no"]
            )
    elif question_type == "information":
        instance_category = object_dict[env.current_episode.instruction.instance_id[0]][
            "unique_description"
        ]["fine grained category"].lower()
        attribute = attribute_to_ask["attribute"]
        content = attribute_to_ask["content"]
        if attribute == "room":
            content = get_room_name(content)
        question = np.random.choice(INFORMATION_TEMPLATE).format(
            ma=attribute, instance_category=instance_category
        )
        answer = np.random.choice(
            [f"The {attribute} of {instance_category} is: {content}", content]
        )
    elif question_type == "stairs":
        assert height_list[0] != height_list[1], "no need to change floor!"
        stair_type = "up" if height_list[-1] > height_list[0] else "down"
        pair = np.random.choice(STAIRS_TEMPLATE)
        question = np.random.choice(pair["question"])
        answer = np.random.choice(pair["answer"]).format(up_or_down=stair_type)
    else:
        raise Exception("Invalid question type!")

    return question, answer


def get_height_list(positions: list, heights: list):
    height_list = []
    for position in positions:
        height_list.append(heights[np.argmin([abs(position[1] - i) for i in heights])])
    return height_list
