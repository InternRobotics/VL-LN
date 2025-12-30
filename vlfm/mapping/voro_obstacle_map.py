# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union

import cv2
import numpy as np
import skimage
import sknw
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from skimage.morphology import skeletonize

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.graph_utils import *
from vlfm.utils.img_utils import fill_small_holes


# new
class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([])
    frontiers: np.ndarray = np.array([])
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def reset(self) -> None:
        super().reset()
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self._voronois_px = np.array([])
        self.voronois = np.array([])

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
        update_frontiers: bool = True,
        update_voronois: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            point_cloud_episodic_frame = transform_points(
                tf_camera_to_episodic, point_cloud_camera_frame
            )
            obstacle_cloud = filter_points_by_height(
                point_cloud_episodic_frame, self._min_height, self._max_height
            )

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)

            size = self._map.shape[0]
            valid_indices = (
                (pixel_points[:, 0] >= 0)
                & (pixel_points[:, 0] < size)
                & (pixel_points[:, 1] >= 0)
                & (pixel_points[:, 1] < size)
            )
            pixel_points = pixel_points[valid_indices]

            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(
            new_explored_area, np.ones((3, 3), np.uint8), iterations=1
        )
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(
                    cnt, tuple([int(i) for i in agent_pixel_location]), True
                )
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        # Compute frontier locations
        if update_frontiers:
            self._frontiers_px = self._get_frontiers()
            if len(self._frontiers_px) == 0:
                self.frontiers = np.array([])
            else:
                self.frontiers = self._px_to_xy(self._frontiers_px)

        # Compute voronoi locations
        if update_voronois:
            self._voronois_px, self.voro_graph = self._get_voronoi(agent_pixel_location)
            if len(self._voronois_px) == 0:
                self.voronois = np.array([])
            else:
                self.voronois = self._px_to_xy(self._voronois_px)

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

    def _get_voronoi(self, agent_pixel_location) -> np.ndarray:
        """Returns the voros of the map."""
        navigable_map = self._navigable_map.astype(np.uint8)  # 1 navigable
        explored_map = self.explored_area.astype(np.uint8)  # 1 explored
        explored_map = cv2.dilate(
            explored_map,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        explored_map[navigable_map == 0] = 0

        for _ in range(2):
            explored_map = skimage.morphology.binary_erosion(explored_map)
        for _ in range(2):
            explored_map = skimage.morphology.binary_dilation(explored_map).astype(
                float
            )

        # selem_idx = np.where(skimage.morphology.disk(6) > 0)
        # explored_map[selem_idx[0]-5+agent_pixel_location[1], selem_idx[1]-5+agent_pixel_location[0]] = True

        # Image.fromarray((explored_map*255).astype(np.uint8)).save(f'explored_map.png')

        skeleton = skeletonize(explored_map)

        # Image.fromarray((skeleton*255).astype(np.uint8)).save(f'skeleton.png')

        voro_graph = sknw.build_sknw(skeleton)
        graph = nx.Graph()
        for i in range(len(voro_graph.nodes)):
            graph.add_node(i, o=voro_graph.nodes[i]["o"])
        for s, e in voro_graph.edges():
            graph.add_edge(s, e, pts=voro_graph[s][e]["pts"])

        # draw_graph(explored_map, voro_graph, f'init_voroi_graph')
        # draw_topo_graph(explored_map, graph, f'1_init_graph')
        graph = prune_skeleton_graph(graph, degree=1)
        # draw_topo_graph(explored_map, graph, f'2_prune_graph')
        agent_node = np.array([agent_pixel_location[1], agent_pixel_location[0]])
        min_distance = float("inf")
        closest_node = None
        # for node in graph.nodes(data=True):
        #     node_coords = node[1]['o']
        #     distance = np.linalg.norm(node_coords - agent_node)
        #     if distance < min_distance:
        #         min_distance = distance
        #         closest_node = node[0]
        for (s, e) in graph.edges():
            ps = graph[s][e]["pts"]
            for p in ps:
                distance = np.linalg.norm(p - agent_node)
                if distance < min_distance:
                    min_distance = distance
                    closest_node = (s, e, p)

        graph.add_node("agent", o=closest_node[2])
        graph.add_edge(closest_node[0], "agent")
        graph.add_edge("agent", closest_node[1])
        graph = remove_degree_2_nodes(graph, explored_map, "agent")

        for _ in range(2):
            graph = remove_close_nodes(graph, "agent")

        # draw_topo_graph(explored_map, graph, f'3_remove_close_graph')
        graph = remove_degree_2_nodes(graph, explored_map, "agent")
        # draw_topo_graph(explored_map, graph, f'4_remove_degree_2_graph')

        # graph.nodes()[closest_node]['o'] = agent_node
        # draw_topo_graph(explored_map, graph, f'5_final_graph')

        voros = []
        voro_nodes = ["agent"]
        for neighbor in graph.neighbors("agent"):
            if (
                get_euclidean_dist(
                    graph.nodes[neighbor]["o"], graph.nodes["agent"]["o"]
                )
                < 10
            ):
                for neighbor2 in graph.neighbors(neighbor):
                    if (
                        get_euclidean_dist(
                            graph.nodes[neighbor2]["o"], graph.nodes["agent"]["o"]
                        )
                        < 10
                    ):
                        continue
                    voro_px = graph.nodes()[neighbor2]["o"][[1, 0]].astype(int)
                    voros.append(voro_px.tolist())
                    voro_nodes.append(neighbor2)
            else:
                voro_px = graph.nodes()[neighbor]["o"][[1, 0]].astype(int)
                voros.append(voro_px.tolist())
                voro_nodes.append(neighbor)

        add_frontiers = []
        for k, frontier_px in enumerate(self._frontiers_px):
            front_pos = frontier_px[[1, 0]]
            flag = True
            # for node in graph.nodes(data=True):
            for node in voro_nodes:
                node_pos = graph.nodes[node]["o"]
                if is_visible(explored_map, np.array(node_pos), np.array(front_pos)):
                    flag = False
                    break
            if flag:
                add = True
                for add_front_pos in add_frontiers:
                    if (
                        get_euclidean_dist(np.array(front_pos), np.array(add_front_pos))
                        < 10
                    ):
                        add = False
                for voro in voros:
                    if (
                        get_euclidean_dist(np.array(front_pos), np.array(voro)[[1, 0]])
                        < 10
                    ):
                        add = False
                if add:
                    add_frontiers.append(front_pos)

        for k, front_pos in enumerate(add_frontiers):
            graph.add_node(f"frontier_{k}", o=front_pos)

        for node in graph.nodes(data=True):
            if "frontier" in str(node[0]):
                voros.append(node[1]["o"][[1, 0]].astype(int))
        return np.array(voros), graph

    def visualize(self, candidates, select, goal) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        # for frontier in self._frontiers_px:
        #     cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        # print(f"num of voros = {len(self._voronois_px)}")
        for i, voro in enumerate(self._voronois_px):
            # if i == select:
            #     color = (0, 200, 0)
            # elif i in candidates:
            #     color = (0, 0, 200)
            # else:
            #     # continue
            #     color = (200, 0, 0)
            color = (0, 0, 200)
            cv2.circle(vis_img, tuple([int(i) for i in voro]), 5, color, 2)

        goal_px = self._xy_to_px(goal.reshape(1, 2))[0]
        # print(f"goalpx = {goal_px}")
        cv2.circle(vis_img, goal_px, 8, (0, 200, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )
        return vis_img

    # def visualize_goal(self, goal) -> np.ndarray:
    #     """Visualizes the map."""
    #     vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
    #     # Draw explored area in light green
    #     vis_img[self.explored_area == 1] = (200, 255, 200)
    #     # Draw unnavigable areas in gray
    #     vis_img[self._navigable_map == 0] = self.radius_padding_color
    #     # Draw obstacles in black
    #     vis_img[self._map == 1] = (0, 0, 0)
    #     # Draw frontiers in blue (200, 0, 0)
    #     # for frontier in self._frontiers_px:
    #     #     cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

    #     # print(f"num of voros = {len(self._voronois_px)}")
    #     for i, voro in enumerate(self._voronois_px):
    #         color = (0, 0, 200)
    #         cv2.circle(vis_img, tuple([int(i) for i in voro]), 5, color, 2)

    #     goal_px = self._xy_to_px(goal.reshape(1, 2))[0]
    #     # print(f"goalpx = {goal_px}")
    #     cv2.circle(vis_img, goal_px, 8, (0, 200, 0), 2)

    #     vis_img = cv2.flip(vis_img, 0)

    #     if len(self._camera_positions) > 0:
    #         self._traj_vis.draw_trajectory(
    #             vis_img,
    #             self._camera_positions,
    #             self._last_camera_yaw,
    #         )
    #     return vis_img


def filter_points_by_height(
    points: np.ndarray, min_height: float, max_height: float
) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]
