import collections
import inspect
import json
import os
from heapq import heappop, heappush
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import quaternion
from habitat.utils.visualizations import maps
from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.ndimage import distance_transform_edt

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def filter_depth_uint8(
    depth_img: np.ndarray,
    max_depth: float,
    clip_far_thresh: Optional[float] = None,
    set_black_value: Optional[float] = None,
    **kwargs,
):
    """Filter a uint8 depth image by converting it to metric depth, applying depth completion/filtering, then 
    converting it back to uint8.

    Args:
        depth_img (np.ndarray): Depth image of shape (H, W) with integer dtype (typically uint8).
        max_depth (float): Maximum metric depth corresponding to value 255 in the uint8 image.
        clip_far_thresh (Optional[float]): If provided, clip depth values above this metric threshold.
        set_black_value (Optional[float]): If provided, replace missing/black pixels (depth == 0) with this metric 
            value.

    Returns:
        np.ndarray: A filtered depth image of shape (H, W) with dtype uint8.
    """
    assert np.issubdtype(depth_img.dtype, np.integer), "depth_img must be np.uint8"
    assert depth_img.ndim == 2, "depth_img must be 2D"
    depth_img = depth_img.astype(np.float32) / 255.0 * max_depth
    filtered_depth_img = filter_depth(
        depth_img, clip_far_thresh, set_black_value, **kwargs
    )
    return (filtered_depth_img / max_depth * 255).astype(np.uint8)


def filter_depth(
    depth_img: np.ndarray,
    clip_far_thresh: Optional[float] = None,
    set_black_value: Optional[float] = None,
    use_multiscale: bool = True,
    recover_nonzero: bool = True,
    **kwargs,
):
    """Filter a float32 depth image using depth completion (fast or multi-scale) and optional post-processing.

    Args:
        depth_img (np.ndarray): Depth image of shape (H, W) with floating dtype (typically float32), in metric units.
        clip_far_thresh (Optional[float]): If provided, clip depth values above this metric threshold.
        set_black_value (Optional[float]): If provided, replace missing/black pixels (depth == 0) with this metric 
            value.
        use_multiscale (bool): If True, use multi-scale dilation based filling; otherwise use the fast variant.
        recover_nonzero (bool): If True, preserve pixels that were non-zero in the input from being overwritten to 
            zero.

    Returns:
        np.ndarray: A filtered depth image of shape (H, W) in metric units (float32-like).
    """
    assert np.issubdtype(depth_img.dtype, np.floating), "depth_img must be np.float32"
    assert depth_img.ndim == 2, "depth_img must be 2D"
    if recover_nonzero:
        nonzero_mask = depth_img != 0
    else:
        nonzero_mask = None
    if use_multiscale:
        filtered_depth_img = fill_in_multiscale(depth_img, **kwargs)[0]
    else:
        filtered_depth_img = fill_in_fast(depth_img, **kwargs)
    if nonzero_mask is not None:
        # Recover pixels that weren't black before but were turned black by filtering
        filtered_depth_img[nonzero_mask] = depth_img[nonzero_mask]
    if clip_far_thresh is not None:
        # Whiten pixels above whiten_far_thresh
        filtered_depth_img = np.clip(filtered_depth_img, 0, clip_far_thresh)
    if set_black_value is not None:
        filtered_depth_img[filtered_depth_img == 0] = set_black_value
    return filtered_depth_img


def fill_in_fast(
    depth_map,
    max_depth=100.0,
    custom_kernel=DIAMOND_KERNEL_5,
    extrapolate=False,
    blur_type="bilateral",
):
    """Perform fast in-place depth completion via inversion, dilation/closing, hole filling, and smoothing.

    Args:
        depth_map (np.ndarray): Input depth map (H, W) in metric units; will be modified and returned.
        max_depth (float): Maximum depth used for inversion (depth <- max_depth - depth).
        custom_kernel (np.ndarray): Morphological kernel used for the initial dilation step.
        extrapolate (bool): If True, extend the highest valid pixel in each column to the top and apply large-kernel 
            fill.
        blur_type (Literal["bilateral", "gaussian"]): Smoothing method, either "bilateral" (structure-preserving) or 
            "gaussian" (lower RMSE).

    Returns:
        np.ndarray: Dense depth map of shape (H, W) in meters (same array object as input).
    """

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[
                0 : top_row_pixels[pixel_col_idx], pixel_col_idx
            ] = top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == "bilateral":
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == "gaussian":
        # Gaussian blur
        valid_pixels = depth_map > 0.1
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale(
    depth_map,
    max_depth=100.0,
    dilation_kernel_far=CROSS_KERNEL_3,
    dilation_kernel_med=CROSS_KERNEL_5,
    dilation_kernel_near=CROSS_KERNEL_7,
    extrapolate=False,
    blur_type="bilateral",
    show_process=False,
):
    """Slower, multi-scale dilation version with additional noise removal that provides better qualitative results.

    Args:
        depth_map (Union[np.ndarray, "ArrayLike"]): projected depths
        max_depth (float): max depth value for inversion
        dilation_kernel_far (np.ndarray): dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med (np.ndarray): dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near (np.ndarray): dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate (bool): whether to extrapolate by extending depths to top of the frame, and applying a 31x31 full 
            kernel dilation
        blur_type (Literal["gaussian", "bilateral"]):
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process (bool): saves process images into an OrderedDict

    Returns:
        np.ndarray: dense depth map.
        Optional[collections.OrderedDict]: OrderedDict of process images.
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = depths_in > 30.0

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = s1_inverted_depths > 0.1
    s1_inverted_depths[valid_pixels] = max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far), dilation_kernel_far
    )
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med), dilation_kernel_med
    )
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near), dilation_kernel_near
    )

    # Find valid pixels for each binned dilation
    valid_pixels_near = dilated_near > 0.1
    valid_pixels_med = dilated_med > 0.1
    valid_pixels_far = dilated_far > 0.1

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5
    )

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = s3_closed_depths > 0.1
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = s4_blurred_depths > 0.1
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[
        top_row_pixels, range(s5_dilated_depths.shape[1])
    ]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[
                0 : top_row_pixels[pixel_col_idx], pixel_col_idx
            ] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0 : top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == "gaussian":
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == "bilateral":
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict["s0_depths_in"] = depths_in

        process_dict["s1_inverted_depths"] = s1_inverted_depths
        process_dict["s2_dilated_depths"] = s2_dilated_depths
        process_dict["s3_closed_depths"] = s3_closed_depths
        process_dict["s4_blurred_depths"] = s4_blurred_depths
        process_dict["s5_combined_depths"] = s5_dilated_depths
        process_dict["s6_extended_depths"] = s6_extended_depths
        process_dict["s7_blurred_depths"] = s7_blurred_depths
        process_dict["s8_inverted_depths"] = s8_inverted_depths

        process_dict["s9_depths_out"] = depths_out

    return depths_out, process_dict


# for main function
def read_global_path_file(file_path):
    result = {}
    man_made = False
    with open(file_path, "r") as f:
        for line in f:
            if "human" in line:
                man_made = True
            if "scene" not in line:  # 跳过不含scene的行
                continue
            data = json.loads(line)
            scene = data["scene"]
            height = data["height"]
            filtered_path = data["filtered_path"]
            if filtered_path:  # 只处理不为空的
                filtered_path_new = [[y, x] for x, y in filtered_path]
                if scene not in result:
                    result[scene] = {}
                if height not in result[scene]:
                    result[scene][height] = []
                if man_made:
                    reordered = [pt for pt in filtered_path_new for _ in range(3)]
                    result[scene][height].append(reordered)
                else:
                    result[scene][height].append(filtered_path_new)
    return result


def get_config(
    habitat_config_path: str,
    baseline_config_path: str,
    opts: Optional[list] = None,
    configs_dir: str = os.path.dirname(inspect.getabsfile(inspect.currentframe())),
) -> DictConfig:
    """Load and merge Habitat config and baseline config into a single OmegaConf DictConfig.

    Args:
        habitat_config_path (str): Path to the Habitat YAML config.
        baseline_config_path (str): Path to the baseline YAML config.
        opts (Optional[List[str]]): Optional list of override strings applied when loading the Habitat config.
        configs_dir (str): Root directory for config resolution (defaults to this module's directory).

    Returns:
        DictConfig: A merged DictConfig containing both Habitat and baseline settings.
    """
    habitat_config = get_habitat_config(
        habitat_config_path, overrides=opts, configs_dir=configs_dir
    )
    baseline_config = OmegaConf.load(baseline_config_path)

    with open_dict(habitat_config):
        config = OmegaConf.merge(habitat_config, baseline_config)

    return config


def sparse_path_by_distance(positions, indices_to_keep, distance_threshold=1):
    """Downsample a path by keeping points that are sufficiently far apart, while forcing specified indices to remain.

    Args:
        positions (Sequence[Sequence[float]]): List of 3D positions (each position is indexable like [x, y, z]).
        indices_to_keep (Sequence[int]): Indices in the original positions list that must be preserved if present.
        distance_threshold (float): Minimum Euclidean distance required to keep a new point (unless forced).

    Returns:
        List[List[float]]: Downsampled list of 3D positions as float values.
        Dict[int, int]: Mapping from original indices (only those in indices_to_keep) to new indices.
    """
    total_len = len(positions)

    if len(positions) == 0:
        return [], {}

    must_keep_indices = set(indices_to_keep)
    final_indices = []
    last_kept_idx = 0
    final_indices.append(0)

    for i in range(1, total_len):
        dist = np.linalg.norm(
            np.array(positions[last_kept_idx]) - np.array(positions[i])
        )
        if dist >= distance_threshold or i in must_keep_indices:
            final_indices.append(i)
            last_kept_idx = i
    final_indices.append(i)
    last_kept_idx = i

    final_indices = sorted(set(final_indices))

    new_index_map = {
        idx: new_idx
        for new_idx, idx in enumerate(final_indices)
        if idx in indices_to_keep
    }

    new_positions = [
        [float(positions[i][0]), float(positions[i][1]), float(positions[i][2])]
        for i in final_indices
    ]
    return new_positions, new_index_map


def make_json_savable(data):
    for k, v in data.items():
        if isinstance(v, dict):
            make_json_savable(v)
        elif isinstance(v, list):
            for i in v:
                if isinstance(i, dict):
                    make_json_savable(i)
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, quaternion.quaternion):
            data[k] = [v.w, v.x, v.y, v.z]
    return data


# for generate trajectories
def get_intrinsic_matrix(sensor_cfg):
    width = sensor_cfg.width
    height = sensor_cfg.height
    fov = sensor_cfg.hfov
    fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    fy = fx  # Assuming square pixels (fx = fy)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    intrinsic_matrix = np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return intrinsic_matrix


def get_axis_align_matrix():
    ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    return ma


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Converts a given position and yaw angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def xyz_pitch_to_tf_matrix(xyz: np.ndarray, pitch: float) -> np.ndarray:
    """Converts a given position and pitch angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        pitch (float): The pitch angle in radians for y axis.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """

    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch), x],
            [0, 1, 0, y],
            [-np.sin(pitch), 0, np.cos(pitch), z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def xyz_yaw_pitch_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Converts a given position and yaw, pitch angles to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians.
        pitch (float): The pitch angle in radians for y axis.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    rot1 = xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
    rot2 = xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot1 @ rot2
    transformation_matrix[:3, 3] = xyz
    return transformation_matrix


def prepare_dirs(params):
    ep_id = params["ep_id"]
    scene_id = params["scene_id"]
    for subdir in ["rgb_images", "depth_images"]:
        os.makedirs(
            os.path.join(
                params["output_path"], f'{scene_id}_{params["idx"]}_{ep_id:04d}', subdir
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                params["output_path"] + "_30down",
                f'{scene_id}_{params["idx"]}_{ep_id:04d}',
                subdir,
            ),
            exist_ok=True,
        )

