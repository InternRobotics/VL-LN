import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import skimage
from habitat.utils.visualizations import maps
from PIL import Image
from skimage.draw import line


def get_euclidean_dist(node_coord1, node_coord2):
    node_coord1 = node_coord1.astype(np.int32)
    node_coord2 = node_coord2.astype(np.int32)
    return np.sqrt(((node_coord1 - node_coord2) ** 2).sum(0))


def compute_angle(node_coord1, node_coord2, node_coord3):
    node_coord1 = node_coord1.astype(np.int32)
    node_coord2 = node_coord2.astype(np.int32)
    node_coord3 = node_coord3.astype(np.int32)
    u = node_coord2 - node_coord1
    v = node_coord3 - node_coord1

    dot_product = np.dot(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    cos_theta = dot_product / (magnitude_u * magnitude_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return (
        angle_deg,
        get_euclidean_dist(node_coord1, node_coord2),
        get_euclidean_dist(node_coord1, node_coord3),
    )


def is_visible(occ_map, node_coord1, node_coord2, thres=60):
    # node_coord1 = graph.nodes[node1]['o']
    # node_coord2 = graph.nodes[node2]['o']
    node_coord1 = node_coord1.astype(np.int32)
    node_coord2 = node_coord2.astype(np.int32)
    rr_line, cc_line = line(
        node_coord1[0], node_coord1[1], node_coord2[0], node_coord2[1]
    )
    line_vals = occ_map[rr_line, cc_line]
    # image = np.zeros((occ_map.shape[0],occ_map.shape[1],3)).astype(np.uint8)
    # image[node_coord1[0]-3:node_coord1[0]+3,node_coord1[1]-3:node_coord1[1]+3,:] = np.array([255,0,0])
    # image[node_coord2[0]-3:node_coord2[0]+3,node_coord2[1]-3:node_coord2[1]+3,:] = np.array([255,0,0])
    # image[rr_line, cc_line, :] = np.array([255,255,0])
    # Image.fromarray(image).save('line.png')
    # print((np.all(line_vals), get_euclidean_dist(node_coord1, node_coord2) < thres))
    return np.all(line_vals) and get_euclidean_dist(node_coord1, node_coord2) < thres


def create_disk_mask(H, W, coord, radius=40):
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((X - coord[1]) ** 2 + (Y - coord[0]) ** 2)
    mask = dist_from_center <= radius
    # image = np.zeros((H,W,3)).astype(np.uint8)
    # image[mask,:] = np.array([255,255,0])
    # image[coord[0]-3:coord[0]+3,coord[1]-3:coord[1]+3,:] = np.array([255,0,0])
    # Image.fromarray(image).save('disk.png')
    return mask


def build_occupancy_map(env, saved_folder, height, floor_idx, cfg):
    def save_occ_map(img, name):
        """save the figure img at directory 'name' using matplotlib"""
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img, cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(name)
        plt.close()

    cell_size = cfg.cell_size
    # lower_bound, upper_bound = env.pathfinder.get_bounds()
    # coordinate_min_x, coordinate_max_x = lower_bound[2], upper_bound[2]
    # coordinate_min_z, coordinate_max_z = lower_bound[0], upper_bound[0]

    # x = np.arange(coordinate_min_x, coordinate_max_x, cell_size)
    # z = np.arange(coordinate_min_z, coordinate_max_z, cell_size)
    # xv, zv = np.meshgrid(x, z)
    # print(xv.shape, len(x), len(z))

    # map_resolution = min(zv.shape)
    # occ_map = maps.get_topdown_map(env.pathfinder, height, map_resolution, draw_border=False)
    occ_map = maps.get_topdown_map(
        env.pathfinder, height, draw_border=False, meters_per_pixel=cell_size
    )
    for _ in range(1):
        occ_map = skimage.morphology.binary_erosion(occ_map.astype(bool)).astype(float)
    for _ in range(1):
        occ_map = skimage.morphology.binary_dilation(occ_map.astype(bool)).astype(float)

    # grid_H, grid_W = occ_map.shape[0], occ_map.shape[1]
    # for grid_z in range(grid_H):
    #     for grid_x in range(grid_W):
    #         if occ_map[grid_z, grid_x] == 0:
    #             world_pos = maps.from_grid(grid_z, grid_x, occ_map.shape, env)
    #             if env.is_navigable(np.array([world_pos[1], height ,world_pos[0]])):
    #                 occ_map[grid_z, grid_x] = 1
    # save_occ_map(occ_map, f'{saved_folder}/occ_map_{floor_idx}.png')
    Image.fromarray(occ_map.astype(np.uint8) * 255).save(
        f"{saved_folder}/occ_map_{floor_idx}.png"
    )
    return occ_map


def prune_skeleton_graph(graph, degree=1):
    to_prune_nodes = []
    for node in graph.nodes():
        node_degree = graph.degree(node)
        if node_degree < degree:
            to_prune_nodes.append(node)
    graph_pruned = graph.copy()
    graph_pruned.remove_nodes_from(to_prune_nodes)
    return graph_pruned


def remove_degree_2_nodes(graph, occ_map, agent_node):
    d2_nodes = [node for node in graph.nodes if graph.degree(node) == 2]
    for node in d2_nodes:
        if node == agent_node:
            continue
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 2:
            # if not is_visible(occ_map, graph.nodes[neighbors[0]]['o'], graph.nodes[neighbors[1]]['o']):
            #     continue
            rel_angle, dist1, dist2 = compute_angle(
                graph.nodes[node]["o"],
                graph.nodes[neighbors[0]]["o"],
                graph.nodes[neighbors[1]]["o"],
            )
            if rel_angle < 100:
                continue
            if not graph.has_edge(neighbors[0], neighbors[1]):
                graph.add_edge(
                    neighbors[0],
                    neighbors[1],
                    weight=get_euclidean_dist(
                        graph.nodes[neighbors[0]]["o"], graph.nodes[neighbors[1]]["o"]
                    ),
                )
            graph.remove_node(node)
    return graph


def remove_close_nodes(graph, agent_node, threshold=20):
    sorted_nodes = [(agent_node, 1000)] + sorted(
        (node for node in graph.degree if node[0] != agent_node),
        key=lambda x: x[1],
        reverse=True,
    )

    for node, deg in sorted_nodes:
        if deg == 1:
            continue
        if node not in graph:
            continue
        for neighbor in list(graph.neighbors(node)):
            if neighbor == node:
                continue
            if neighbor == agent_node:
                continue
            if (
                get_euclidean_dist(graph.nodes[node]["o"], graph.nodes[neighbor]["o"])
                < threshold
            ):
                nnodes = list(graph.neighbors(neighbor))
                for n in nnodes:
                    if n != node:
                        graph.add_edge(
                            node,
                            n,
                            weight=get_euclidean_dist(
                                graph.nodes[node]["o"], graph.nodes[n]["o"]
                            ),
                        )
                graph.remove_node(neighbor)
    return graph


def compute_slopes(points):
    slopes = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = float("inf")  # Infinite slope (vertical line)
        slopes.append(slope)
    return slopes


def find_violate_changes(points):
    slopes = compute_slopes(points)
    changes = []

    for i in range(1, len(slopes)):
        change = abs(slopes[i] - slopes[i - 1])
        changes.append((change, points[i]))

    changes.sort(key=lambda x: x[0], reverse=True)

    violate_points = [point for _, point in changes]

    return np.array(violate_points)


def add_visible_nodes(graph, voro_graph, occ_map):
    def find_visible_path(points, occ_map):
        def find_midpoints(p_ind1, p_ind2, occ_map, vis_path):
            p1 = points[p_ind1]
            p2 = points[p_ind2]
            if is_visible(occ_map, np.array(p1), np.array(p2)):
                vis_path.append((p_ind1, p_ind2))
            else:
                mid_ind = (p_ind1 + p_ind2) // 2
                if mid_ind in [p_ind1, p_ind2]:
                    return
                find_midpoints(p_ind1, mid_ind, occ_map, vis_path)
                find_midpoints(mid_ind, p_ind2, occ_map, vis_path)

        start_idx = 0
        end_idx = len(points) - 1
        vis_path = []
        find_midpoints(start_idx, end_idx, occ_map, vis_path)
        return vis_path

    from itertools import islice

    def k_shortest_paths(G, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )

    sorted_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    node_idx = len(voro_graph.nodes) + 1
    processed_nodes = []
    # graph_new = copy.deepcopy(graph)

    for node, deg in sorted_nodes:
        processed_nodes.append(node)
        if deg == 1:
            continue
        if node not in graph:
            continue
        for neighbor in list(graph.neighbors(node)):
            if neighbor in processed_nodes:
                continue

            if is_visible(occ_map, graph.nodes[node]["o"], graph.nodes[neighbor]["o"]):
                continue

            if neighbor not in voro_graph:
                continue

            # shortest_path = nx.shortest_path(voro_graph, source=node, target=neighbor, weight='weight')
            # paths = nx.all_simple_paths(voro_graph, source=node, target=neighbor, cutoff=len(shortest_path)+3)
            paths = k_shortest_paths(voro_graph, node, neighbor, 5, weight="weight")
            for sp_nodes in paths:
                conn_voros = []
                for e in range(len(sp_nodes) - 1):
                    mid_conn_voros = voro_graph[sp_nodes[e]][sp_nodes[e + 1]]["pts"]
                    if get_euclidean_dist(
                        voro_graph.nodes[sp_nodes[e]]["o"], mid_conn_voros[0]
                    ) > get_euclidean_dist(
                        voro_graph.nodes[sp_nodes[e]]["o"], mid_conn_voros[-1]
                    ):
                        mid_conn_voros = mid_conn_voros[::-1]
                    conn_voros.append(mid_conn_voros)
                conn_voros = np.concatenate(conn_voros, axis=0)

                conn_nodes = []
                for voro in conn_voros:
                    if is_visible(occ_map, graph.nodes[node]["o"], voro) and is_visible(
                        occ_map, graph.nodes[neighbor]["o"], voro
                    ):
                        conn_nodes.append(voro)
                if len(conn_nodes) > 0:
                    graph.add_node(node_idx, o=conn_nodes[(len(conn_nodes) - 1) // 2])
                    graph.add_edge(
                        node,
                        node_idx,
                        weight=get_euclidean_dist(
                            graph.nodes[node]["o"], graph.nodes[node_idx]["o"]
                        ),
                    )
                    graph.add_edge(
                        node_idx,
                        neighbor,
                        weight=get_euclidean_dist(
                            graph.nodes[neighbor]["o"], graph.nodes[node_idx]["o"]
                        ),
                    )
                    graph.remove_edge(node, neighbor)
                    node_idx += 1
                    break

            if len(conn_nodes) == 0:
                vis_path = find_visible_path(conn_voros.tolist(), occ_map)
                # if node == 51 and neighbor == 45:
                #     print(graph.nodes[node]['o'], graph.nodes[neighbor]['o'])
                #     print(conn_voros)
                #     print(vis_path)
                #     exit()
                node_i = 0
                neighbor_i = len(conn_voros) - 1
                record = {node_i: node, neighbor_i: neighbor}
                # if len(vis_path) > 4:
                # draw_topo_graph(occ_map, graph, f'init_{node}_{neighbor}')
                # print(f'init_{node}_{neighbor}', vis_path)
                for snode, enode in vis_path:
                    if snode not in record:
                        graph.add_node(node_idx, o=conn_voros[snode])
                        record[snode] = node_idx
                        node_idx += 1
                    if enode not in record:
                        graph.add_node(node_idx, o=conn_voros[enode])
                        record[enode] = node_idx
                        node_idx += 1
                    graph.add_edge(
                        record[snode],
                        record[enode],
                        weight=get_euclidean_dist(
                            graph.nodes[record[snode]]["o"],
                            graph.nodes[record[enode]]["o"],
                        ),
                    )
                    # if len(vis_path) > 4:
                    # draw_topo_graph(occ_map, graph, f'{record[snode]}_{record[enode]}')

                graph.remove_edge(node, neighbor)
                # if len(vis_path) > 4:
                # draw_topo_graph(occ_map, graph, f'{node}_{neighbor}_remove')

    return graph
    # graph_new = nx.relabel.convert_node_labels_to_integers(graph, first_label=0)
    # return graph_new


def add_visible_edges(graph, occ_map):
    nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            s_node = nodes[i][0]
            e_node = nodes[j][0]
            if graph.has_edge(s_node, e_node):
                continue
            if is_visible(occ_map, graph.nodes[s_node]["o"], graph.nodes[e_node]["o"]):
                flag = True
                for neighbor in graph.neighbors(s_node):
                    rel_angle, dist1, dist2 = compute_angle(
                        graph.nodes[s_node]["o"],
                        graph.nodes[e_node]["o"],
                        graph.nodes[neighbor]["o"],
                    )
                    if rel_angle <= 30:
                        flag = False
                        if dist1 < dist2:
                            graph.add_edge(s_node, e_node, weight=dist1)
                            graph.remove_edge(s_node, neighbor)
                        break

                for neighbor in graph.neighbors(e_node):
                    rel_angle, dist1, dist2 = compute_angle(
                        graph.nodes[e_node]["o"],
                        graph.nodes[s_node]["o"],
                        graph.nodes[neighbor]["o"],
                    )
                    if rel_angle <= 30:
                        flag = False
                        if dist1 < dist2:
                            graph.add_edge(s_node, e_node, weight=dist1)
                            graph.remove_edge(e_node, neighbor)
                        break

                if flag:
                    graph.add_edge(
                        s_node,
                        e_node,
                        weight=get_euclidean_dist(
                            graph.nodes[s_node]["o"], graph.nodes[e_node]["o"]
                        ),
                    )
    return graph


def add_world_coordinates(graph, env, grid_resolution):
    for node, grid in graph.nodes(data="o"):
        world_coord = maps.from_grid(grid[0], grid[1], grid_resolution, env)
        graph.nodes[node]["pos"] = world_coord
        graph.nodes[node]["o"] = graph.nodes[node]["o"].tolist()
    return graph


def sparsify_graph(topology_graph: nx.Graph, voxel_size: float):
    """
    Sparsify a topology graph by removing nodes with degree 2.
    This algorithm first starts at degree-one nodes (dead ends) and
    removes all degree-two nodes until confluence nodes are found.
    Next, we find close pairs of higher-order degree nodes and
    delete all nodes if the shortest path between two nodes consists
    only of degree-two nodes.
    Args:
        graph (nx.Graph): graph to sparsify
    Returns:
        nx.Graph: sparsified graph
    """
    graph = copy.deepcopy(topology_graph)

    if len(graph.nodes) < 10:
        return graph
    # all nodes with degree 1 or 3+
    new_node_candidates = [
        node for node in list(graph.nodes) if (graph.degree(node) != 2)
    ]

    new_graph = nx.Graph()
    for i, node in enumerate(new_node_candidates):
        new_graph.add_node(node)

    all_path_dense_graph = dict(nx.all_pairs_dijkstra_path(graph, weight="dist"))

    sampled_edges_to_add = list()
    new_nodes = set(new_graph.nodes)
    new_nodes_list = list(new_graph.nodes)
    for i in range(len(new_graph.nodes)):
        for j in range(len(new_graph.nodes)):
            if i < j:
                # Go through all edges along path and extract dist
                node1 = new_nodes_list[i]
                node2 = new_nodes_list[j]
                path = all_path_dense_graph[node1][node2]
                for node in path[1:-1]:
                    if graph.degree(node) > 2:
                        break
                else:
                    sampled_edges_to_add.append(
                        (
                            path[0],
                            path[-1],
                            np.linalg.norm(np.array(path[0]) - np.array(path[-1])),
                        )
                    )
                    dist = [
                        graph.edges[path[k], path[k + 1]]["dist"]
                        for k in range(len(path) - 1)
                    ]
                    mov_agg_dist = 0
                    predecessor = path[0]
                    # connect the nodes if there is a path between them that does not go through any other of the new nodes
                    if len(path) and len(set(path[1:-1]).intersection(new_nodes)) == 0:
                        for cand_idx, cand_node in enumerate(path[1:-1]):
                            mov_agg_dist += dist[cand_idx]
                            if mov_agg_dist * voxel_size > 0.6:
                                sampled_edges_to_add.append(
                                    (
                                        predecessor,
                                        cand_node,
                                        np.linalg.norm(
                                            np.array(predecessor) - np.array(cand_node)
                                        ),
                                    )
                                )
                                predecessor = cand_node
                                mov_agg_dist = 0
                            else:
                                continue
                        sampled_edges_to_add.append(
                            (
                                predecessor,
                                path[-1],
                                np.linalg.norm(
                                    np.array(predecessor) - np.array(path[-1])
                                ),
                            )
                        )

    for edge_param in sampled_edges_to_add:
        k, l, dist = edge_param
        if k not in new_graph.nodes:
            new_graph.add_node(k)
        if l not in new_graph.nodes:
            new_graph.add_node(l)
        new_graph.add_edge(k, l, dist=dist)
    return new_graph


def draw_graph(occ_map, graph, fig_name):
    plt.imshow(occ_map, cmap="gray")
    # draw edges by pts
    for (s, e) in graph.edges():
        try:
            ps = graph[s][e]["pts"]
            plt.plot(ps[:, 1], ps[:, 0], "green")
        except:
            pass

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]["o"] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], "r.")

    plt.savefig(f"{fig_name}.png")
    plt.close()


def draw_topo_graph(
    occ_map, graph_raw, fig_name, goal_node=None, vp_node=None, trajectory=None
):
    if trajectory is not None:
        t_xy = [
            (graph_raw.nodes[n]["o"][1], graph_raw.nodes[n]["o"][0]) for n in trajectory
        ]
    if goal_node is not None:
        goal_x, goal_y = (
            graph_raw.nodes[goal_node]["o"][1],
            graph_raw.nodes[goal_node]["o"][0],
        )
    if vp_node is not None:
        vp_x, vp_y = graph_raw.nodes[vp_node]["o"][1], graph_raw.nodes[vp_node]["o"][0]

    graph = nx.relabel.convert_node_labels_to_integers(graph_raw, first_label=0)
    nodes = graph.nodes()
    all_node_coords = (
        np.array([nodes[i]["o"] for i in nodes]).transpose()[[1, 0], :].astype(np.int32)
    )
    x = all_node_coords[0, :].flatten()
    y = all_node_coords[1, :].flatten()
    edges = list(graph.edges())
    edges = np.array(edges).astype(np.int32)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
    ax.imshow(occ_map)
    ax.plot(
        x[edges.T],
        y[edges.T],
        linestyle="-",
        color="y",
        markerfacecolor="red",
        marker="o",
        zorder=1,
    )
    ax.scatter(
        x=all_node_coords[0, :], y=all_node_coords[1, :], c="red", s=30, zorder=2
    )

    if trajectory is not None:
        for t_x, t_y in t_xy:
            ax.scatter(x=t_x, y=t_y, c="blue", s=50, zorder=4)
    if goal_node is not None:
        ax.scatter(x=goal_x, y=goal_y, c="green", s=50, zorder=4)
    if vp_node is not None:
        ax.scatter(x=vp_x, y=vp_y, c="cyan", s=50, zorder=4)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig(f"{fig_name}.png")
    plt.close()
