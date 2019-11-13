from surfacemovement import Trajectory, Flight
# import utils
import numpy as np
import pandas as pd
import similaritymeasures
import matplotlib.pyplot as plt
import osmnx as ox
from collections import OrderedDict
from geometric import GeometricPoint, GeometricEdge
from IPython.display import display
from collections import deque
import copy
import networkx as nx
import itertools

class WeightBasedMapMatching(object):
    def __init__(self, graph, traveled_dist_threshold=5, k1=0.43, k2=0.43, dist_confidence=10):
        self.__graph = graph
        self.tracked_points = None
        self.cur_point = None
        self.cur_idx = None
        self.traveled_dist_threshold = traveled_dist_threshold
        self.k1 = k1
        self.k2 = k2
        self.verbal = False
        self.dist_confidence = dist_confidence


    def set_verbal(self, value):
        self.verbal = value

    def determine_candidate_segments(self, gps_point, num_nearest_node=3):
        l_nearest_nodes, l_dists = self.__graph.get_nearest_node(gps_point, num_node_return=num_nearest_node)
        candidates = []
        for n_id in l_nearest_nodes:
            candidates += [edge_id for edge_id in self.__graph.get_list_edge_id() if n_id in edge_id]

        return list(set(candidates))

    def calculate_distance_score(self, point, candidates):
        candidates_edge = [self.__graph.get_edge(edge_id) for edge_id in candidates]
        candidates_dist = [edge.dist_to_point(point) for edge in candidates_edge]
        candidates_dist_score = np.array(candidates_dist) / sum(candidates_dist)
        dist_weight = 1

        return candidates_dist_score, candidates_dist, dist_weight

    def calculate_direction_diff_score(self, point, prev_point, candidates, travel_dist_threshold=3):
        if prev_point is None:
            candidates_dir_score = np.zeros(len(candidates))
            candidates_dir = np.zeros(len(candidates))
            dir_weight = 0
        else:
            tracked_points_line = GeometricEdge(prev_point, point)
            candidates_edge = [self.__graph.get_edge(edge_id) for edge_id in candidates]
            candidates_dir = [tracked_points_line.angle_with(edge) for edge in candidates_edge]
            candidates_dir_score = np.array(candidates_dir) / sum(candidates_dir)

            line_dist = tracked_points_line.get_dist()
            dir_weight = 1 if  line_dist > travel_dist_threshold else line_dist / travel_dist_threshold
        return candidates_dir_score, candidates_dir, dir_weight

    def calculate_heading_diff(self, point, next_point, point_velocity, candidates, travel_dist_threshold=3):
        if next_point is None:
            candidates_dir_score = np.zeros(len(candidates))
            candidates_dir = np.zeros(len(candidates))
            dir_weight = 0
        else:
            tracked_points_line = GeometricEdge(point, next_point)
            candidates_edge = [self.__graph.get_edge(edge_id) for edge_id in candidates]
            candidates_dir = [tracked_points_line.angle_with(edge) for edge in candidates_edge]
            candidates_dir_score = np.array(candidates_dir) / sum(candidates_dir)

            line_dist = tracked_points_line.get_dist()
            dir_weight = 1 if line_dist > travel_dist_threshold else line_dist / travel_dist_threshold
        return candidates_dir_score, candidates_dir, dir_weight

    def check_the_same_edge(self, str_edge1, str_edge2):

        if str_edge1 == str_edge2:
            return True

        if str_edge2 is None or str_edge1 is None:
            return False

        edge1 = tuple([int(s) for s in str_edge1[1:-1].split(',')])
        edge2 = tuple([int(s) for s in str_edge2[1:-1].split(',')])

        if edge2 == (edge1[1], edge1[0]):
            return True
        else:
            return False


    def calculate_candidate_score(self, point, prev_point, next_point, point_velocity, candidates):

        candidates_dist_score, candidates_dist, dist_weight = self.calculate_distance_score(point, candidates)
        candidates_heading_diff_score, candidates_heading_diff, heading_weight = self.calculate_heading_diff(point, next_point, point_velocity, candidates)
        candidates_dir_diff_score, candidates_dir_diff, dir_weight = self.calculate_direction_diff_score(point, prev_point, candidates)

        candidates_score = dist_weight * candidates_dist_score + heading_weight * candidates_heading_diff_score + dir_weight * candidates_dir_diff_score

        df_candidates = pd.DataFrame({'candidates': [str(edge_id) for edge_id in candidates],
                                      'dist': candidates_dist,
                                      'dist_score': candidates_dist_score,
                                      'heading_diff': candidates_heading_diff,
                                      'heading_diff_score': candidates_heading_diff_score,
                                      'dir_diff': candidates_dir_diff,
                                      'dir_diff_score': candidates_dir_diff_score,
                                      'Score': candidates_score})
        df_candidates['dist_weight'] = dist_weight
        df_candidates['heading_weight'] = heading_weight
        df_candidates['dir_weight'] = dir_weight

        df_candidates = df_candidates.sort_values(['Score', 'heading_diff'], ascending=True)
        best_segment = df_candidates.iloc[0]
        second_best_segment = df_candidates.iloc[1]

        # Check is the samge edge
        if self.check_the_same_edge(best_segment['candidates'], second_best_segment['candidates']):
            second_best_segment = df_candidates.iloc[2]

        if self.verbal:
            display(pd.concat([pd.DataFrame(best_segment), pd.DataFrame(second_best_segment)], axis=1).T)
        # display(second_best_segment)
        if heading_weight == 0 and dir_weight == 0:
            confidence_level = 0
        elif best_segment['dist'] > self.dist_confidence:
            confidence_level = 0
        else:
            confidence_level = 1
        # elif prev_point is None:
        #     confidence_level = 1 / (dist_weight + heading_weight + dist_weight) * (
        #             (dist_weight * (second_best_segment['dist'] - best_segment['dist']) / (
        #                         second_best_segment['dist'] + best_segment['dist'])) +
        #             (heading_weight * (second_best_segment['heading_diff'] - best_segment['heading_diff']) / (
        #                         second_best_segment['heading_diff'] + best_segment['heading_diff']))
        #     )
        # else:
        #     confidence_level = 1 / (dist_weight + heading_weight + dist_weight) * (
        #         (dist_weight * (second_best_segment['dist'] - best_segment['dist']) / (second_best_segment['dist'] + best_segment['dist'])) +
        #         (heading_weight * (second_best_segment['heading_diff'] - best_segment['heading_diff']) / (second_best_segment['heading_diff'] + best_segment['heading_diff'])) +
        #         (dir_weight * (second_best_segment['dir_diff'] - best_segment['dir_diff']) / (second_best_segment['dir_diff'] + best_segment['dir_diff']))
        #     )

        return df_candidates, confidence_level

    def intersection_crossing_detection(self, point, prev_point, point_velocity, cur_segment):
        # Condition 1
        if cur_segment.is_point_beyond(point):
            return True

        # Condition 2
        cur_direction = GeometricEdge(prev_point, point)
        if cur_direction.angle_with(cur_segment) > 45:
            return True

        # Condition 3

        return False

    def get_conectivity_info(self, cur_segment, prev_segment):
        if cur_segment[0] == prev_segment[1]:
            edge1 = (cur_segment[0], cur_segment[1])
            edge2 = (prev_segment[1], prev_segment[0])
            angle = self.__graph.get_edge(edge1).angle_with(self.__graph.get_edge(edge2))
            is_direct_connected = True
            nodes_between = []
        else:
            try:
                nodes_between = self.__graph.shortest_path(prev_segment[1], cur_segment[0], weight='length')
            except:
                print(prev_segment[1], cur_segment[0])
                raise
            is_direct_connected = False
            edge1 = (prev_segment[1], nodes_between[1])
            edge2 = (prev_segment[1], prev_segment[0])
            angle = self.__graph.get_edge(edge1).angle_with(self.__graph.get_edge(edge2))

        return is_direct_connected, angle, nodes_between


    def run_steps_v2(self, points, points_velocity, num_best_segment=2):
        prev_point = None
        prev_segment = None
        status = "init"
        should_assign_segment = False
        best_segment = None
        for idx, (point, point_velocity) in enumerate(zip(points, points_velocity)):
            if self.verbal:
                print(idx)
                print(status, point.get_coordinate())
            if idx < (len(points) - 1):
                next_point = points[idx+1]
            if status in ["next-segment", 'init', "same-segment"]:
                candidates = self.determine_candidate_segments(point)
                df_candidate, confidence = self.calculate_candidate_score(point, prev_point, next_point, point_velocity, candidates)

                df_result = df_candidate.dropna()
                if len(df_result) > 0:
                    best_segments = []
                    for i in range(num_best_segment):
                        best_segment = tuple([int(s) for s in df_result.iloc[i]['candidates'][1:-1].split(',')])
                        if best_segment not in best_segments:
                            best_segments.append(best_segment)


                    score = df_result.iloc[0]['Score']
                    # best_segment_edge = self.__graph.get_edge(best_segment)
                    # display(df_candidate)
                    # confidence = self.k2 + 0.07 - df_result.iloc[0]['Score']

                    if confidence < self.k2:
                        best_segments = None

                    yield best_segments, score, point, points_velocity
                else:
                    yield None, None, point, points_velocity

    def is_valid_adding(self, route, node):
        if node in route:
            return False  # Circle in graph

        if len(route) == 1:
            return True

        # cur_edge = (route[-1], route[-2])
        # next_edge = (route[-1], node)
        # angle = self.__graph.get_edge(cur_edge).angle_with(self.__graph.get_edge(next_edge))

        cur_edge = (route[-2], route[-1])
        next_edge = (route[-1], node)
        angle = self.__graph.get_angle(cur_edge, next_edge)
        if angle < 90:
            return False
        else:
            return True

    def compute_route_score(self, route, seg_counts):
        score = 0
        for seg in zip(route, route[1:]):
            seg = tuple(sorted(seg))
            score += seg_counts.get(seg, 0)
        return score

    def run_v2(self, points, points_velocity, num_init_point=12):
        generator = self.run_steps_v2(points, points_velocity, num_best_segment=2)
        accept_point_id = []
        l_results = []
        for point_idx, l in enumerate(generator):
            if l[0] is not None:
                l_results.append(l)
                accept_point_id.append(point_idx)

        # l_results = [l for l in list(generator) if l[0] is not None]
        self.l_results = l_results

        # node_start = l_results[0][0][0]
        # node_end = l_results[-1][0][1]
        connect_dict = {}
        pre_seg = None
        l_raw_seg = []
        # Add more nodes
        pre_segs = None
        l_best_segs = []
        for segs, _, _, _ in l_results:
            if segs is None:
                continue

            l_best_segs.append(tuple(sorted(segs[0])))

            if pre_segs is not None:
                for seg, pre_seg in itertools.product(segs, pre_segs):
                    _, _, nodes = self.get_conectivity_info(seg, pre_seg)
                    if len(nodes) > 1:
                        l_raw_seg += list(zip(nodes, nodes[1:]))

            l_raw_seg += segs
            pre_segs = segs

        # print(set(l_raw_seg))
        l_seg = []
        all_node = []
        for seg in l_raw_seg:
            if seg is None:
                continue

            all_node += list(seg)

            connect_dict[seg[0]] = list(set(connect_dict.get(seg[0], []) + [seg[1]]))
            l_seg.append(tuple(sorted(seg)))

        self.connect_dict = connect_dict

        seg_counts = pd.Series(l_best_segs).value_counts()
        possible_routes = []
        candidate_routes = deque()

        cur_route = list(l_raw_seg[0])
        cur_node = cur_route[-1]

        l_unique_node = list(OrderedDict.fromkeys(all_node))
        self.l_unique_node = l_unique_node

        for node in l_unique_node[:num_init_point]:
            candidate_routes.append([node])

        # l_seg_unique = list(OrderedDict.fromkeys(l_raw_seg))
        # self.l_seg_unique = l_seg_unique
        #
        # # Add more candidate
        # for seg in l_seg_unique[1:5]:
        #     if seg not in candidate_routes:
        #         candidate_routes.append(list(seg))



        while (1):
            if cur_route is None:
                break

            # if self.verbal:
            #     print('cur_node', cur_node)
            #     print('cur_route', cur_route)
            #     print('candidate_routes', candidate_routes)

            # if cur_node == node_end:
            #     # Get successful trajectory
            #     possible_routes.append(copy.deepcopy(cur_route))
            #     if len(candidate_routes) > 0:
            #         cur_route = candidate_routes.popleft()
            #         cur_node = cur_route[-1]
            #     else:
            #         cur_route = None

            if (cur_node not in connect_dict):  # need solve circle in graph
                possible_routes.append(copy.deepcopy(cur_route))
                if len(candidate_routes) > 0:
                    cur_route = candidate_routes.popleft()
                    cur_node = cur_route[-1]
                else:
                    cur_route = None
            else:
                all_connect_nodes = connect_dict[cur_node]

                # Add candidate
                for node in all_connect_nodes[1:]:
                    new_route = copy.deepcopy(cur_route)
                    if self.is_valid_adding(new_route, node):
                        new_route.append(node)
                        candidate_routes.append(new_route)

                if self.is_valid_adding(cur_route, all_connect_nodes[0]):
                    cur_route.append(all_connect_nodes[0])
                    cur_node = all_connect_nodes[0]
                else:
                    possible_routes.append(copy.deepcopy(cur_route))
                    # Eliminate current route
                    if len(candidate_routes) > 0:
                        cur_route = candidate_routes.popleft()
                        cur_node = cur_route[-1]
                    else:
                        cur_route = None

        best_route = None
        best_score = 0
        self.possible_routes = possible_routes
        self.seg_counts = seg_counts
        df_score = pd.DataFrame({'route': possible_routes,
                                 'score': [self.compute_route_score(route, seg_counts) for route in possible_routes],
                                 'length': [len(route) for route in possible_routes]})

        df_score = df_score.sort_values(['score', 'length'],  ascending=[False, True])
        self.df_score = df_score
        best_route = df_score.iloc[0]['route']

        # for route in possible_routes:
        #     score = self.compute_route_score(route, seg_counts)
        #     if score > best_score:
        #         best_score = score
        #         best_route = route

        return best_route, accept_point_id

    def run_on_flight(self, flight, sampling_distance=10):
        if sampling_distance is not None:
            points, points_velocity, timestamp = flight.get_sampling(distance_diff=sampling_distance)
        else:
            points = [p for p in flight]
            points_velocity = flight.get_velocity(type_velocity='single')

        route, accept_point_id = self.run_v2(points, points_velocity)
        # segments = list(zip(route, route[1:]))
        return route, accept_point_id


def remove_point_by_velocity(flight, min_velocity=0.5, max_velocity=15):
    accept_idx = []
    for idx, v in enumerate(flight.get_velocity(type_velocity='single')):
        if v > min_velocity and v < max_velocity:
            accept_idx.append(idx)

    new_flight = copy.deepcopy(flight)
    new_flight.reselect(accept_idx)
    return new_flight


def get_pointwise_info(flight):
    l_point_id = []
    l_dist_from_previous_point = []
    l_direction = []

    num_point = len(flight)
    for idx, point in enumerate(flight):
        if (idx == 0) or (idx == len(flight) -1):
           continue

        prev_point = flight[idx-1]
        cur_point = flight[idx]
        next_point = flight[idx+1]

        dist = prev_point.dist_to(cur_point)
        angle = GeometricEdge(prev_point, cur_point).angle_with(GeometricEdge(cur_point, next_point))


def func_cutting_by_runway(airport_graph, list_node_id):
    cutting_nodes = []
    if list_node_id[0] in airport_graph.get_runway_nodes():
        detected_type = 'arrival'
    elif list_node_id[-1] in airport_graph.get_runway_nodes():
        detected_type = 'departure'
    else:
        detected_type = None

    if detected_type == 'departure':
        for node_id in list_node_id:
            if node_id in airport_graph.get_runway_nodes():
                cutting_nodes.append(node_id)
                break
            cutting_nodes.append(node_id)

    elif detected_type == 'arrival':
        for idx, node_id in enumerate(list_node_id):
            if node_id in airport_graph.get_runway_nodes():
                continue

            if list_node_id[idx-1] not in cutting_nodes:
                cutting_nodes.append(list_node_id[idx-1])
            cutting_nodes.append(node_id)

    else:
        cutting_nodes = list_node_id

    return cutting_nodes

def find_route_v2(airport_graph, list_points, cutting_by_runway=True):
    # list_points = [GeometricPoint(lat, lng) for lat, lng in list_coordinate]
    assigned_node_ids = [airport_graph.assign_node_id_to_point(point) for point in list_points]
    node_ids = [int(node_id) for node_id in assigned_node_ids if node_id is not None]
    node_ids = list(OrderedDict.fromkeys(node_ids).keys())
    connected_seq_node_id = airport_graph.forced_connected_property(node_ids)

    if cutting_by_runway:
        connected_seq_node_id = func_cutting_by_runway(airport_graph, connected_seq_node_id)

    return connected_seq_node_id

def map_matching_v3(airport_graph, flight, cutting_by_runway=False, type_matching='v2', **kwargs):
    if type_matching == 'v2':
        # print('V2')
        route = find_route_v2(airport_graph, [p for p in flight], cutting_by_runway, **kwargs)
    else:
        route = airport_graph.find_route([p for p in flight], cutting_by_runway, **kwargs)

    traj = Trajectory(airport_graph, route)
    return traj

def map_matching_v2(airport_graph, flight, cutting_by_runway=True, type_matching='v2', **kwargs):
    if type_matching == 'v2':
        # print('V2')
        route = find_route_v2(airport_graph, [p for p in flight], cutting_by_runway, **kwargs)
    else:
        route = airport_graph.find_route([p for p in flight], cutting_by_runway, **kwargs)

    traj = Trajectory(airport_graph, route)
    end_point = traj.get_points()[-1]
    starting_point = traj.get_points()[0]
    cutted_flight = utils.cut_flight_by_point(flight, end_point)
    cutted_flight = utils.cut_flight_by_starting_point(cutted_flight, starting_point, threshold=10)
    return traj, cutted_flight

def map_matching(airport_graph, flight, cutting_by_runway=True, type_matching='v2', **kwargs):
    if type_matching == 'v2':
        # print('V2')
        route = airport_graph.find_route_v2([p for p in flight], cutting_by_runway, **kwargs)
    else:
        route = airport_graph.find_route([p for p in flight], cutting_by_runway, **kwargs)

    traj = Trajectory(airport_graph, route)
    end_point = traj.get_points()[-1]
    starting_point = traj.get_points()[0]
    cutted_flight = utils.cut_flight_by_point(flight, end_point)
    cutted_flight = utils.cut_flight_by_starting_point(cutted_flight, starting_point, threshold=10)
    return traj, cutted_flight

def points_to_array(points):
    array = np.zeros((len(points), 2))
    for idx, p in enumerate(points):
        array[idx, :] = p.get_coordinate(type_coordinate='geographic')
    return array

def frechet_dist(flight, trajectory):
    flight_array = points_to_array([p for p in flight])
    trajectory_array = points_to_array(trajectory.get_points())
    dist = similaritymeasures.frechet_dist(flight_array, trajectory_array)
    return dist

def total_dist(flight, trajectory, normalize=True):
    flight_dist = sum(flight.compute_dists())
    traj_dist = sum(trajectory.get_dist())
    diff_dist = flight_dist - traj_dist
    if normalize:
        return diff_dist / flight_dist
    else:
        return diff_dist

def total_offset_dist(flight, trajectory):
    l_dist = []
    for idx, offset in flight.get_offset():
        print('Id', idx)
        print('Offset', idx)
        flight_point = flight[idx]
        traj_point = trajectory.point_at_offset(offset)
        l_dist.append(flight_point.dist_to(traj_point))

    return l_dist

def projected_dist(flight, trajectory):
    points_dist = []
    for point in flight:
        dist = trajectory.shapely_dist_to_point(point)
        points_dist.append(dist)

    return np.mean(points_dist)


def estimate_trajectory_velocities_and_durations(graph, trajectory, points, point_velocities, point_timestamps):
    nodes = trajectory.get_node_ids()
    subgraph = graph.subgraph(nodes)
    edges = [ox.get_nearest_edge(subgraph, p) for p in points]
    edge_velocities = {}
    edge_timestamps = {}
    for edge_id, velocity, timestamp in zip(edges, point_velocities, point_timestamps):
        edge_velocities[edge_id] = edge_velocities.get(edge_id, []) + [velocity]
        edge_timestamps[edge_id] = edge_timestamps.get(edge_id, []) + [timestamp]


def plot_flight_trajectory_comparision(flight, traj):
    exp_data = points_to_array([p for p in flight])
    num_data = points_to_array(traj.get_points())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # ax.scatter(exp_data[:, 0], exp_data[:, 1], '.', label='Experimental data',
    #         c=np.linspace(0, 1, exp_data.shape[0]), cmap='YlGnBu')

    x_min = np.min([np.min(exp_data[:, 0]), np.min(num_data[:, 0])])
    x_max = np.max([np.max(exp_data[:, 0]), np.max(num_data[:, 0])])
    y_min = np.min([np.min(exp_data[:, 1]), np.min(num_data[:, 1])])
    y_max = np.max([np.max(exp_data[:, 1]), np.max(num_data[:, 1])])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # First Point
    ax.scatter(exp_data[0, 0], exp_data[0, 1], c='red', s=50)
    ax.annotate("Starting point", (exp_data[0, 0], exp_data[0, 1]))


    ax.scatter(exp_data[:, 0], exp_data[:, 1], label='Experimental data',
            c=np.linspace(0, 1, exp_data.shape[0]), cmap='YlGnBu')
    ax.plot(num_data[:, 0], num_data[:, 1], '-', label='Numerical model', c='r')
    print(x_min, x_max)
    print(y_min, y_max)
    ax.grid(True)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    # fig.xlabel('X', fontsize=16)
    # fig.ylabel('Y', fontsize=16)
    fig.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    # legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.axis('equal')
    return fig, ax


