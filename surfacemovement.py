from datetime import datetime, timedelta
import smopy
import folium
from multiprocessing import Pool
import multiprocessing

import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

from collections import OrderedDict
from geometric import GeometricPoint, GeometricEdge
import geometric
from airportgraph import AirportGraph
import copy
from shapely.geometry import Polygon, Point, LinearRing, LineString


CHANGI_BOUNDING_BOX = (1.3346, 103.9710, 1.3804,104.0106)
TIME_FORMATING = "%m/%d/%Y, %H:%M:%S"

def plot_coordinates(coordinates, size=2, colors=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(18, 10))  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    l_lat = []
    l_lng = []
    for lat, lng in coordinates:
        l_lat.append(lat)
        l_lng.append(lng)

    if colors is None:
        ax.scatter(l_lat, l_lng, s=size, c=np.linspace(0, 1, len(coordinates)), cmap='YlGnBu')
    else:
        ax.scatter(l_lat, l_lng, c=colors, s=size)

    return ax

def resampling_flight(flight, distance_diff = 5):
    p_dist_diff = []
    p_lat = []
    p_lng = []
    p_velocity = []
    p_time_diff = []
    points = [p for p in flight]
    single_velocities = flight.get_velocity(type_velocity='single')
    timestamps = flight.get_timestamp()
    for idx, (point, velocity, timestamp) in enumerate(zip(points, single_velocities, timestamps)):
        if idx == 0:
            dist_diff = 0
            time_diff = 0
        else:
            prev_point = points[idx-1]
            prev_timestamp = timestamps[idx-1]
            dist_diff = prev_point.dist_to(point)
            time_diff = (timestamp - prev_timestamp).seconds

        lat, lng = tuple(point.get_coordinate())
        p_lat.append(lat)
        p_lng.append(lng)
        p_velocity.append(velocity)
        p_dist_diff.append(dist_diff)
        p_time_diff.append(time_diff)

    p_cum_sum_dist = np.cumsum(p_dist_diff)
    p_cum_sum_time = np.cumsum(p_time_diff)

    # new_cum_sum_dist = np.cumsum([distance_diff]*len(p_dist_diff))
    num_point = int((max(p_cum_sum_dist) - min(p_cum_sum_dist))/distance_diff) + 1
    new_cum_sum_dist = np.linspace(min(p_cum_sum_dist), max(p_cum_sum_dist), num_point)

    inteporlate_p_lat = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_lat)
    inteporlate_p_lng = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_lng)
    inteporlate_p_velocity = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_velocity)
    inteporlate_p_cum_sum_time = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_cum_sum_time)
    l_points = [GeometricPoint(lat, lng) for lat, lng in zip(inteporlate_p_lat, inteporlate_p_lng)]
    new_timestamp = [flight.starting_time + timedelta(seconds=s) for s in inteporlate_p_cum_sum_time]

    new_flight = Flight(flight.flight_id, aircraft_type=flight.aircraft_type, gate=flight.gate, is_arrival=flight.is_arrival,
                        runway=flight.runway, points=l_points, timestamps=new_timestamp,
                        velocities=None, flight_identity=flight.flight_identity)
    new_flight.set_single_velocity(inteporlate_p_velocity)
    return new_flight

class Flight(object):
    def __init__(self, flight_id, aircraft_type, gate, is_arrival, runway, points, timestamps, velocities, flight_identity=None):
        self.flight_id = flight_id
        self.aircraft_type = aircraft_type
        self.gate = gate
        self.runway = runway
        self.is_arrival = is_arrival
        self.__points = points
        self.__timestamp = timestamps
        self.__timeindexes = pd.Series(data=list(range(len(self.__timestamp))), index=pd.to_datetime(self.__timestamp))
        self.__velocities = velocities
        if self.__velocities is not None:
            self.__single_velocities = [np.sqrt(v['Vx']**2 + v['Vy']**2) for v in self.__velocities]
        else:
            self.__single_velocities = None
        self.__offsets = self.__compute_offset(self.__points)
        self.__idx = 0
        self.starting_time = self.__timestamp[0]
        self.ending_time = self.__timestamp[-1]
        if flight_identity is not None:
            self.flight_identity = flight_identity
        else:
            self.flight_identity = "%s-%s" % (self.flight_id, self.starting_time.strftime("%m/%d/%Y, %H:%M:%S"))

    def __compute_offset(self, points):
        dists = [0]
        for p_id in range(1, len(points)):
            previous_p = points[p_id -1]
            p = points[p_id]
            dist = p.dist_to(previous_p)
            dists.append(dist)

        cumsum_dists = np.cumsum(dists)
        total_dist = np.sum(dists)
        offset = [d/total_dist for d in cumsum_dists]
        assert len(points) == len(offset)
        return offset

    def update_starting_time(self, new_starting_time):
        delta_time = self.starting_time - new_starting_time
        self.__timestamp = [d - delta_time for d in self.__timestamp]
        self.__timeindexes = pd.Series(data=list(range(len(self.__timestamp))), index=pd.to_datetime(self.__timestamp))

    def point_timeindexes(self, time_begin, time_end):
        idxs = self.__timeindexes[time_begin:time_end].tolist()
        return [self.__points[idx] for idx in idxs]

    def set_single_velocity(self, single_velocities):
        if single_velocities is not None:
            assert len(single_velocities) == len(self.__points)
        self.__single_velocities = single_velocities

    def reselect(self, idxs):
        self.__points = [self.__points[idx] for idx in idxs]
        self.__timestamp = [self.__timestamp[idx] for idx in idxs]
        if self.__velocities is not None:
            self.__velocities = [self.__velocities[idx] for idx in idxs]
        else:
            self.__velocities = None

        if self.__single_velocities is not None:
            self.__single_velocities = [self.__single_velocities[idx] for idx in idxs]
        else:
            self.__single_velocities = None

        self.__offsets = [self.__offsets[idx] for idx in idxs]
        self.__timeindexes = pd.Series(data=list(range(len(self.__timestamp))), index=pd.to_datetime(self.__timestamp))
        self.__idx = 0
        self.starting_time = self.__timestamp[0]
        self.ending_time = self.__timestamp[-1]

    def get_index_by_timeindexes(self, time_begin, time_end):
        idxs = self.__timeindexes[time_begin:time_end].tolist()
        return idxs

    def reselect_by_timeindexes(self, time_begin, time_end):
        idxs = self.__timeindexes[time_begin:time_end].tolist()
        if len(idxs) == 0:
            self.reselect([])
        else:
            self.reselect(idxs)

    @classmethod
    def from_dataframe(cls, graph: AirportGraph, df: pd.DataFrame):
        # assert df['ID'].nunique() == 1
        # assert (df['TAC'].nunique() == 1) or (df['TAC'].nunique() == 0)

        flight_id = df['ID'].values[0]
        aircraft_type = df['WTC'].values[0]
        gate = df['AST'].values[0]
        # runway = df['runway'].values[0]

        points = []
        timestamps = []
        velocities = []
        for _, row in df.iterrows():
            points.append(GeometricPoint(row['Lat'], row['Lon']))
            timestamps.append(row['datetime'])
            velocities.append({'Vx': row['Vx'], 'Vy': row['Vy']})

        runway_start = graph.check_runway(points[0])
        runway_end = graph.check_runway(points[-1])

        if runway_start is not None:
            is_arrival = True
            runway = runway_start
        else:
            is_arrival = False
            runway = runway_end

        return Flight(flight_id, aircraft_type, gate, is_arrival, runway, points, timestamps, velocities)


    def get_sampling(self, distance_diff = 5):
        p_dist_diff = []
        p_lat = []
        p_lng = []
        p_velocity = []
        p_time_diff = []
        for idx, (point, velocity, timestamp) in enumerate(zip(self.__points, self.__single_velocities, self.__timestamp)):
            if idx == 0:
                dist_diff = 0
                time_diff = 0
            else:
                prev_point = self.__points[idx-1]
                prev_timestamp = self.__timestamp[idx-1]
                dist_diff = prev_point.dist_to(point)
                time_diff = (timestamp - prev_timestamp).seconds

            lat, lng = tuple(point.get_coordinate())
            p_lat.append(lat)
            p_lng.append(lng)
            p_velocity.append(velocity)
            p_dist_diff.append(dist_diff)
            p_time_diff.append(time_diff)

        p_cum_sum_dist = np.cumsum(p_dist_diff)
        p_cum_sum_time = np.cumsum(p_time_diff)

        # new_cum_sum_dist = np.cumsum([distance_diff]*len(p_dist_diff))
        num_point = int((max(p_cum_sum_dist) - min(p_cum_sum_dist))/distance_diff) + 1
        new_cum_sum_dist = np.linspace(min(p_cum_sum_dist), max(p_cum_sum_dist), num_point)

        inteporlate_p_lat = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_lat)
        inteporlate_p_lng = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_lng)
        inteporlate_p_velocity = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_velocity)
        inteporlate_p_cum_sum_time = np.interp(new_cum_sum_dist, p_cum_sum_dist, p_cum_sum_time)
        l_points = [GeometricPoint(lat, lng) for lat, lng in zip(inteporlate_p_lat, inteporlate_p_lng)]

        return l_points, inteporlate_p_velocity, inteporlate_p_cum_sum_time

    def __to_viz_data(self, num_point_interpolate=None):
        viz_data = {'nodes': []}
        p_lat = []
        p_lng = []
        p_velocity = []
        p_time_points = self.get_timestamp(type_time='timepoint')
        interp_points = []

        for point, v in zip(self.__points, self.__single_velocities):
            lat, lng = tuple(point.get_coordinate())
            p_lat.append(lat)
            p_lng.append(lng)
            # v = np.sqrt(velocity['Vx']**2 + velocity['Vy']**2)
            p_velocity.append(v)

        if num_point_interpolate is not None:
            time_intervals = np.linspace(min(p_time_points), max(p_time_points), num_point_interpolate)
            p_lat = np.interp(time_intervals, p_time_points, p_lat)
            p_lng = np.interp(time_intervals, p_time_points, p_lng)
            p_velocity = np.interp(time_intervals, p_time_points, p_velocity)


        for lat,lng in zip(p_lat, p_lng):
            interp_points.append(GeometricPoint(lat=lat, lng=lng))

        l_dist = [0]
        for p1, p2 in zip(interp_points, interp_points[1:]):
            l_dist.append(p2.dist_to(p1))

        idx = 1
        total_dist = sum(l_dist)
        for lat, lng, velocity in zip(p_lat, p_lng, p_velocity):
            node_data = {'location': {}}
            node_data['location']['lat'] = lat
            node_data['location']['lng'] = lng
            node_data['velocity'] = velocity
            node_data['offset'] = sum(l_dist[:idx]) / total_dist
            idx += 1

            viz_data['nodes'].append(node_data)


        viz_data['distance'] = total_dist
        viz_data['flight_info'] = {'flight_id': self.flight_id,
                                   'flight_identity': self.flight_identity,
                                   'type': self.aircraft_type,
                                   'gate': self.gate,
                                   'runway': self.runway,
                                   'start_time': self.starting_time.strftime("%H:%M:%S"),
                                   'is_arrival': self.is_arrival}

        return viz_data

    def to_dict(self, type_data=None, num_point_interpolate=None):
        if type_data == "viz":
            return self.__to_viz_data(num_point_interpolate)

        data = {}
        data['flight_id'] = self.flight_id
        data['aircraft_type'] = self.aircraft_type
        data['gate'] = self.gate
        data['runway'] = self.runway
        data['is_arrival'] = self.is_arrival
        data['points'] = [{'lat':lat, 'lng':lng} for lat, lng in [tuple(point.get_coordinate()) for point in self.__points]]
        data['timestamps'] = [time.strftime(TIME_FORMATING) for time in self.__timestamp]
        if self.__velocities is not None:
            data['velocity'] = self.__velocities

        if self.__single_velocities is not None:
            data['single_velocities'] = self.__single_velocities
        data['flight_identity'] = self.flight_identity
        return data

    @classmethod
    def load_from_dict(cls, data):
        points = [GeometricPoint(p['lat'], p['lng']) for p in data['points']]
        timestamps = [datetime.strptime(s, TIME_FORMATING) for s in data['timestamps']]
        flight =  Flight(data['flight_id'], data['aircraft_type'], data['gate'], data['is_arrival'], data['runway'],
                      points, timestamps, data.get('velocity', None), data.get('flight_identity', None))

        if 'single_velocities' in data:
            flight.set_single_velocity(data['single_velocities'])

        return flight


    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx < len(self.__points):
            result = self.__points[self.__idx]
            self.__idx +=1
            return result
        else:
            raise StopIteration()

    def __len__(self):
        return len(self.__points)

    def __getitem__(self, idx):
        return self.__points[idx]

    def __get_timestamp(self, idx, type_time):
        if type_time == 'datetime':
            return self.__timestamp[idx]
        elif type_time == 'total_seconds':
            return (self.__timestamp[idx] - datetime(1970,1,1)).total_seconds()
        elif type_time == 'timepoint':
            return (self.__timestamp[idx] - self.__timestamp[0]).total_seconds()

    # def remove_data_point(self, idx):
    #     del self.__points[idx]
    #     del self.__timestamp[idx]
    #     del self.__velocities[idx]

    def get_timestamp(self, point_idx=None, type_time='datetime'):
        if point_idx is None:
            return [self.__get_timestamp(idx, type_time) for idx in range(len(self.__timestamp))]
        else:
            return self.__get_timestamp(point_idx, type_time)

    def get_velocity(self, point_dx=None, type_velocity='single'):
        if type_velocity == 'single':
            if point_dx is None:
                return self.__single_velocities
            else:
                return self.__single_velocities[point_dx]
        else:
            if point_dx is None:
                return self.__velocities
            else:
                return self.__velocities[point_dx]

    def get_offset(self, point_idx=None):
        if point_idx is None:
            return self.__offsets
        else:
            return self.__offsets[point_idx]

    def plot(self, **kwargs):
        coordinates = [p.get_coordinate() for p in self.__points]
        return plot_coordinates(coordinates, **kwargs)

    def plot_folium(self, folium_map=None, point_radius=2, point_weight=1, tiles='cartodbpositron', zoom=1):
        if folium_map is None:
            lat, lng = self.__getitem__(0).get_coordinate()
            folium_map = folium.Map(location=(lat, lng), zoom_start=zoom, tiles=tiles)

        for point in self:
            lat, lng = tuple(point.get_coordinate())
            folium.CircleMarker(location=[lat, lng],
                                radius=point_radius,
                                weight=point_weight).add_to(folium_map)
        return folium_map

    def compute_dists(self):
        l_dist = []
        for idx in range(1, len(self.__points)):
            p1 = self.__points[idx-1]
            p2 = self.__points[idx]
            dist = p1.dist_to(p2)
            l_dist.append(dist)
        return l_dist

    def plot_on_smopy_map(self, map=None, figure_size=(18, 10), zoom=150):
        if map is None:
            map = smopy.Map(CHANGI_BOUNDING_BOX, z=zoom)

        ax = map.show_mpl(figsize=figure_size)
        l_lat = []
        l_lng = []
        for point in self.__points:
            lat, lng = point.get_coordinate()
            if hasattr(map, 'to_pixels'):
                lat, lng = map.to_pixels(lat, lng)
            l_lat.append(lat)
            l_lng.append(lng)

        ax.scatter(l_lat, l_lng, c=np.linspace(0, 1, len(self.__points)), cmap='YlGnBu')
        return ax

class Trajectory(object):
    def __init__(self, graph, route):
        self.__node_ids = route
        self.__edge_ids = list(zip(route, route[1:]))
        self.__edges = [graph.get_edge(edge_id) for edge_id in self.__edge_ids]
        self.__dists = [edge.get_dist() for edge in self.__edges]
        self.__seq_points = [graph.get_node(node_id) for node_id in self.__node_ids]
        self.shapely_object = LineString([Point(tuple(reversed(graph.get_node(node_id).get_coordinate()))) for node_id in route])

        self.__edge_velocities = pd.Series(None, index=pd.MultiIndex.from_tuples(set(self.__edge_ids)))
        self.__edge_durations = pd.Series(None, index=pd.MultiIndex.from_tuples(set(self.__edge_ids)))
        self.__edge_dists = pd.Series(None, index=pd.MultiIndex.from_tuples(set(self.__edge_ids)))
        self.__durations = []
        self.starting_time = None
        self.estimate_info = None

        self.real_start_point = None
        self.real_start_dist = None

        for edge_id, _ in self.__edge_dists.iteritems():
            self.__edge_dists[edge_id] = graph.get_edge(edge_id).get_dist()

        if graph.get_runway_nodes() is not None:
            if self.__node_ids[0] in graph.get_runway_nodes():
                self.is_arrival = True
                self.exit_gate = self.__node_ids[-1]
            elif self.__node_ids[-1] in graph.get_runway_nodes():
                self.is_arrival = False
                self.exit_gate = self.__node_ids[-1]
            else:
                self.is_arrival = None
                self.exit_gate = None

        assert len(self.__node_ids) == (len(self.__edge_ids) + 1)


    def project_real_start_point(self, point, graph):
        # The first point of trajectory may not the starting node
        lat, lng = tuple(point.get_coordinate())
        p = Point(lng, lat)
        lng, lat = list(self.shapely_object.interpolate(self.shapely_object.project(p)).coords)[0]
        second_node_id = self.__node_ids[1]
        self.real_start_point = GeometricPoint(lat, lng)
        self.real_start_dist = self.real_start_point.dist_to(graph.get_node(second_node_id))
        self.__edge_dists[(self.__node_ids[0], self.__node_ids[1])] = self.real_start_dist
        self.__dists[0] = self.real_start_dist

    def estimate_info_from_flight(self, flight, graph):
        # TODO: This process should be move out trajectory class
        point, velocity, raw_times = flight.get_sampling(distance_diff=1)

        # Create graph
        raw_graph = nx.MultiDiGraph(name='name', crs={'init': 'epsg:4326'})
        for node_id, p in enumerate(point):
            lat, lng = tuple(p.get_coordinate())
            raw_graph.add_node(node_id, x=lng, y=lat)

        for node_id in range(len(point) - 1):
            length = point[node_id].dist_to(point[node_id + 1])
            raw_graph.add_edge(node_id, node_id + 1, length=length)
            raw_graph.add_edge(node_id + 1, node_id, length=length)

        node_ids = self.get_node_ids()
        l_result = []
        for node_id in node_ids:
            lat, lng = tuple(graph.get_node(node_id).get_coordinate())
            idx, dist = ox.get_nearest_node(raw_graph, (lat, lng), return_dist=True)
            l_result.append({'node_id': node_id, 'idx': idx, 'dist': dist, 'time': raw_times[idx]})

        df_result = pd.DataFrame(l_result)
        df_result = df_result.set_index('node_id')
        df_result = df_result.loc[node_ids]
        self.estimate_info = df_result
        self.starting_time = flight.starting_time

        # Specific for starting point
        self.project_real_start_point(point[0], graph)
        self.real_start_duration = self.estimate_info.iloc[1]['time']
        self.real_start_velocity = self.real_start_dist / self.real_start_duration
        self.__edge_durations[(self.__node_ids[0], self.__node_ids[1])] = self.real_start_duration
        self.__edge_velocities[(self.__node_ids[0], self.__node_ids[1])] = self.real_start_velocity


        # remove starting node
        node_ids = node_ids[1:]
        for pre_node, node in zip(node_ids, node_ids[1:]):
            e_duration = self.estimate_info.loc[node, 'time'] - self.estimate_info .loc[pre_node, 'time']

            self.__edge_durations[(pre_node, node)] = e_duration
            self.__edge_velocities[(pre_node, node)] = self.__edge_dists[(pre_node, node)] / self.__edge_durations[(pre_node, node)]

    def shapely_dist_to_point(self, point):
        # TODO: General dist_to_point function, should not depend on shapely object
        dist = geometric.distance_shapely_linering_to_point(self.shapely_object, point)
        return dist

    def point_at_offset(self, offset):
        dist_to_point = offset * sum(self.get_dist())
        return self.point_at_dist(dist_to_point)

    def point_at_dist(self, dist):
        if dist < 0 and dist > sum(self.get_dist()):
            return None

        for idx, edge in enumerate(self.__edges):
            print("Edge id", self.__edge_ids[idx])
            dist_to_this_edge = sum(self.__dists[:(idx+1)])
            if dist <= dist_to_this_edge:
                dist_to_previous_edge = sum(self.__dists[:idx])
                remain_dist = dist - dist_to_previous_edge
                return edge.point_at_dist(remain_dist)

    @classmethod
    def compute_edge_duration(cls, edges_point_tracked, time_tracked):
        edge_tracked = []
        edge_duration = []
        previous_edge = edges_point_tracked[0]
        previous_time = time_tracked[0]
        for i in range(1, len(edges_point_tracked)):
            current_edge = edges_point_tracked[i]

            if i == len(edges_point_tracked) -1:
                # Last item
                current_time = time_tracked[i]
                duration = (current_time - previous_time).total_seconds()
                edge_tracked.append(previous_edge)
                edge_duration.append(duration)
            else:
                if current_edge == previous_edge:
                    continue
                else:
                    current_time = time_tracked[i]
                    duration = (current_time - previous_time).total_seconds()
                    edge_tracked.append(previous_edge)
                    edge_duration.append(duration)

                    previous_time = current_time
                    previous_edge = current_edge

        return edge_tracked, edge_duration

    # @classmethod
    # def load_from_dict(cls, graph, data):
    #     edge_ids = [(start, end) for start, end in zip(data['starting_node'], data['ending_node'])]
    #     durations = data['duration']
    #     return Trajectory(graph, edge_ids, durations)

    @classmethod
    def load_from_dict(cls, graph, data):
        route = data['route']
        traj = Trajectory(graph, route)
        if 'edge_velocities' in data:
            traj.set_edge_velocities(data['edge_velocities'])
        if "edge_durations" in data:
            traj.set_edge_duration(data['edge_durations'])
        if "edge_dists" in data:
            traj.set_edge_dists(data['edge_dists'])
        return traj

    def set_edge_velocities(self, dict_data):
        self.__edge_data_from_dict(dict_data, self.__edge_velocities)

    def set_edge_duration(self, dict_data):
        self.__edge_data_from_dict(dict_data, self.__edge_durations)

        for edge_id in self.__edge_ids:
            self.__durations.append(self.__edge_durations[edge_id])

    def set_edge_dists(self, dict_data):
        self.__edge_data_from_dict(dict_data, self.__edge_dists)

    def __edge_data_to_dict(self, edge_data):
        """
        Edge data is series with multiple (edge id) index
        Args:
            edge_data (pd.Series): data of edges

        Returns:
            list[dict]: dict that can json-able
        """
        l_data = []
        for (n1, n2), val in edge_data.items():
            l_data.append({'node_start': n1, 'node_end': n2, 'value': val})
        return l_data

    def __edge_data_from_dict(self, dict_data, edge_data=None):
        if edge_data is None:
            data = {}
            for d in dict_data:
                data[(d['node_start'], d['node_end'])] = d['value']
            edge_data = pd.Series(data)
        else:
            for d in dict_data:
                edge_data[(d['node_start'], d['node_end'])] = d['value']
        return edge_data

    def to_dict(self, type_dict=None):
        if type_dict == 'viz':
            coordinates = [p.get_coordinate() for p in self.__seq_points]
            points = [{"location": {"lat": float(p[0]), "lng": float(p[1])}} for p in coordinates]
            total_dist = sum(self.__dists)
            cumsum_dist = np.cumsum(self.__dists)
            edges = [{"offset": int(dist/total_dist*1000), 'timePerEachStep': 50} for dist in cumsum_dist]
            return {"points": points, "edges": edges}

        if any(self.__edge_velocities.isnull()):
            return {'route': self.__node_ids}
        else:
            return {'route': self.__node_ids,
                    'edge_velocities': self.__edge_data_to_dict(self.__edge_velocities),
                    'edge_durations': self.__edge_data_to_dict(self.__edge_durations),
                    'edge_dists': self.__edge_data_to_dict(self.__edge_dists)}

        # starting_node = []
        # ending_node = []
        # for edge_id in self.__edge_ids:
        #     starting_node.append(edge_id[0])
        #     ending_node.append(edge_id[1])
        #
        # return {'starting_node': starting_node,
        #         'ending_node': ending_node,
        #         'duration': self.__durations}

    def __conpute_velocity(self, dists, durations):
        return np.array(dists)/np.array(durations)

    def compute_duration(self, dists, velocities):
        return np.array(dists)/np.array(velocities)

    def get_point_at_dist(self, dist):
        """
        Return the point at dist from starting point
        Args:
            dist:

        Returns:
            GeometricPoint

        """
        pass

    # def estimate_velocities_and_durations(self, graph, points, point_velocities, point_timestamps=None):
    #     subgraph = graph.subgraph(self.__node_ids)
    #     edges = [ox.get_nearest_edge(subgraph, p.get_coordinate()) for p in points]
    #     formated_edegs = []
    #     for e in edges:
    #         if (e[1], e[2]) in self.__edge_ids:
    #             formated_edegs.append((e[1], e[2]))
    #         elif (e[2], e[1]) in self.__edge_ids:
    #             formated_edegs.append((e[2], e[1]))
    #         else:
    #             print('Appeared edge outside trajectory')
    #
    #     edge_velocities = {}
    #     if point_velocities is not None:
    #         for edge_id, velocity in zip(formated_edegs, point_velocities):
    #             edge_velocities[edge_id] = edge_velocities.get(edge_id, []) + [velocity]
    #
    #         for edge_id, velocites in edge_velocities.items():
    #             if len(velocites) > 0:
    #                 self.__edge_velocities[edge_id] = np.mean(velocites)
    #
    #         window_size = 2
    #         while (any(self.__edge_velocities.isnull())):
    #             moving_everage_fill = self.__edge_velocities.fillna(0).rolling(window=window_size,
    #                                                                       center=True, min_periods=1).mean()
    #             moving_everage_fill = moving_everage_fill[list(self.__edge_velocities[self.__edge_velocities.isnull()].index)]
    #             self.__edge_velocities.update(moving_everage_fill)
    #             window_size += 1
    #
    #     self.__edge_durations = self.__edge_dists / self.__edge_velocities

    def __iter__(self):
        self.__edge_idx = 0
        return self

    def __next__(self):
        if self.__edge_idx < len(self.__edges):
            result = self.__edges[self.__edge_idx]
            self.__edge_idx +=1
            return result
        else:
            raise StopIteration()

    def __getitem__(self, edge_idx):
        return self.__edges[edge_idx]

    def __len__(self):
        return len(self.__edges)

    def get_points(self):
        return self.__seq_points

    def get_edge_ids(self, idx=None):
        if idx is not None:
            return self.__edge_ids[idx]
        else:
            return self.__edge_ids

    def get_edge_velocity(self, edge_idx=None):
        if edge_idx is None:
            return self.__edge_velocities[self.get_edge_ids()]
        else:
            return self.__edge_velocities[edge_idx]

    def plot_on_graph(self, graph, **kwargs):
        ox.plot_graph_route(graph, self.__node_ids, **kwargs)

    def plot(self, **kwargs):
        coordinates = [p.get_coordinate() for p in self.__seq_points]
        return plot_coordinates(coordinates, **kwargs)

    def get_edge_duration(self, edge_idx=None):
        if edge_idx is None:
            return self.__edge_durations[self.get_edge_ids()]
        else:
            return self.__edge_durations[edge_idx]

    def get_edge_dists(self, edge_idx=None):
        if edge_idx is None:
            return self.__edge_dists[self.get_edge_ids()]
        else:
            return self.__edge_dists[edge_idx]

    def get_edge_starting_time(self, edge_idx=None):
        if edge_idx is None:
            return [self.get_edge_duration(idx) for idx in range(len(self.__edge_ids))]
        else:
            return sum(self.__durations[:edge_idx])

    def get_edge_ending_time(self, edge_idx=None):
        if edge_idx is None:
            return [self.get_edge_ending_time(idx) for idx in range(len(self.__edge_ids))]
        else:
            return sum(self.__durations[:(edge_idx+1)])

    def get_dist(self, edge_idx=None):
        return self.__dists

    def get_node_ids(self):
        return copy.deepcopy(self.__node_ids)

    def plot_folium(self, G, folium_map=None):
        folium_map = ox.plot_route_folium(G, self.get_node_ids(), route_map=folium_map)
        return folium_map
