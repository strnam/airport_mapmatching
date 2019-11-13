import warnings
import osmnx as ox
import networkx as nx
import pandas as pd
from collections import OrderedDict
from shapely.geometry import LineString
from geometric import GeometricPoint, GeometricEdge
import geometric
import folium
from folium.features import DivIcon


CHANGI_RUNWAYS = {
    "runway1": {
        "node_start": {
            "lat": 1.348503,
            "lng": 103.977240
        },
        "node_end": {
            "lat": 1.383222,
            "lng": 103.991906
        },
        "nodes": [ 1229, 1239,1242,1316, 1484, 1429, 135, 13, 1587, 1585]
    },
    "runway2": {
        "node_start": {
            "lat": 1.328285,
            "lng": 103.984761
        },
        "node_end": {
            "lat": 1.362511,
            "lng": 103.999203
        },
        "nodes": [ 614, 651, 650, 679, 764, 787, 922, 977, 960, 966, 972, 954, 463,
        ]
    }
}

CHANGI_RUNWAYS_2 = {
    "runway1": {
        "node_start": {
            "lat": 1.348503,
            "lng": 103.977240
        },
        "node_end": {
            "lat": 1.383222,
            "lng": 103.991906
        },
        "nodes": [199, 268, 273, 182, 526, 9, 394, 12, 404, 16, 186, 21, 553, 423, 549, 389, 401]
    },
    "runway2": {
        "node_start": {
            "lat": 1.328285,
            "lng": 103.984761
        },
        "node_end": {
            "lat": 1.362511,
            "lng": 103.999203
        },
        "nodes": [ 1, 148, 393, 555, 144, 128, 123, 98, 91, 534, 531, 533, 208
        ]
    }
}

CHANGI_RUNWAYS_3 = {
    "runway1": {
        "node_start": {
            "lat": 1.348503,
            "lng": 103.977240
        },
        "node_end": {
            "lat": 1.383222,
            "lng": 103.991906
        },
        "nodes": [1580, 2768, 13, 135, 1429, 1484, 1316, 2767, 1229]
    },
    "runway2": {
        "node_start": {
            "lat": 1.328285,
            "lng": 103.984761
        },
        "node_end": {
            "lat": 1.362511,
            "lng": 103.999203
        },
        "nodes": [461, 2764, 2765, 2766, 922, 787, 764, 679, 650, 614]

    }
}

DEPARTURE_RUNWAY_MAP = {
    "runway1_north2south": [1580, 2768, 13, 135, 1429],
    "runway1_south2north": [1484, 1316, 2767, 1229],
    "runway2_north2south": [614, 650, 679, 764],
    "runway2_south2north": [787, 922, 2766, 2765, 2764, 461]
}

CHANGI_RUNWAY_MAP = {
    "runway1_north": [12, 404, 16, 186, 21, 187, 479, 423, 385, 387, 389, 401, 3],
    "runway1_south": [2, 199, 268, 273, 182, 496, 395, 9, 394],
    "runway2_north": [123, 128, 144, 151, 392, 393, 148, 1],
    "runway2_south": [0, 208, 76, 75, 82, 80, 88, 86, 91, 98]
}

# Taxi way nodes: specific the direction that aircraft follow that must be in this taxiway.
CHANGI_TAXIWAYS = {
    'EP': [66, 122, 81, 77, 87, 424, 441, 83, 209, 211, 94, 95, 93, 96, 219, 204, 224, 225, 193, 100, 112, 137, 113,
           127, 126, 125, 429, 428, 136, 327, 131, 130, 426, 65, 291, 292, 297, 152, 172, 170, 138, 304, 306, 314, 321, 324],
    'NC2': [68, 67, 160, 161, 334, 338, 340, 341, 350, 353, 354, 359, 58, 57],
    'NC1': [368, 367, 364, 362, 360, 357, 352, 349, 343, 169],
    'NC3': [309, 311, 312, 71, 70, 158, 322, 484, 323, 325, 326, 330, 331, 333, 503, 335, 344, 345, 53, 52],
    'A6': [146, 159, 157, 145, 153, 390, 141, 140, 293, 288, 287],
    'A7': [179, 175, 177, 174, 149, 155, 156, 72, 316, 69, 329],
    'A1': [147, 154, 517, 302],
    'E1': [301, 393],
    'A2': [150, 173],
    'E2': [171, 391],
    'A3': [142, 178, 176, 296],
    'E3': [152, 143, 144],
    'A4': [139, 295, 180, 63, 64],
    'B2': [286, 425, 281, 280, 135],
    'B1': [425, 282, 283, 284, 426],
    'B3': [281, 277, 125],
    'E4': [129, 133, 128],
    'E5': [125, 124, 123],
    'C2': [105, 271, 265, 104],
    'C1': [115, 261, 104, 121, 114],
    'C3': [109, 119, 110, 111],
    'C7': [103, 120, 118, 102, 197, 229, 227, 207, 220, 435],
    'C6': [196, 228, 195, 234, 235, 202, 192, 108, 117, 259, 116, 272, 105, 106],
    'L6': [436, 216, 432],
    'L10': [409, 410, 430, 416, 417],
    'L9': [412, 420, 418, 411],
    'L8': [212, 419, 415, 210],
    'L7': [215, 214, 431, 213, 92],
    'L4': [438, 433, 434, 437],
    'SC2': [223, 205, 226, 230, 195, 232, 236, 454, 237, 238, 200, 240, 460, 463],
    'SC1': [189, 250, 248, 203, 246, 245, 255, 243, 191, 202, 107, 233, 197, 231, 101, 193],
    'S1': [443, 475, 440, 444, 445, 467, 469, 466, 465, 446],
    'S3': [472, 471],
    'S2': [452, 458],
    'S4': [457, 456, 458],
    'W10': [452, 194, 201, 198],
    'W9': [188, 252],
    'W8': [270, 267],
    'V8': [265, 262],
    'WA': [449, 448, 200, 242, 224, 203, 251, 254, 257, 26, 263, 266, 274, 276, 31,
           33, 279, 285, 37, 38, 298, 42, 44, 48, 50, 319, 332, 54, 56, 61, 59, 27],
    'WP': [493, 384, 4, 382, 381, 25, 5, 24, 376, 19, 373, 371, 62, 365, 20, 505, 60, 55,
           13, 51, 15, 49, 45, 43, 39, 11, 3, 486, 34, 183, 32, 477, 28, 269, 6, 260, 255, 189, 461, 462, 463]










}




class GraphEdge(GeometricEdge):
    def __init__(self, starting_point, ending_point, starting_node_id, ending_node_id, edge_data):
        super().__init__(starting_point, ending_point)
        self.starting_node_id = starting_node_id
        self.ending_node_id = ending_node_id
        self.__starting_time = edge_data['starting_time']
        self.__ending_time = edge_data['ending_time']
        self.__velocity = edge_data['velocity']
        self.__duration = (self.__ending_time - self.__starting_time).total_seconds()

    def get_velocity(self):
        return self.__velocity

    def get_duration(self):
        return self.__duration



class AirportGraph(object):
    def __init__(self, networkx_graph):
        self.__base_graph = networkx_graph
        self.__all_nodes = OrderedDict()
        self.__all_edges = OrderedDict()

        for node_id, node_data in self.__base_graph.node(data=True):
            self.__all_nodes[node_id] = GeometricPoint(node_data['y'], node_data['x'])

        for starting_node_id, ending_node_id, edge_data in self.__base_graph.edges(data=True):
            if 'geometry' in edge_data:
                geometric_edge = GeometricEdge(starting_point=self.__all_nodes[starting_node_id],
                                               ending_point=self.__all_nodes[ending_node_id], shapely_obj=edge_data['geometry'])
            else:
                geometric_edge = GeometricEdge(self.__all_nodes[starting_node_id], self.__all_nodes[ending_node_id])

            self.__all_edges[(starting_node_id, ending_node_id)] = geometric_edge

        self.__runways = None

    def set_runway(self, runways: dict):
        self.__runways = runways


    def shortest_path(self, node_start, node_end, **kwargs):
        return nx.shortest_path(self.__base_graph, node_start, node_end, **kwargs)


    def get_runway_nodes(self, runway_name=None):
        if self.__runways is None:
            return None

        if runway_name is not None:
            runway_nodes = self.__runways[runway_name]["nodes"]
        else:
            runway_nodes = []
            for name, runway in self.__runways.items():
                runway_nodes += runway["nodes"]

        return runway_nodes

    def add_node(self, node_id=None, **kwargs):
        if node_id is not None:
            self.__base_graph.add_node(node_id, **kwargs)
        else:
            node_id = max(list(self.__all_nodes.keys())) + 1
            self.__base_graph.add_node(node_id, **kwargs)

        # print('ADD NODE %s' % node_id)
        self.__all_nodes[node_id] = GeometricPoint(kwargs['y'], kwargs['x'])

    # def add_node_from_point(self, node_id, point, **kwargs):
    #     lat, lng = tuple(point.get_coordinate())
    #     kwargs['y'] = lat
    #     kwargs['x'] = lng
    #
    #     if node_id is not None:
    #         self.__base_graph.add_node(node_id, **kwargs)
    #     else:
    #         node_id = max(list(self.__all_nodes.keys())) + 1
    #         self.__base_graph.add_node(node_id, **kwargs)
    #     self.__all_nodes[node_id] = point

    def add_midpoint(self, n1, n2):
        p1 = self.get_node(n1)
        p2 = self.get_node(n2)
        lat_n1, lng_n1 = p1.get_coordinate()
        lat_n2, lng_n2 = p2.get_coordinate()
        lat_new_n = (lat_n1 + lat_n2) / 2
        lng_new_n = (lng_n1 + lng_n2) / 2
        new_node_id = max(list(self.__all_nodes.keys())) + 1
        self.add_node(node_id=new_node_id, x=lng_new_n, y=lat_new_n, osmid=None)

    def add_edge(self, starting_node_id, ending_node_id, **edge_data):
        geometric_edge = GeometricEdge(self.__all_nodes[starting_node_id], self.__all_nodes[ending_node_id])
        self.__all_edges[(starting_node_id, ending_node_id)] = geometric_edge
        if "length" not in edge_data:
            edge_data["length"] = geometric_edge.get_dist()
        self.__base_graph.add_edge(starting_node_id, ending_node_id, **edge_data)

    # def merge_two_node(self, from_node_id, to_node_id):

    def remove_edge(self, n_start, n_end):
        del self.__all_edges[(n_start, n_end)]
        del self.__all_edges[(n_end, n_start)]

        self.__base_graph.remove_edge(n_start,n_end)
        self.__base_graph.remove_edge(n_end, n_start)

    def remove_node(self, node_id):
        # print("REMOVE NODE %s" % node_id)
        self.__base_graph.remove_node(node_id)

        del self.__all_nodes[node_id]

        l_edge_to_delete = []
        for edge_id in self.__all_edges.keys():
            if node_id in edge_id:
                l_edge_to_delete.append(edge_id)

        for edge_id in l_edge_to_delete:
            # print('REMOVE EDGE %s' % str(edge_id))
            del self.__all_edges[edge_id]

        # for node_id, node_data in self.__base_graph.node(data=True):
        #     self.__all_nodes[node_id] = GeometricPoint(node_data['y'], node_data['x'])
        #
        # for starting_node_id, ending_node_id, edge_data in self.__base_graph.edges(data=True):
        #     geometric_edge = GeometricEdge(self.__all_nodes[starting_node_id], self.__all_nodes[ending_node_id])
        #     self.__all_edges[(starting_node_id, ending_node_id)] = geometric_edge

    def merge_nodes(self, from_node, to_node):
        nodes_connected_to_from_node = list(set(self.get_connected_nodes(from_node)) - {to_node})
        for n in nodes_connected_to_from_node:
            if (n, to_node) not in self.__all_edges:
                self.add_edge(n, to_node)
            if (to_node, n) not in self.__all_edges:
                self.add_edge(to_node, n)
        self.remove_node(from_node)

    def remove_crossing_points(self, n1, n2, n3, n4):
        """
        Remove crossroads points (4-points case)
        n1 opposite n3
        n2 opposite n4
        https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
        Args:
            n1:
            n2:
            n3:
            n4:

        Returns:
        """
        p1 = self.get_node(n1)
        p2 = self.get_node(n2)
        p3 = self.get_node(n3)
        p4 = self.get_node(n4)
        line1 = LineString([tuple(p1.get_coordinate()), tuple(p3.get_coordinate())]) # A = (X, Y) , B = (X, Y)
        line2 = LineString([tuple(p2.get_coordinate()), tuple(p4.get_coordinate())])
        int_pt = line1.intersection(line2)
        new_node_id = max(list(self.__all_nodes.keys())) + 1
        self.add_node(node_id=new_node_id, x=int_pt.y, y=int_pt.x, osmid=None)

        connect_node_to_n1 = list(set(self.get_connected_nodes(n1)) - {n1, n2, n3, n4})[0]
        connect_node_to_n2 = list(set(self.get_connected_nodes(n2)) - {n1, n2, n3, n4})[0]
        connect_node_to_n3 = list(set(self.get_connected_nodes(n3)) - {n1, n2, n3, n4})[0]
        connect_node_to_n4 = list(set(self.get_connected_nodes(n4)) - {n1, n2, n3, n4})[0]

        self.add_edge(connect_node_to_n1, new_node_id)
        self.add_edge(new_node_id, connect_node_to_n1)
        self.add_edge(connect_node_to_n2, new_node_id)
        self.add_edge(new_node_id, connect_node_to_n2)
        self.add_edge(connect_node_to_n3, new_node_id)
        self.add_edge(new_node_id, connect_node_to_n3)
        self.add_edge(connect_node_to_n4, new_node_id)
        self.add_edge(new_node_id, connect_node_to_n4)

        # self.__base_graph.remove_node(n1)
        # self.__base_graph.remove_node(n2)
        # self.__base_graph.remove_node(n3)
        # self.__base_graph.remove_node(n4)

        self.remove_node(n1)
        self.remove_node(n2)
        self.remove_node(n3)
        self.remove_node(n4)

    def remove_triangle_crossing(self, edge_id, n_id):
        e_n1, e_n2 = edge_id
        p1 = self.get_node(e_n1)
        p2 = self.get_node(e_n2)
        lat_n1, lng_n1 = p1.get_coordinate()
        lat_n2, lng_n2 = p2.get_coordinate()
        lat_new_n = (lat_n1 + lat_n2) / 2
        lng_new_n = (lng_n1 + lng_n2) / 2
        new_node_id = max(list(self.__all_nodes.keys())) + 1
        self.add_node(node_id=new_node_id, x=lng_new_n, y=lat_new_n, osmid=None)

        l_connected_n_id = list(set(self.get_connected_nodes(n_id)) - {e_n1, e_n2})
        l_connected_e_n1 = list(set(self.get_connected_nodes(e_n1)) - {n_id, e_n2})
        l_connected_e_n2 = list(set(self.get_connected_nodes(e_n2)) - {n_id, e_n1})

        for connected_n_id in l_connected_n_id:
            self.add_edge(connected_n_id, new_node_id)
            self.add_edge(new_node_id, connected_n_id)

        for connected_e_n1 in l_connected_e_n1:
            self.add_edge(connected_e_n1, new_node_id)
            self.add_edge(new_node_id, connected_e_n1)

        for connected_e_n2 in l_connected_e_n2:
            self.add_edge(connected_e_n2, new_node_id)
            self.add_edge(new_node_id, connected_e_n2)


        # self.__base_graph.remove_node(e_n1)
        # self.__base_graph.remove_node(e_n2)
        # self.__base_graph.remove_node(n_id)

        self.remove_node(e_n1)
        self.remove_node(e_n2)
        self.remove_node(n_id)

    def get_base_graph(self):
        return self.__base_graph

    def subgraph(self, node_ids):
        self.__base_graph = self.__base_graph.subgraph(node_ids)

    def get_list_node_id(self):
        return list(self.__all_nodes.keys())

    def get_list_edge_id(self):
        return list(self.__all_edges.keys())

    def get_edge(self, edge_id):
        return self.__all_edges[edge_id]

    def get_node(self, node_id):
        return self.__all_nodes[node_id]

    def get_connected_nodes(self, node_id):
        # Get all nodes connect to input node
        list_nodes = []
        for edge in self.__all_edges.keys():
            if node_id in edge:
                list_nodes += list(edge)
        return list(set(list_nodes) - set([node_id]))

    def assign_node_id_to_point(self, point, threshold=10):
        node_id, dist = ox.get_nearest_node(self.__base_graph,
                                            point.get_coordinate(type_coordinate='geographic').tolist(),
                                            return_dist=True)
        if dist < threshold:
            return node_id
        else:
            return None

    def plot_routes(self, sequence_node_id, **kwargs):
        return ox.plot_graph_routes(self.__base_graph, sequence_node_id, **kwargs)

    def plot_folium(self, folium_map=None, popup_attribute='name', edge_width=2, anotation=False, **kwargs):
        graph_map = ox.plot_graph_folium(self.__base_graph, graph_map=folium_map,
                                    popup_attribute=popup_attribute, edge_width=edge_width, **kwargs)
        print(anotation)
        if anotation == True:
            for n, n_data in self.__base_graph.nodes(data=True):
                folium.map.Marker(
                    [n_data['y'], n_data['x'] ],
                    icon=DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(0, 0),
                        html='<div style="font-size: 12pt">%s</div>' % n,
                    )
                ).add_to(graph_map)
        return graph_map

    def forced_connected_property(self, node_id_sequence):
        """
        Two continous nodes in sequence must connected to each other
        Args:
            node_id_sequence:

        Returns:

        """
        fixed_node_id_sequence = [node_id_sequence[0]]
        for n_i, n_j in zip(node_id_sequence, node_id_sequence[1:]):
            shortest_from_i_to_j = nx.shortest_path(self.__base_graph, n_i, n_j, weight='length')
            fixed_node_id_sequence += shortest_from_i_to_j[1:]
        return fixed_node_id_sequence

    def check_runway(self, point, threshold=15):
        point_to_runway = {}
        for runway_name, runway in self.__runways.items():
            runway_node_start = GeometricPoint(runway['node_start']['lat'], runway['node_start']['lng'])
            runway_node_end =  GeometricPoint(runway['node_end']['lat'], runway['node_end']['lng'])
            runway_edge = GeometricEdge(runway_node_start, runway_node_end)
            if runway_node_start.dist_to(point) < runway_node_end.dist_to(point):
                runway_name += "_south2north"
            else:
                runway_name += "_north2south"
            point_to_runway[runway_name] = runway_edge.dist_to_point(point)

        min_runway_name = min(point_to_runway, key=point_to_runway.get)
        if point_to_runway[min_runway_name] < threshold:
            return min_runway_name
        else:
            return None

    def cutting_by_runway(self, list_node_id):
        cutting_nodes = []
        if list_node_id[0] in self.get_runway_nodes():
            detected_type = 'arrival'
        else:
            detected_type = 'departure'
        for node_id in list_node_id:
            if node_id in self.get_runway_nodes():
                if detected_type == 'departure':
                    cutting_nodes.append(node_id)
                    break
                else:
                    continue

            cutting_nodes.append(node_id)

        return cutting_nodes

    def _get_nearest_in_list_edges(self, point, list_edges):
        all_nodes = []
        for n1, n2 in list_edges:
            all_nodes += [n1, n2]
        all_nodes = list(set(all_nodes))
        subgraph = self.__base_graph.subgraph(all_nodes)
        line, node_1, node_2 = ox.get_nearest_edge(subgraph, tuple(point.get_coordinate()))

        if (node_1, node_2) in list_edges:
            nearest_edge = (node_1, node_2)
        elif (node_2, node_1) in list_edges:
            nearest_edge  = (node_2, node_1)
        else:
            print(node_1)
            print(node_2)
            raise ValueError('Out of list edges')

        return nearest_edge

    def _abstract_distance_to_edge(self, point, edge):
        p1 = edge.starting_point
        p2 = edge.ending_point

        angle_p1 = GeometricEdge(p1, p2).angle_with(GeometricEdge(p1, point))
        angle_p2 = GeometricEdge(p2, p1).angle_with(GeometricEdge(p2, point))

        if angle_p1 < 90 and angle_p2 < 90:
            flag = -1
        else:
            flag = 1

        dist_to_p1 = p1.dist_to(point)
        dist_to_p2 = p2.dist_to(point)

        return min(dist_to_p1, dist_to_p2) * flag

    def matching_socre(self, list_coordinate, list_nodes):
        pass

    def find_route_v2(self, list_points,  cutting_by_runway=True):
        # list_points = [GeometricPoint(lat, lng) for lat, lng in list_coordinate]
        assigned_node_ids = [self.assign_node_id_to_point(point) for point in list_points]
        node_ids = [int(node_id) for node_id in assigned_node_ids if node_id is not None]
        node_ids = list(OrderedDict.fromkeys(node_ids).keys())
        connected_seq_node_id = self.forced_connected_property(node_ids)

        if cutting_by_runway and self.__runways is not None:
            connected_seq_node_id = self.cutting_by_runway(connected_seq_node_id)

        return connected_seq_node_id


    def find_route(self, list_points,  cutting_by_runway=True, step=4, reduce_ratio=0.9):
        # list_points = [GeometricPoint(lat, lng) for lat, lng in list_coordinate]
        assigned_node_ids = [self.assign_node_id_to_point(point) for point in list_points]
        node_ids = [int(node_id) for node_id in assigned_node_ids if node_id is not None]
        node_ids = list(OrderedDict.fromkeys(node_ids).keys())
        connected_seq_node_id = self.forced_connected_property(node_ids)

        if cutting_by_runway and self.__runways is not None:
            connected_seq_node_id = self.cutting_by_runway(connected_seq_node_id)

        source_node = connected_seq_node_id[0]
        target_node = connected_seq_node_id[-1]

        # related_nodes = []
        # for node in connected_seq_node_id:
        #     related_nodes += self.get_connected_nodes(node)
        #
        # related_nodes = list(set(related_nodes))
        # subgraph = self.__base_graph.subgraph(related_nodes)
        subgraph = self.__base_graph

        # for lat, lng in  list_coordinate:
        for idx in range(0, len(list_points), step):
            p = list_points[idx]
            lat, lng = p.get_coordinate()
            _, n1, n2 = ox.get_nearest_edge(subgraph, (lat, lng))
            subgraph[n1][n2][0]['length'] *= reduce_ratio

        return nx.shortest_path(subgraph, source_node, target_node, weight='length')


    def analyze_edges_sequence(self, original_route, edges):
        seq_edges = [edges[0]]
        for edge_idx in range(1, len(edges)):
            if edges[edge_idx - 1] != edges[edge_idx]:
                seq_edges.append(edges[edge_idx])

        issues = []

        if pd.Series(seq_edges).value_counts().max() > 1:
            issues.append('Edge route contain circle')

        reconstruct_route = [seq_edges[0][0]] + [edge[1] for edge in seq_edges]
        if reconstruct_route != original_route:
            issues.append('Reconstruct route differnt original route')

        return issues


    def matching_points_to_route(self, points, route):
        edge_ids = list(zip(route, route[1:]))
        edges = [self.get_edge(edge_id) for edge_id in edge_ids]
        assigned_edges = []
        list_edge_idx = list(range(len(edges)))
        for i, point in enumerate(points):
            if i > 0:
                list_search_edge_idx = list_edge_idx[(i-1):i] + list_edge_idx[i:] + list_edge_idx[:(i-1)] # save time to search
            else:
                list_search_edge_idx = list_edge_idx

            assigned_edge = None
            for edge_id in list_search_edge_idx:
                edge = edges[edge_id]
                if edge.is_regtange_area_containt_point(point):
                    assigned_edge = list_edge_idx[edge_id]

            assigned_edges.append(assigned_edge)

        return assigned_edges

    def map_matching_from_coordinates_v2(self, list_coordinate, cutting_by_runway=True):
        list_points = [GeometricPoint(lat, lng) for lat, lng in list_coordinate]
        route = self.find_route(list_points, cutting_by_runway)
        edges = self.matching_points_to_route(list_points, route)
        return edges

    def get_nearest_node(self, point, method='haversine', return_dist=True, num_node_return=3):
        """
        Return list of node nearest the specific point
        """
        G = self.__base_graph
        point = tuple(point.get_coordinate())

        if not G or (G.number_of_nodes() == 0):
            raise ValueError('G argument must be not be empty or should contain at least one node')

        # dump graph node coordinates into a pandas dataframe indexed by node id
        # with x and y columns
        coords = [[node, data['x'], data['y']] for node, data in G.nodes(data=True)]
        df = pd.DataFrame(coords, columns=['node', 'x', 'y']).set_index('node')

        # add columns to the dataframe representing the (constant) coordinates of
        # the reference point
        df['reference_y'] = point[0]
        df['reference_x'] = point[1]

        # calculate the distance between each node and the reference point
        if method == 'haversine':
            # calculate distance vector using haversine (ie, for
            # spherical lat-long geometries)
            distances = ox.great_circle_vec(lat1=df['reference_y'],
                                         lng1=df['reference_x'],
                                         lat2=df['y'],
                                         lng2=df['x'])

        elif method == 'euclidean':
            # calculate distance vector using euclidean distances (ie, for projected
            # planar geometries)
            distances = ox.euclidean_dist_vec(y1=df['reference_y'],
                                           x1=df['reference_x'],
                                           y2=df['y'],
                                           x2=df['x'])

        else:
            raise ValueError('method argument must be either "haversine" or "euclidean"')

        # nearest node's ID is the index label of the minimum distance
        l_nearest_nodes = list(distances.sort_values(ascending=True).index[:num_node_return])

        # if caller requested return_dist, return distance between the point and the
        # nearest node as well
        if return_dist:
            return l_nearest_nodes, [distances.loc[nearest_node] for nearest_node in l_nearest_nodes]
        else:
            return l_nearest_nodes

    def get_angle(self, edge1, edge2):
        if edge1[1] == edge2[0]:
            # connected edges
            edge1_coords = list(self.get_edge(edge1).shapely_obj.coords)
            edge2_coords = list(self.get_edge(edge2).shapely_obj.coords)
            assert edge1[-1] == edge2[0]

            lng_a, lat_a = edge1_coords[-2]
            lng_b, lat_b = edge1_coords[-1]
            lng_c, lat_c = edge2_coords[0]
            lng_d, lat_d = edge2_coords[1]
            angle = geometric.compute_angle((lat_b, lng_b), (lat_a, lng_a),
                                            (lat_c, lng_c), (lat_d, lng_d))
        else:
            angle = self.get_edge(edge1).angle_with(self.get_edge(edge2))

        return angle