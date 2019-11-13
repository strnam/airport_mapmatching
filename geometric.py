import math
import numpy as np
from geopy.distance import geodesic
from pyproj import Proj, transform
import shapely
from shapely.geometry import Polygon, Point, LinearRing, LineString

XY_PROJ =  Proj(init='epsg:3857')
LNG_LAT_PROJ = Proj(init='epsg:4326')
RATIO = 110677.49640590935 # ration between distance lng lat and meter
# This code only use the simple version of function computing distance by approximate the xy coordinates by lat lng coordinate
# Take note that this approximate is only accepted in small area like area of Changi Airport.
# TODO: General method to compute distance from lat, lng coordinate consider geometry operator should be implemented

def distance_shapely_linering_to_point(shapely_obj, point):
    lat, lng = point.get_coordinate()
    p = Point(lng, lat)
    closed_point = shapely_obj.interpolate(shapely_obj.project(p))
    dist = closed_point.distance(p) * RATIO
    return dist


class GeometricPoint(object):
    def __init__(self, lat, lng):
        self.__lat = lat
        self.__lng = lng

    def dist_to(self, other_point, unit="meters"):
        dist = geodesic(self.get_coordinate(type_coordinate='geographic'),
                        other_point.get_coordinate(type_coordinate='geographic'))

        if unit == "meters":
            return dist.meters

    def __eq__(self, other):
        return (self.__lat == other.lat) and (self.__lng == other.lon)

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return "(%s, %s)" % (self.__lat, self.__lng)

    def get_coordinate(self, type_coordinate='geographic'):
        if type_coordinate == 'geographic':
            return np.array([self.__lat, self.__lng])
        elif type_coordinate == 'radians':
            return np.radians([self.__lat, self.__lng])
        elif type_coordinate == 'xy':
            x, y = transform(LNG_LAT_PROJ, XY_PROJ, self.__lng, self.__lat)
            return np.array([x, y])

def compute_angle(a, b, c, d):
    # Compute angle between ab and cd
    ar, br, cr, dr = np.radians(a), np.radians(b), np.radians(c), np.radians(d)
    ab = br - ar
    cd = dr - cr
    try:
        return np.degrees(
            math.acos(np.dot(ab, cd) / (np.linalg.norm(ab) * np.linalg.norm(cd))))
    except:
        return 0

class GeometricEdge(object):
    def __init__(self, starting_point=None, ending_point=None, shapely_obj=None):
        self.starting_point = starting_point
        self.ending_point = ending_point
        self.shapely_starting_point = Point(tuple(reversed(self.starting_point.get_coordinate())))
        self.shapely_ending_point = Point(tuple(reversed(self.ending_point.get_coordinate())))

        self.__dist = self.ending_point.dist_to(self.starting_point)
        self.__vector_in_radians = self.__compute_vector_in_radian(self.starting_point, self.ending_point)
        if shapely_obj is None:
            self.shapely_obj = LineString([self.shapely_starting_point, self.shapely_ending_point])
        else:
            self.shapely_obj = shapely_obj
            self.__dist = self.shapely_obj.length * RATIO

    def __compute_vector_in_radian(self, starting_point, ending_point):
        starting_point_in_radians =  starting_point.get_coordinate(type_coordinate='radians')
        ending_point_in_radians = ending_point.get_coordinate(type_coordinate='radians')
        vector_in_radians = ending_point_in_radians - starting_point_in_radians
        return vector_in_radians

    def point_at_dist(self, dist):
        """
        Calculate a point on the line at a specific distance.
        (TODO) NOTE: In case of airport, it is small area we can consider lat, lng as x, y in plane. This pice of code is wrong for bigger scale.
        Ref: https://math.stackexchange.com/questions/237090/calculate-a-point-on-the-line-at-a-specific-distance
        Args:
            dist (float): distance
        Returns:
            GeometricPoint:
        """
        x1, y1 = tuple(self.starting_point.get_coordinate('xy'))
        x2, y2 = tuple(self.ending_point.get_coordinate('xy'))
        t = dist / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # x = t*x1 + (1-t)*x2
        # y = t*y1 + (1-t)*y2
        x = (1-t)*x1 + t*x2
        y = (1-t)*y1 + t*y2
        lng, lat = transform(XY_PROJ,LNG_LAT_PROJ, x ,y)
        return GeometricPoint(lat, lng)


    def get_vector_in_radians(self):
        return self.__vector_in_radians

    def get_dist(self):
        return self.__dist

    def angle_with(self, other_edge):
        u = self.__vector_in_radians
        v = other_edge.get_vector_in_radians()
        try:
            return np.degrees(
                math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
        except:
            return 0

    def is_regtange_area_containt_point(self, point, rectangle_lenght=5):
        """
        Check whether point place in the rectangle area construct by this edge
        Args:
            point:

        Returns:
            bool
        """
        dist_to_point = self.dist_to_point(point)

        if dist_to_point > rectangle_lenght:
            return False

        angle_HAB = self.angle_with(GeometricEdge(self.starting_point, point))
        if angle_HAB > 90:
            return False

        angle_HBA = GeometricEdge(self.ending_point, self.starting_point).angle_with(GeometricEdge(self.ending_point, point))
        if angle_HBA > 90:
            return False

        return True

    def is_point_beyond(self, point):
        angle_HAB = self.angle_with(GeometricEdge(self.starting_point, point))
        if angle_HAB > 90:
            return False

        angle_HBA = GeometricEdge(self.ending_point, self.starting_point).angle_with(GeometricEdge(self.ending_point, point))
        if angle_HBA > 90:
            return False

        return True

    def get_projected_point(self, point):
        # Ref: https://stackoverflow.com/questions/49061521/projection-of-a-point-to-a-line-segment-python-shapely
        x = np.array(point.get_coordinate())
        u = np.array(self.starting_point.get_coordinate())
        v = np.array(self.ending_point.get_coordinate())

        n = v - u
        n /= np.linalg.norm(n, 2)
        p = u + n * np.dot(x-u, n)
        return GeometricPoint(p[0], p[1]) # Assumtion: lat ,lng in small area similar to x, y

    def dist_to_point(self, point):
        if self.shapely_obj is None:
            # reference: https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
            # p1 = self.starting_point.get_coordinate()
            # p2 = self.ending_point.get_coordinate()
            # p = point.get_coordinate()
            # return np.linalg.norm(np.cross(p2 - p1, p1 - p))/np.linalg.norm(p2 - p1)
            def compute_dist_to_line(a, b, line_dist):
                p = (a + b + line_dist) / 2
                S = np.sqrt(p * (p - a) * (p - b) * (p - line_dist))
                h = 2 * S / line_dist
                return h
            point_to_p1 = self.starting_point.dist_to(point)
            point_to_p2 = self.ending_point.dist_to(point)
            dist = compute_dist_to_line(point_to_p1, point_to_p2, self.__dist)
            return dist
        else:
            # Dist from point to closet point on the line
            lat, lng = point.get_coordinate()
            p = Point(lng, lat)
            closed_point = self.shapely_obj.interpolate(self.shapely_obj.project(p))
            dist = closed_point.distance(p) * RATIO
            return dist

    def __str__(self):
        return "%s to %s" % (str(self.starting_point), str(self.ending_point))

    def __eq__(self, other):
        return self.__str__() == str(other) and type(self) == type(other)
