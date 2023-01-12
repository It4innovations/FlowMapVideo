import osmnx as ox
from shapely import wkt

def get_route_network(map_file):
    return ox.load_graphml(map_file)