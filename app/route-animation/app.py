import osmnx as ox
import matplotlib.pyplot as plt

from app_io import load_input
from datetime import datetime
from app_base_graph import get_route_network, get_route_network_small

if __name__ == "__main__":
    start = datetime.now()
    g = get_route_network_small()

    times = load_input("../data/gv_325630_records.parquet", g, 11)

    finish = datetime.now()
    print(finish - start)