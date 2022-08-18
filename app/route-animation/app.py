import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import animation

from app_io import load_input
from datetime import datetime
from time import time
from ax_settings import Ax_settings
from app_base_graph import get_route_network, get_route_network_small
from app_collection_plot import plot_routes
from app_df import get_max_vehicle_count, get_max_time, get_min_time


def anim(g, times, ax, ax_settings, timestamp_from):

    def step(i):
        segments = times.loc[timestamp_from + i]

        ax.clear()
        ax_settings.apply(ax)
        ax_density.axis('off')

        plot_routes(g, segments, ax=ax)

    return step


if __name__ == "__main__":
    start = datetime.now()
    g = get_route_network()

    times_df = load_input("../data/gv_202106016_7_10.parquet", g)

    f, ax_map = plt.subplots()
    fig, ax_map = ox.plot_graph(g, ax=ax_map, show=False, node_size=0)
    fig.set_size_inches(30, 24)
    ax_density = ax_map.twinx()
    ax_map_settings = Ax_settings(ylim=ax_map.get_ylim(), aspect=ax_map.get_aspect())

    timestamp_from = get_min_time(times_df)

    anim = animation.FuncAnimation(plt.gcf(), anim(g, times_df, ax_settings=ax_map_settings, ax=ax_density, timestamp_from=timestamp_from),
                                   interval=150, frames=get_max_time(times_df) - timestamp_from, repeat=False)
    timestamp = round(time() * 1000)

    anim.save(str(timestamp) + "-rt.mp4", writer="ffmpeg")

    finish = datetime.now()
    print('doba trvani: ', finish - start)
