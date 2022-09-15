import click
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd

from os import path
from matplotlib import animation
from app_io import load_input
from datetime import datetime
from time import time
from ax_settings import Ax_settings
from app_base_graph import get_route_network_old, get_route_network_small
from src.collection_plot import plot_routes
from app_df import get_max_vehicle_count, get_max_time, get_min_time


def animate(g, times, ax, ax_settings, timestamp_from):
    def step(i):
        segments = times.loc[timestamp_from + i]

        ax.clear()
        ax_settings.apply(ax)
        ax.axis('off')

        plot_routes(g, segments, ax=ax)
    return step


@click.command()
@click.argument('data-file')
@click.argument('map-file')
@click.option('--save-path', default="", help='Path to the folder for the output video.')
@click.option('--frame-start', default=0, help="Number of frames to skip before plotting.")
@click.option('--frames-len', help="Number of frames to plot")

def main(data_file, map_file, save_path, frame_start, frames_len):
    start = datetime.now()
    g = get_route_network_old()
    times_df = load_input(data_file, g, 50)
    finish = datetime.now()
    print('doba trvani: ', finish - start)
#     return
    f, ax_map = plt.subplots()
    fig, ax_map = ox.plot_graph(g, ax=ax_map, show=False, node_size=0)
    fig.set_size_inches(30, 24)
    ax_density = ax_map.twinx()
    ax_map_settings = Ax_settings(ylim=ax_map.get_ylim(), aspect=ax_map.get_aspect())

    timestamp_from = get_min_time(times_df) + frame_start
    times_len = get_max_time(times_df) - timestamp_from
    times_len = min(int(frames_len), times_len) if frames_len else times_len

    anim = animation.FuncAnimation(plt.gcf(), animate(g, times_df, ax_settings=ax_map_settings, ax=ax_density, timestamp_from=timestamp_from),
                                   interval=150, frames=times_len, repeat=False)
    timestamp = round(time() * 1000)
    anim.save(path.join(save_path, str(timestamp) + "-rt.mp4"), writer="ffmpeg")

    finish = datetime.now()
    print('doba trvani: ', finish - start)


if __name__ == '__main__':
    main()

