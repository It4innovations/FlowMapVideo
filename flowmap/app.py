import click
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from math import floor
from os import path
from matplotlib import animation
from datetime import datetime
from time import time
from collections import defaultdict
from ruth.simulator import Simulation

from flowmapviz.plot import plot_routes, WidthStyle

from .input import fill_missing_times
from .ax_settings import Ax_settings


def animate(g, t_seg_dict, ax, ax_settings, timestamp_from, max_count, width_modif, width_style, time_text_artist, speed):
    def step(i):
        ax.clear()
        ax_settings.apply(ax)
        ax.axis('off')

        timestamp = timestamp_from + i # * speed # * round(1000 / fps)
        # TODO: fix time label
#         if(i % 5*60 == 0):
#             time_text_artist.set_text(datetime.utcfromtimestamp(timestamp//10**3))
        segments = t_seg_dict[timestamp]
        nodes_from = [seg.node_from.id for seg in segments]
        nodes_to = [seg.node_to.id for seg in segments]
        densities = [seg.counts for seg in segments]
        plot_routes(
            g,
            ax=ax,
            nodes_from=nodes_from,
            nodes_to=nodes_to,
            densities=densities,
            max_width_density=max_count,
            width_modifier=width_modif,
            width_style=width_style)
    return step


@click.command()
@click.argument('simulation_path')
@click.option('--fps', '-f', default=25, help="Set video frames per second.", show_default=True)
@click.option('--save-path', default="", help='Path to the folder for the output video.')
@click.option('--frame-start', default=0, help="Number of frames to skip before plotting.")
@click.option('--frames-len', help="Number of frames to plot")
@click.option('--processed-data','-p', is_flag=True, help="Data is already processed")
@click.option('--save-data','-s', is_flag=True, help="Save processed data")
@click.option('--width-style', type=click.Choice([el.name for el in WidthStyle]), default='CALLIGRAPHY',
              help="Choose style of width plotting")
@click.option('--width-modif', default=10, type=click.IntRange(2, 200, clamp=True), show_default=True,
              help="Adjust width.")
@click.option('--title','-t', default="", help='Set video title')
@click.option('--speed', default=1, help="Speed up the video.", show_default=True)
@click.option('--divide', '-d', default=2, help="Into how many parts will each segment be split.", show_default=True)
def main(simulation_path, fps, save_path, frame_start, frames_len, processed_data, save_data, width_style, width_modif, title, speed, divide):

    start = datetime.now()
    sim = Simulation.load(simulation_path)
    g = sim.routing_map.network

    start = datetime.now()
    t_segments = fill_missing_times(sim.history.to_dataframe(), g, speed, fps, divide)
    print("data len: ", len(t_segments))
    print("time of preprocessing: ", datetime.now() - start)

    if save_data:
        times_df.to_csv('data.csv')


    timestamps = [seg.timestamp for seg in t_segments]
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)

    max_count = max([max(seg.counts) for seg in t_segments]) #times_df['count_from'].max()

    timestamp_from = min_timestamp + frame_start
    times_len = max_timestamp - timestamp_from
    times_len = min(int(frames_len), times_len) if frames_len else times_len

    t_seg_dict = defaultdict(list)
    for seg in t_segments:
        t_seg_dict[seg.timestamp].append(seg)

    f, ax_map = plt.subplots()
    fig, ax_map = ox.plot_graph(g, ax=ax_map, show=False, node_size=0)
    fig.set_size_inches(30, 24)

    plt.title(title, fontsize=40)
    time_text = plt.figtext(0.5, 0.09, datetime.utcfromtimestamp(timestamp_from//10**3), ha="center", fontsize=25)

    ax_density = ax_map.twinx()
    ax_map_settings = Ax_settings(ylim=ax_map.get_ylim(), aspect=ax_map.get_aspect())

    # TODO: fix style
    width_style_enum_option = WidthStyle.CALLIGRAPHY
    for el in WidthStyle:
        if el.name == width_style:
            width_style_enum_option = el

    print(floor(times_len/speed))
    anim = animation.FuncAnimation(plt.gcf(),
                                    animate(
                                        g,
                                        t_seg_dict,
                                        ax_settings=ax_map_settings,
                                        ax=ax_density,
                                        timestamp_from=timestamp_from,
                                        max_count = max_count,
                                        width_modif = width_modif,
                                        width_style = width_style_enum_option,
                                        time_text_artist = time_text,
                                        speed = speed
                                        ),
                                    interval=75, frames=floor(times_len), repeat=False)
    timestamp = round(time() * 1000)

    anim_start = datetime.now()
    anim.save(path.join(save_path, str(timestamp) + "-rt.mp4"), writer="ffmpeg", fps=fps)
    print('doba trvani ulozeni animace: ', datetime.now() - anim_start)

    finish = datetime.now()
    print('doba trvani: ', finish - start)


if __name__ == '__main__':
    main()

