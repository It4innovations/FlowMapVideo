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
from ruth.simulator import Simulation

from flowmapviz.collection_plot import plot_routes, WidthStyle

from input import preprocess_history_records
# from input_alt import preprocess_history_records
from df import get_max_vehicle_count, get_max_time, get_min_time
from ax_settings import Ax_settings


def animate(g, times, ax, ax_settings, timestamp_from, max_count, width_modif, width_style, time_text_artist, speed):
    def step(i):
        ax.clear()
        ax_settings.apply(ax)
        ax.axis('off')

        timestamp = timestamp_from + i # * speed # * round(1000 / fps)
        # TODO: fix time label
#         if(i % 5*60 == 0):
#             time_text_artist.set_text(datetime.utcfromtimestamp(timestamp//10**3))

        if timestamp in times.index:
            segments = times.loc[timestamp]
            plot_routes(g, segments, ax=ax,
                        max_width_density=max_count,
                        width_modifier=width_modif,
                        width_style=width_style
                        )
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
# TODO: add option for fps

def main(simulation_path, fps, save_path, frame_start, frames_len, processed_data, save_data, width_style, width_modif, title, speed):

    start = datetime.now()
    sim = Simulation.load(simulation_path)
    g = sim.routing_map.network

    start = datetime.now()
    times_df = preprocess_history_records(sim.history.to_dataframe(), g, speed, fps)
    print("data len: ", len(times_df))
    print("time of preprocessing: ", datetime.now() - start)

    if save_data:
        times_df.to_csv('data.csv')

    return

    # ---------------------------------------------------------------------------

    max_count = times_df['count_from'].max()
    max_to = times_df['count_to'].max()
    if max_to > max_count:
        max_count = max_to

    f, ax_map = plt.subplots()
    fig, ax_map = ox.plot_graph(g, ax=ax_map, show=False, node_size=0)
    fig.set_size_inches(30, 24)

    timestamp_from = get_min_time(times_df) + frame_start
    times_len = get_max_time(times_df) - timestamp_from
    times_len = min(int(frames_len), times_len) if frames_len else times_len

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
                                        times_df,
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

