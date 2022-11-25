import click
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd

from os import path
from matplotlib import animation
from input import load_input
from datetime import datetime
from time import time
from ax_settings import Ax_settings
from base_graph import get_route_network
from flowmapviz.collection_plot import plot_routes, WidthStyle
from df import get_max_vehicle_count, get_max_time, get_min_time
from math import floor


def animate(g, times, ax, ax_settings, timestamp_from, max_count, width_modif, width_style, time_text_artist, speed):
    def step(i):
        timestamp = timestamp_from + i * speed
        segments = times.loc[timestamp]

        ax.clear()
        ax_settings.apply(ax)
        ax.axis('off')
        if(i % 5*60 == 0):
            time_text_artist.set_text(datetime.utcfromtimestamp(timestamp))
        plot_routes(g, segments, ax=ax,
                    max_width_density=max_count,
                    width_modifier=width_modif,
                    width_style=width_style
                    )
    return step


@click.command()
@click.argument('data-file')
@click.argument('map-file')
@click.option('--save-path', default="", help='Path to the folder for the output video.')
@click.option('--frame-start', default=0, help="Number of frames to skip before plotting.")
@click.option('--frames-len', help="Number of frames to plot")
@click.option('--processed-data','-p', is_flag=True, help="Data is already processed")
@click.option('--save-data','-s', is_flag=True, help="Save processed data")
@click.option('--width-style', type=click.Choice([el.name for el in WidthStyle]), default='EQUIDISTANT',
              help="Choose style of width plotting")
@click.option('--width-modif', default=10, type=click.IntRange(2, 200, clamp=True), show_default=True,
              help="Adjust width.")
@click.option('--title','-t', default="", help='Set video title')
@click.option('--speed', default=1, help="Speed up the video.", show_default=True)

def main(data_file, map_file, save_path, frame_start, frames_len, processed_data, save_data, width_style, width_modif, title, speed):
    start = datetime.now()
    g = get_route_network(map_file)
    if processed_data:
        times_df = pd.read_csv(data_file)
        times_df.set_index('timestamp', inplace=True)
    else:
        times_df = load_input(data_file, g)

    if save_data:
        times_df.to_csv('data.csv')

    finish = datetime.now()
    print('doba trvani: ', finish - start)

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
    time_text = plt.figtext(0.5, 0.09, datetime.utcfromtimestamp(timestamp_from), ha="center", fontsize=25)

    ax_density = ax_map.twinx()
    ax_map_settings = Ax_settings(ylim=ax_map.get_ylim(), aspect=ax_map.get_aspect())

    width_style_enum_option = WidthStyle.EQUIDISTANT
    for el in WidthStyle:
        if el.name == width_style:
            width_style_enum_option = el

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
                                    interval=75, frames=floor(times_len/speed), repeat=False)
    timestamp = round(time() * 1000)
    anim.save(path.join(save_path, str(timestamp) + "-rt.mp4"), writer="ffmpeg")

    finish = datetime.now()
    print('doba trvani: ', finish - start)


if __name__ == '__main__':
    main()

