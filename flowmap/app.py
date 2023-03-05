import click
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import itertools as itertools

from math import floor
from os import path
from matplotlib import animation
from datetime import datetime
from time import time
from ruth.simulator import Simulation
from multiprocessing.pool import Pool

from flowmapviz.collection_plot import plot_routes, WidthStyle

from input import preprocess_history_records
# from input_alt import preprocess_history_records
from df import get_max_vehicle_count, get_max_time, get_min_time
from ax_settings import Ax_settings


def preprocess_fill_mp_alt(df, g, speed, fps, lock, counts_dict):
    res, _, _ = preprocess_fill_missing_times(df, g, speed, fps, counts_dict, lock)
    return res


def preprocess_counts_mp_alt(records, g, speed, fps, counts_dict):
    return preprocess_add_counts(records, counts_dict)


def preprocess_mp_alt(df, g, speed, fps):
    start = datetime.now()
    cpu_count = mp.cpu_count()
    num_of_processes = cpu_count

    min_vehicle_id = df['vehicle_id'].min()
    max_vehicle_id = df['vehicle_id'].max()
    number_of_ids = max_vehicle_id - min_vehicle_id + 1
    one_df_vehicle_ids = round(number_of_ids / num_of_processes)
    split_vehicle_ids = [min_vehicle_id + x * one_df_vehicle_ids for x in range(num_of_processes)]
    split_vehicle_ids.append(max_vehicle_id + 1)

    lock = mp.Lock()
    manager = mp.Manager()
    counts_dict = manager.dict()
    res_list = [None] * num_of_processes
    fill_mp_partial = partial(preprocess_fill_mp_alt, g=g, speed=speed, fps=fps, lock=lock, counts_dict=counts_dict)
    counts_mp_partial = partial(preprocess_counts_mp_alt, g=g, speed=speed, fps=fps, counts_dict=counts_dict)
    args = [df.loc[(df['vehicle_id'] >= split_vehicle_ids[i]) & (df['vehicle_id'] < split_vehicle_ids[i + 1]), :] for i in range(num_of_processes)]

    with Pool(num_of_processes) as pool:
        df_list = []

        for result in pool.map(fill_mp_partial, args):
            df_list.append(result)

        res_list_conc = list(itertools.chain.from_iterable(res_list))
        res_list_conc = sorted(res_list_conc, key=lambda x: x.timestamp)

        number_of_rows = len(res_list_conc)
        one_df_rows = round(number_of_rows / num_of_processes)
        split_datetimes = [res_list_conc[x * one_df_rows].timestamp for x in range(num_of_processes)]

        args = []
        for i in range(num_of_processes):
            current_list = []
            if i == num_of_processes - 1:
                current_list = res_list_conc[j:]
            else:
                for k in range(j, len(res_list_conc)):
                    if res_list_conc[k].timestamp > split_datetimes[i + 1]:
                        break
                    current_list.append(res_list_conc[k])
                    j += 1
            args.append(current_list)

        res_list = []
        for result in pool.map(fill_mp_partial, args):
            res_list.append(result)

    res_list = list(itertools.chain.from_iterable(res_list))
    return res_list


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
    times_df = sim.history.to_dataframe()  # NOTE: this method has some non-trivial overhead
#     times_df = sim.global_view.to_dataframe()  # NOTE: this method has some non-trivial overhead
    times_df = times_df.loc[(times_df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (times_df['timestamp'] < np.datetime64('2021-06-16T08:00:05.000')),:]

    start = datetime.now()
    times_df = preprocess_mp_alt(times_df, g, speed, fps)
    print("df shape: ", len(times_df))
    print("time of preprocessing: ", datetime.now() - start)

    return

    # ---------------------------------------------------------------------------

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

