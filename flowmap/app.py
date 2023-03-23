import click
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
from datetime import timedelta
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from math import floor
from os import path
from matplotlib import animation
from datetime import datetime
from time import time
from collections import defaultdict
from ruth.simulator import Simulation
from ruth.utils import TimerSet, Timer

from flowmapviz.plot import plot_routes, WidthStyle

from .input import fill_missing_times
from .ax_settings import Ax_settings

@click.group()
def cli():
    pass


def compute_frame(segments, g, ax, max_with_density, with_modifier, with_style):
    pass


manager = Manager()
glob_data = manager.dict(defaultdict(list))

def prepare_step(step, t_seg_dict, timestamp_from):
    glob_data.append(t_seg_dict)

    timestamp = timestamp_from + step # * speed # * round(1000 / fps)
    # TODO: fix time label
#         if(i % 5*60 == 0):
#             time_text_artist.set_text(datetime.utcfromtimestamp(timestamp//10**3))

    segments = t_seg_dict[timestamp]
    nodes_from = [seg.node_from.id for seg in segments]
    nodes_to = [seg.node_to.id for seg in segments]
    densities = [seg.counts for seg in segments]

    return nodes_from, nodes_to, densities


def prepare_steps(t_seg_dict, timestamp_from, n_steps):

    single_step_data = partial(prepare_step,
                               t_seg_dict=t_seg_dict,
                               timestamp_from=timestamp_from)

    # killed by parallel map
    with Pool(processes=cpu_count()) as p:  # TODO: pass pool as a parameter
        return p.map(single_step_data, range(n_steps))
    # return map(single_step_data, range(n_steps))


def comptue_frames(t_seg_dict, g, ax, max_with_density, with_modifier, with_style):
    pass


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


@cli.command()
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
def generate_animation(simulation_path, fps, save_path, frame_start, frames_len, processed_data, save_data, width_style, width_modif, title, speed, divide):

    ts = TimerSet()

    start = datetime.now()

    with ts.get("preprocessing"):
        sim = Simulation.load(simulation_path)
        g = sim.routing_map.network

        t_segments = fill_missing_times(sim.history.to_dataframe(), g, speed, fps, divide)
        print("data len: ", len(t_segments))

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

        print(floor(times_len))

    with ts.get("mpl_anim"):
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

    with ts.get("saving_video"):
        anim.save(path.join(save_path, str(timestamp) + "-rt.mp4"), writer="ffmpeg", fps=fps)

    print('celkova doba trvani: ', datetime.now() - start)

    for k,v in ts.collect().items():
        print(f"{k}: {v} ms")


class TimeUnit(Enum):
    SECONDS = timedelta(seconds=1)
    MINUTES = timedelta(minutes=1)
    HOURS = timedelta(hours=1)

    @staticmethod
    def from_str(name):
        if name == "seconds":
            return TimeUnit.SECONDS
        elif name == "minutes":
            return TimeUnit.MINUTES
        elif name == "hours":
            return TimeUnit.HOURS

        raise Exception(f"Invalid time unit: '{name}'.")


@cli.command()
@click.argument("simulation-path", type=click.Path(exists=True))
@click.option("--time-unit", type=str,
              help="Time unit. Possible values: [seconds|minutes|hours]",
              default="hours")
def get_info(simulation_path, time_unit):

    sim = Simulation.load(simulation_path)

    time_unit = TimeUnit.from_str(time_unit)

    real_time = (sim.history.data[-1][0] - sim.history.data[0][0]) / time_unit.value

    print (f"Real time duration: {real_time} {time_unit.name.lower()}.")


@cli.command()
@click.argument("simulation-path", type=click.Path(exists=True))
def test(simulation_path):

    speed = 60
    fps = 25
    divide = 2
    frame_start = 0

    sim = Simulation.load(simulation_path)
    g = sim.routing_map.network

    t_segments = fill_missing_times(sim.history.to_dataframe(), g, speed, fps, divide)

    timestamps = [seg.timestamp for seg in t_segments]
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)

    max_count = max([max(seg.counts) for seg in t_segments]) #times_df['count_from'].max()

    timestamp_from = min_timestamp + frame_start
    times_len = max_timestamp - timestamp_from
    # times_len = min(int(frames_len), times_len) if frames_len else times_len

    # t_seg_dict = defaultdict(list)
    # for seg in t_segments:
    #     t_seg_dict[seg.timestamp].append(seg)
    for seg in t_segments:
        glob_data[seg.timestamp].append(seg)

    with Timer("prepare_steps") as t:
        single_steps_data = prepare_steps(glob_data, timestamp_from, times_len)
        # for ss in single_steps_data:
        #     print (ss)


    print(f"{t.duration_ms} ms")


def main():
    cli()



if __name__ == '__main__':
    main()
