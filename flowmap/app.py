import click
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle

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
from itertools import chain
import pathlib

from flowmapviz.plot import plot_routes, WidthStyle

from .input import fill_missing_times
from .ax_settings import Ax_settings

@click.group()
def cli():
    pass


def generate_frame(t_seg_dict, timestamp_from, times_len, g, max_count, width_modif, width_style, title):
    f, ax_map = plt.subplots()
    fig, ax_map = ox.plot_graph(g, ax=ax_map, show=False, node_size=0)
    fig.set_size_inches(30, 24)
    plt.title(title, fontsize=40)
    time_text = plt.figtext(0.5, 0.09, datetime.utcfromtimestamp(timestamp_from//10**3), ha="center", fontsize=25)

    ax_density = ax_map.twinx()
    ax_settings = Ax_settings(ylim=ax_map.get_ylim(), aspect=ax_map.get_aspect())

    frames = []
    for timest in range(timestamp_from, timestamp_from + times_len):
        ax_density.clear()
        ax_settings.apply(ax_density)
        ax_density.axis('off')

        segments = t_seg_dict[timest]
        nodes_from = [seg.node_from.id for seg in segments]
        nodes_to = [seg.node_to.id for seg in segments]
        densities = [seg.counts for seg in segments]
        plot_routes(
            g,
            ax=ax_density,
            nodes_from=nodes_from,
            nodes_to=nodes_to,
            densities=densities,
            max_width_density=max_count,
            width_modifier=width_modif,
            width_style=width_style)

        canvas = fig.canvas
        canvas.draw()

        frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
    plt.close(fig)
    return frames


@cli.command()
@click.argument('simulation_path')
@click.option('--fps', '-f', default=25, help="Set video frames per second.", show_default=True)
@click.option('--save-path', default="", help='Path to the folder for the output video.')
@click.option('--frame-start', default=0, help="Number of frames to skip before plotting.")
@click.option('--frames-len', help="Number of frames to plot")
@click.option('--processed-data','-p', is_flag=True, help="Data is already processed")
@click.option('--save-data','-s', is_flag=True, help="Save processed data")
@click.option('--width-style', type=click.Choice([el.name for el in WidthStyle]), default='BOXED',
              help="Choose style of width plotting")
@click.option('--width-modif', default=10, type=click.IntRange(2, 200, clamp=True), show_default=True,
              help="Adjust width.")
@click.option('--title','-t', default="", help='Set video title')
@click.option('--speed', default=1, help="Speed up the video.", show_default=True)
@click.option('--divide', '-d', default=2, help="Into how many parts will each segment be split.", show_default=True)
def generate_animation(simulation_path, fps, save_path, frame_start, frames_len, processed_data, save_data, width_style, width_modif, title, speed, divide):
    ts = TimerSet()
    with ts.get("data loading"):
        sim = Simulation.load(simulation_path)
        g = sim.routing_map.network
        sim_history = sim.history.to_dataframe()

    with ts.get("data preprocessing"):
        t_segments = fill_missing_times(sim_history, g, speed, fps, divide)

        # TODO: fix saving data
        if save_data:
            times_df.to_csv('data.csv')

    with ts.get("preparing for animation"):
        timestamps = [seg.timestamp for seg in t_segments]
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)

        max_count = max([max(seg.counts) for seg in t_segments])

        timestamp_from = min_timestamp + frame_start
        times_len = max_timestamp - timestamp_from
        times_len = min(int(frames_len), times_len) if frames_len else times_len

        width_style = WidthStyle[width_style]

    with ts.get("generate frames"):
        compute_frame_partial = partial(generate_frame,
                             g=g,
                             max_count=max_count,
                             width_modif=width_modif,
                             width_style=width_style,
                             title=title)

#         manager = Manager()
#         glob_dict = manager.dict()
#
#         for seg in t_segments:
#             if seg.timestamp not in glob_dict.keys():
#                 glob_dict[seg.timestamp] = [seg]
#             else:
#                 l = glob_dict[seg.timestamp]
#                 l.append(seg)
#                 glob_dict[seg.timestamp] = l

        dicts = [defaultdict(list)] * cpu_count()
        part_range = int(times_len // (cpu_count() + 1)) + 1
        t_segments = sorted(t_segments, key=lambda x: x.timestamp)
        j = 0
        timestamps_from = [timestamp_from]
        times_lens = []
        for _, seg in enumerate(t_segments):
            if seg.timestamp > timestamp_from + (j + 1) * part_range and j < cpu_count() - 1:
                times_lens.append(seg.timestamp - timestamps_from[j])
                j += 1
                timestamps_from.append(seg.timestamp)
            dicts[j][seg.timestamp].append(seg)
        times_lens.append(t_segments[-1].timestamp - timestamps_from[cpu_count() - 1] + 1)

        with Pool() as pool:
            frames = pool.starmap(compute_frame_partial, list(zip(dicts, timestamps_from, times_lens)))
        frames = list(chain.from_iterable(frames))

    with ts.get("create animation"):
        plt.axis('off')
        anim = animation.FuncAnimation(plt.gcf(), lambda i: plt.imshow(frames[i]), interval=75, frames=floor(times_len), repeat=False)

        timestamp = round(time() * 1000)
        anim.save(path.join(save_path, str(timestamp) + "-rt.mp4"), writer="ffmpeg", fps=fps)

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


def main():
    cli()


if __name__ == '__main__':
    main()
