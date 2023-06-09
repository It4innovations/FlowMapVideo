import click
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import logging

from enum import Enum
from datetime import timedelta, datetime
from math import floor
from os import path
from matplotlib import animation
from time import time
from collections import defaultdict
from ruth.simulator import Simulation
from ruth.utils import TimerSet

from flowmapviz.plot import plot_routes, WidthStyle
from flowmapviz.zoom import plot_graph_with_zoom

from .input import fill_missing_times
from .ax_settings import Ax_settings


@click.group()
def cli():
    pass


def animate(
    g,
    timed_seg_dict,
    ax,
    ax_settings,
    timestamp_from,
    max_count,
    width_modif,
    width_style,
    time_text_artist,
    interval
):
    def step(i):
        ax.clear()
        ax_settings.apply(ax)
        ax.axis('off')

        timestamp = timestamp_from + i
        if i % 5 * 60 == 0:
            time_text_artist.set_text(datetime.utcfromtimestamp(timestamp * 1000 * interval // 10**3))
        segments = timed_seg_dict[timestamp]
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
            width_style=width_style
        )

    return step


@cli.command()
@click.argument('simulation_path')
@click.option('--fps', '-f', default=25, help="Set video frames per second.", show_default=True)
@click.option('--save-path', default='', help="Path to the folder for the output video.")
@click.option('--frame-start', default=0, help="Number of frames to skip before plotting.")
@click.option('--frames-len', help="Number of frames to plot")
@click.option('--processed-data', '-p', is_flag=True, help="Data is already processed")
@click.option('--save-data', '-s', is_flag=True, help="Save processed data")
@click.option(
    '--width-style',
    type=click.Choice([el.name for el in WidthStyle]),
    default='BOXED',
    help="Choose style of width plotting"
)
@click.option(
    '--width-modif',
    default=10,
    type=click.IntRange(2, 200, clamp=True),
    show_default=True,
    help="Adjust width."
)
@click.option('--title', '-t', default='', help="Set video title")
@click.option('--speed', default=1, help="Speed up the video.", show_default=True)
@click.option(
    '--divide',
    '-d',
    default=2,
    help="Into how many parts will each segment be split.",
    show_default=True
)
def generate_animation(
    simulation_path,
    fps,
    save_path,
    frame_start,
    frames_len,
    processed_data,
    save_data,
    width_style,
    width_modif,
    title,
    speed,
    divide
):
    mpl.use('Agg')
    ts = TimerSet()
    with ts.get("data loading"):
        logging.info('Loading simulation data...')
        sim = Simulation.load(simulation_path)
        g = sim.routing_map.network
        sim_history = sim.history.to_dataframe()
        logging.info('Simulation data loaded.')

    with ts.get("data preprocessing"):
        logging.info('Preprocessing data...')
        if not processed_data:
            timed_segments = fill_missing_times(sim_history, g, speed, fps, divide)

        if save_data:
            with open('preprocessed_data.pickle', 'wb') as h:
                pickle.dump(timed_segments, h)

        timestamps = [seg.timestamp for seg in timed_segments]
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)

        max_count = max([max(seg.counts) for seg in timed_segments])

        timestamp_from = min_timestamp + frame_start
        num_of_frames = max_timestamp - timestamp_from
        num_of_frames = min(int(frames_len), num_of_frames) if frames_len else num_of_frames

        timed_seg_dict = defaultdict(list)
        for seg in timed_segments:
            timed_seg_dict[seg.timestamp].append(seg)
        logging.info('Data preprocessed.')

    with ts.get("base map preparing"):
        logging.info('Preparing base map...')
        width_style = WidthStyle[width_style]

        fig, ax_map = plt.subplots()
        plot_graph_with_zoom(g, ax_map, secondary_sizes=[1, 0.7, 0.5, 0.3])

        size = fig.get_size_inches()
        new_size = 20
        size[1] = size[1] * new_size / size[0]
        size[0] = new_size
        fig.set_size_inches(size)

        plt.title(title, fontsize=40)
        interval = speed / fps
        time_text = plt.figtext(
            0.5,
            0.09,
            datetime.utcfromtimestamp(timestamp_from * 1000 * interval // 10**3),
            ha='center',
            fontsize=25
        )

        ax_density = ax_map.twinx()
        ax_map_settings = Ax_settings(ylim=ax_map.get_ylim(), aspect=ax_map.get_aspect())
        logging.info('Base map prepared.')

    with ts.get("create animation"):
        logging.info('Creating animation...')
        anim = animation.FuncAnimation(
            plt.gcf(),
            animate(
                g,
                timed_seg_dict,
                ax_settings=ax_map_settings,
                ax=ax_density,
                timestamp_from=timestamp_from,
                max_count=max_count,
                width_modif=width_modif,
                width_style=width_style,
                time_text_artist=time_text,
                interval=interval
            ),
            interval=75,
            frames=floor(num_of_frames),
            repeat=False
        )

        timestamp = round(time() * 1000)
        anim.save(path.join(save_path, str(timestamp) + '-rt.mp4'), writer='ffmpeg', fps=fps)
        logging.info('Animation created.')

    print()
    for k, v in ts.collect().items():
        print(f'{k}: {v} ms')


class TimeUnit(Enum):
    SECONDS = timedelta(seconds=1)
    MINUTES = timedelta(minutes=1)
    HOURS = timedelta(hours=1)

    @staticmethod
    def from_str(name):
        if name == 'seconds':
            return TimeUnit.SECONDS
        elif name == 'minutes':
            return TimeUnit.MINUTES
        elif name == 'hours':
            return TimeUnit.HOURS

        raise Exception(f"Invalid time unit: '{name}'.")


@cli.command()
@click.argument('simulation-path', type=click.Path(exists=True))
@click.option(
    '--time-unit',
    type=str,
    help="Time unit. Possible values: [seconds|minutes|hours]",
    default='hours'
)
@click.option('--speed', default=1, help="Speed up the video.", show_default=True)
def get_info(simulation_path, time_unit, speed):

    sim = Simulation.load(simulation_path)

    time_unit = TimeUnit.from_str(time_unit)
    time_unit_minutes = TimeUnit.from_str('minutes')

    real_time = (sim.history.data[-1][0] - sim.history.data[0][0]) / time_unit.value

    print(f'Real time duration: {real_time} {time_unit.name.lower()}.')

    real_time_minutes = (
        sim.history.data[-1][0] - sim.history.data[0][0]
    ) / time_unit_minutes.value

    print(f'Video length: {real_time_minutes / speed} minutes.')


def main():
    cli()


if __name__ == '__main__':
    main()
