import pprint

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.widgets import Slider
from datetime import datetime

from app_io import load_input
from ax_settings import twin_axes

from app_collection_plot import plot_routes
from app_base_graph import get_route_network, get_route_network_small
from app_plot import plot_segments
# from app_plot_cars import plot_cars

from app_df import get_max_vehicle_count, get_max_time, get_min_time


def with_slider():

    g = get_route_network()
    times_df = load_input("../data/gv_202106016_7_10.parquet", g)
    f, ax_density, ax_map_settings = twin_axes(g)

    # get max car count
    max_count = get_max_vehicle_count(times_df)

    # add slider
    plt.subplots_adjust(bottom=0.1)
    time_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(ax=time_slider_ax, label='Time [s]', valmin=get_min_time(times_df), valmax=get_max_time(times_df), valstep=1)

    def on_press(event):
        if event.key == 'right':
            time_slider.set_val(time_slider.val + 1)
        elif event.key == 'left':
            time_slider.set_val(time_slider.val - 1)

    f.canvas.mpl_connect('key_press_event', on_press)

    def update(val):
        start = datetime.now()
        # clear ax and copy base map
        ax_density.clear()
        ax_map_settings.apply(ax_density)
        ax_density.axis('off')

        segments = times_df.loc[val]

        # MAIN PLOT FUNCTION
        _ = plot_routes(g, segments, ax=ax_density, max_count=max_count)

#         plot_cars(g, segments, ax_density)

        f.canvas.draw_idle()

        finish = datetime.now()
        print(finish - start)

    time_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    with_slider()