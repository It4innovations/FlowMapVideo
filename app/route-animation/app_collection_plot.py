import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd

from app_plot import get_density, plot_route_width


# TRANSFORMS X AND Y COORDS LISTS INTO LIST OF LINES (DEFINED BY START AND END POINT)
def reshape(x, y):
    points = np.vstack([x, y]).T.reshape(-1, 1, 2)
    points = np.concatenate([points[:-1], points[1:]], axis=1)
    return points

# první fáze
# [node1, node2] = [density1, density2]
# [node2, node1] = [density5, density6]
# [node1, node3] = [density1, density3]
# [node2, node4] = [density5, density4]

# druhá fáze
# [node1, node2] = [density1, edge_den1, edge_den2, density2]
# [node1, node3] = [density1, density3]

# MAIN PLOT FUNCTION
def plot_route(G, segment, ax,
                min_density, max_density,
                min_width_density, max_width_density,
                **pg_kwargs):
    x = []
    y = []
    lines = []
    color_scalars = []

#     print(segment['node_from'], segment['node_to'])
    edge = G.get_edge_data(segment['node_from'], segment['node_to'])
    if edge is not None:
        data = min(edge.values(), key=lambda d: d["length"])
        if "geometry" in data:
            xs, ys = data["geometry"].xy
            x.extend(xs)
            y.extend(ys)
        else:
            x.extend((G.nodes[segment['node_from']]["x"], G.nodes[segment['node_to']]["x"]))
            y.extend((G.nodes[segment['node_from']]["y"], G.nodes[segment['node_to']]["y"]))

        # color gradient
        density_from = segment['count_list'][0]
        density_to = segment['count_list'][1]

        line = reshape(x, y)
        color_scalar = np.linspace(density_from, density_to, len(x) - 1)

        lines.append(line)
        color_scalars.append(color_scalar)

        # width as filling XXX
#         plot_route_width(ax, x, y, density_from, density_to, min_width_density, max_width_density)

    return lines, color_scalars


def plot_routes(G, segments, ax,
                min_density=1, max_density=10,
                min_width_density=10, max_width_density=15,
                **pg_kwargs):

    lines = []
    color_scalars = []
#     print(list(segments.index.values))
#     segments_dict = segments.to_dict('records')
    if isinstance(segments, pd.Series):
        lines, color_scalars = plot_route(G, segments, ax,
                min_density, max_density,
                min_width_density, max_width_density,
                **pg_kwargs)
    else:
        for _, s in segments.iterrows():
            lines_new, color_scalars_new = plot_route(G, s, ax,
                    min_density, max_density,
                    min_width_density, max_width_density,
                    **pg_kwargs)
            lines.extend(lines_new)
            color_scalars.extend(color_scalars_new)
    # plotting of gradient lines
#     print(lines)
#     print(color_scalars)
    lines = np.vstack(lines)
    color_scalars = np.hstack(color_scalars)
    norm = plt.Normalize(min_density, max_density)

    coll = LineCollection(lines, cmap='autumn_r', norm=norm)

    # width in collection XXX
    line_widths = np.interp(color_scalars, [min_width_density, max_width_density], [2, 4])
    coll.set_linewidth(line_widths)

    coll.set_array(color_scalars)
    ax.add_collection(coll)