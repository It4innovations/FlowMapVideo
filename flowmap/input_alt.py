from dataclasses import dataclass
from datetime import datetime, timedelta
import operator
from itertools import zip_longest
from pandas import date_range, DateOffset, to_datetime
from numpy import linspace, int64, where
from math import ceil
from operator import attrgetter
from itertools import groupby
import gc
import numpy as np

graph = None

@dataclass
class Density:
    timestamp: int
    node_from: int
    node_to: int
    count_from: int
    count_to: int

    def __init__(
        self,
        timestamp: int,
        node_from: int,
        node_to: int,
        count_from: int,
        count_to: int
        ):
        self.timestamp = timestamp
        self.node_to = node_to
        self.node_from = node_from
        self.count_from = count_from
        self.count_to = count_to

@dataclass
class Row: # (Record)
    timestamp: int
    vehicle_id: int
    segment_id: str
    start_offset_m: float
    speed_mps: float
    status: str
    node_from: int
    node_to: int
    length: float
    count_from: int
    count_to: int

    def __init__(
        self,
        vehicle_id: int,
        segment_id: str,
        start_offset_m: float,
        speed_mps: float,
        status: str,
        node_from: int,
        node_to: int,
        interval: int,
        length: float = None,
        timestamp: int = None
        ):
        global graph
        self.timestamp = timestamp
        self.vehicle_id = vehicle_id
        self.segment_id = segment_id
        self.start_offset_m = start_offset_m
        self.speed_mps = speed_mps
        self.status = status
        self.node_to = node_to
        self.node_from = node_from

        if length is not None:
            self.length = length
        else:
            if node_from is None or node_to is None:
                self.length = None
            else:
                data = graph.get_edge_data(node_from, node_to)
                # NOTE: uncomment assert, comment returning nan
                # assert data
                if data is not None:
                    data = data[0]
                    assert "length" in data
                    self.length = data['length']
                else:
                    self.length = None

        self.count_from = 0
        self.count_to = 0


def preprocess_history_records(simulation, g, speed=1, fps=25, version=1):
    global graph
    graph = g

    interval = speed / fps
#     df = simulation.global_view.to_dataframe()  # NOTE: this method has some non-trivial overhead
#     df = simulation.history.to_dataframe()  # NOTE: this method has some non-trivial overhead
#     if version == 1:
#         df = df.loc[(df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (df['timestamp'] < np.datetime64('2021-06-16T08:01:00.000')),:]
#     elif version == 2:
#         df = df.loc[(df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (df['timestamp'] < np.datetime64('2021-06-16T08:30:00.000')),:]
#     else:
#         df = df.loc[(df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (df['timestamp'] < np.datetime64('2021-06-16T09:00:00.000')),:]

    print('data loaded')
    start = datetime.now()

    # change datetime to int
    df['timestamp']= to_datetime(df['timestamp']).astype(int64)//10**6 # resolution in milliseconds

    df['timestamp'] = df['timestamp'].div(1000 * interval).round().astype(int64)
    df = df.groupby(['timestamp','vehicle_id']).first().reset_index()

    rows = [Row(interval=interval, **kwargs) for kwargs in df.to_dict(orient='records')]
    del df
    gc.collect()

    rows = sorted(rows, key=lambda x: (x.vehicle_id, x.timestamp))
    rows = list(zip_longest(rows, rows[1:], fillvalue=Row(None,None,None,None,None,None,None,None,None)))

    rows_new = []
    print(len(rows))
    counts_dict = {}

    for i, row in enumerate(rows):
        if row[0].length is None or row[1].length is None:
            continue

        if row[0].vehicle_id != row[1].vehicle_id:
            rows_new.append(row[0])
            if(row[0].start_offset_m < row[0].length/2):
                closer_node = row[0].node_from
                row[0].count_from = 1
            else:
                closer_node =  row[0].node_to
                row[0].count_to = 1

            if (row[0].timestamp, closer_node) not in counts_dict:
                counts_dict[(row[0].timestamp, closer_node)] = 1
            else:
                counts_dict[(row[0].timestamp, closer_node)] += 1

        else:
            new_timestamps = [*range(row[0].timestamp, row[1].timestamp)][:-1]
            new_params = []

            if (row[0].segment_id == row[1].segment_id):
                new_offsets = linspace(row[0].start_offset_m, row[1].start_offset_m,len(new_timestamps) + 1)[:-1]
                new_params = zip(new_timestamps, new_offsets)
                is_first_segment = None
            else:
                new_offsets = linspace(row[0].start_offset_m, row[1].start_offset_m + row[0].length,len(new_timestamps) + 1)[:-1]
                is_first_segment = where(new_offsets<row[0].length, True, False)
                new_offsets = where(new_offsets>row[0].length, new_offsets - row[0].length, new_offsets)
                new_params = zip(new_timestamps, new_offsets)

            closer_node = None
            half_length_first = row[0].length / 2
            half_length_second = row[1].length / 2
            for i, param in enumerate(new_params):
                if(is_first_segment is None or is_first_segment[i]):
                    row_new = Row(
                        timestamp = param[0],
                        segment_id = row[0].segment_id,
                        start_offset_m = param[1],
                        speed_mps = row[0].speed_mps,
                        status = row[0].status,
                        node_from = row[0].node_from,
                        node_to = row[0].node_to,
                        vehicle_id = row[0].vehicle_id,
                        length = row[0].length,
                        interval=interval
                        )
                    rows_new.append(row_new)

                    if(param[1] < half_length_first):
                        closer_node = row[0].node_from
                        row_new.count_from = 1

                    else:
                        closer_node = row[0].node_to
                        row_new.count_to = 1

                else:
                    row_new = Row(
                        timestamp = param[0],
                        segment_id = row[1].segment_id,
                        start_offset_m = param[1],
                        speed_mps = row[1].speed_mps,
                        status = row[1].status,
                        node_from = row[1].node_from,
                        node_to = row[1].node_to,
                        vehicle_id = row[1].vehicle_id,
                        length = row[1].length,
                        interval=interval
                        )
                    rows_new.append(row_new)

                    if(param[1] < half_length_second):
                        closer_node = rows_new[-1].node_from
                        row_new.count_from = 1
                    else:
                        closer_node =  rows_new[-1].node_to
                        row_new.count_to = 1



                if (rows_new[-1].timestamp, closer_node) not in counts_dict:
                    counts_dict[(rows_new[-1].timestamp, closer_node)] = 1
                else:
                    counts_dict[(rows_new[-1].timestamp, closer_node)] += 1

    print("rows filled in ", datetime.now() - start)
    start_n = datetime.now()
    print(len(rows_new))
    del rows
    gc.collect()

    density_list = []
    rows_new = sorted(rows_new, key=lambda x: (x.timestamp, x.node_from, x.node_to))
    for key, _ in groupby(rows_new, lambda x: (x.timestamp, x.node_from, x.node_to)):
#         group = list(group)
#         node_from_sum = np.sum([c.count_from for c in group])
#         node_to_sum = np.sum([c.count_to for c in group])
        density_list.append(Density(key[0],key[1],key[2], counts_dict[(key[0], key[1])], counts_dict[(key[0], key[2])]))

    finish = datetime.now()
    print("counts added in ", finish - start_n)

    print('doba trvani: ', finish - start)
    print(len(density_list))

    return density_list
