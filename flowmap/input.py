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


def add_count_to_dictionary(dictionary, key, lock=None):
    if lock is not None:
        with lock:
            if key not in dictionary:
                dictionary[key] = 1
            else:
                dictionary[key] += 1
    else:
        if key not in dictionary:
            dictionary[key] = 1
        else:
            dictionary[key] += 1

def preprocess_fill_missing_times(df, g, speed=1, fps=25, counts_dict = {}, lock = None):
    global graph
    graph = g

    interval = speed / fps

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

    for i, row in enumerate(rows):
        if row[1].timestamp is None:
            rows_new.append(row[0])
            if(row[0].start_offset_m < row[0].length/2):
                closer_node = row[0].node_from
                row[0].count_from = 1
            else:
                closer_node =  row[0].node_to
                row[0].count_to = 1

            add_count_to_dictionary(counts_dict, (row[0].timestamp, closer_node), lock)
            continue

        if row[0].length is None or row[1].length is None:
            continue

        if row[1].timestamp is None or row[0].vehicle_id != row[1].vehicle_id:
            rows_new.append(row[0])
            if(row[0].start_offset_m < row[0].length/2):
                closer_node = row[0].node_from
                row[0].count_from = 1
            else:
                closer_node =  row[0].node_to
                row[0].count_to = 1

            add_count_to_dictionary(counts_dict, (row[0].timestamp, closer_node), lock)

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

                add_count_to_dictionary(counts_dict, (rows_new[-1].timestamp, closer_node), lock)

    del rows
    gc.collect()
    return rows_new, counts_dict


def preprocess_add_counts(records, counts_dict):
    density_list = []
    records = sorted(records, key=lambda x: (x.timestamp, x.node_from, x.node_to))
    for key, _ in groupby(records, lambda x: (x.timestamp, x.node_from, x.node_to)):
        count_from = 0
        count_to = 0
        if (key[0], key[1]) in counts_dict:
            count_from = counts_dict[(key[0], key[1])]
        if (key[0], key[2]) in counts_dict:
            count_to = counts_dict[(key[0], key[2])]
        density_list.append(Density(key[0],key[1],key[2], count_from, count_to))

    return density_list


def preprocess_history_records(df, g, speed=1, fps=25):
    start = datetime.now()
    records, counts_dict = preprocess_fill_missing_times(df, g, speed, fps)
    print("rows filled in: ", datetime.now() - start)

    start2 = datetime.now()
    density_list = preprocess_add_counts(records, counts_dict)
    print(df.shape)
    print("counts added in: ", datetime.now() - start2)
    print("total time: ", datetime.now() - start)
    return density_list
