from collections import defaultdict
from dataclasses import dataclass, InitVar, asdict, field
from datetime import datetime, timedelta
import operator
from itertools import zip_longest
from pandas import date_range, DateOffset, to_datetime
import numpy as np
from math import ceil
from operator import attrgetter
from itertools import groupby
import gc
import numpy as np
from networkx import MultiDiGraph as Graph

@dataclass
class Density:
    """Segment in time with count of vehicles on left and right side of the segment."""

    # TODO: find out a better name!
    timestamp: int
    node_from: int
    node_to: int
    counts: list


@dataclass
class Record:
    timestamp: int
    vehicle_id: int
    segment_id: str
    start_offset_m: float
    speed_mps: float
    status: str
    node_from: int
    node_to: int
    length: InitVar[float] = None
    graph: InitVar[Graph] = None

    def __post_init__(self, length, graph = None):
        self.length = length

        if length is None and graph is not None and (self.node_from is not None or self.node_to is not None):
            data = graph.get_edge_data(self.node_from, self.node_to)
            assert data is not None

            data = data[0]
            assert "length" in data
            self.length = data['length']

    @staticmethod
    def create_from_existing(timestamp, start_offset_m, base_record):
        params = asdict(base_record)
        params['timestamp'] = timestamp
        params['start_offset_m'] = start_offset_m

        return Record(**params)


def add_count_to_dictionary(record, divide, edge_counts, node_counts):
    part_of_edge = record.start_offset_m // (record.length / divide)

    if part_of_edge == 0:
        node_counts[(record.timestamp, record.node_from)] += 1
    elif part_of_edge == (divide - 1):
        node_counts[(record.timestamp, record.node_to)] += 1
    else:
        edge_counts[(record.timestamp, record.node_from, record.node_to, part_of_edge)] += 1


def preprocess_fill_missing_times(df, graph, speed=1, fps=25, divide=2):
    node_counts = defaultdict(lambda: 0)
    edge_counts = defaultdict(lambda: 0)

    interval = speed / fps

    # change datetime to int
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**6 # resolution in milliseconds

    df['timestamp'] = df['timestamp'].div(1000 * interval).round().astype(np.int64)
    df = df.groupby(['timestamp','vehicle_id']).first().reset_index()

    records = [Record(**kwargs, graph=graph) for kwargs in df.to_dict(orient='records')]
    records = sorted(records, key=lambda x: (x.vehicle_id, x.timestamp))

    new_records = []
    for i, (processing_record, next_record) in enumerate(zip(records[:], records[1:] + [None])):
        if next_record is None or processing_record.vehicle_id != next_record.vehicle_id:
            new_records.append(processing_record)

            add_count_to_dictionary(processing_record, divide, edge_counts, node_counts)

        else:  # fill missing records
            new_timestamps = [*range(processing_record.timestamp, next_record.timestamp)]
            new_params = []

            if processing_record.segment_id == next_record.segment_id:
                new_offsets = np.linspace(processing_record.start_offset_m, next_record.start_offset_m, len(new_timestamps), endpoint=False)
                processing_segments = [processing_record] * len(new_offsets)
            else:
                # 1. fill missing offset between two consecutive records
                new_offsets = np.linspace(processing_record.start_offset_m, next_record.start_offset_m + processing_record.length, len(new_timestamps), endpoint=False)
                # 2. adjust offset where the segment change
                processing_segments = np.where(new_offsets > processing_record.length, next_record, processing_record)
                new_offsets = np.where(new_offsets > processing_record.length, new_offsets - processing_record.length, new_offsets)

            new_params = zip(new_timestamps, new_offsets)

            closer_node = None
            for segment, (timestamp, start_offset_m) in zip(processing_segments, new_params):
                new_record = Record.create_from_existing(timestamp, start_offset_m, segment)
                new_records.append(new_record)

                add_count_to_dictionary(segment, divide, edge_counts, node_counts)

    return new_records, node_counts, edge_counts


def preprocess_add_counts(records, node_counts, edge_counts, divide):
    density_list = []
    records = sorted(records, key=lambda x: (x.timestamp, x.node_from, x.node_to))

    for (timestamp, node_from, node_to), _ in groupby(records, lambda x: (x.timestamp, x.node_from, x.node_to)):
        counts = [node_counts[(timestamp, node_from)]]
        for part_of_edge in range(1, divide - 1):
            counts.append(edge_counts[(timestamp, node_from, node_to, part_of_edge)])
        counts.append(node_counts[(timestamp, node_to)])

        density_list.append(Density(timestamp, node_from, node_to, counts))

    return density_list


def preprocess_history_records(df, g, speed=1, fps=25, divide=2):
    if(divide < 2):
        print('Division smaller than 2 is not possible, setting division to 2.')
        divide = 2

    records, node_counts, edge_counts = preprocess_fill_missing_times(df, g, speed, fps, divide)
    density_list = preprocess_add_counts(records, node_counts, edge_counts, divide)

    return density_list
