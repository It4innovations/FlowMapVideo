from collections import defaultdict
from dataclasses import dataclass, InitVar, asdict
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
    #
    timestamp: int
    node_from: int
    node_to: int
    count_from: int
    count_to: int


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
    count_from: int = 0
    count_to: int = 0
    length: InitVar[float] = None
    graph: InitVar[Graph] = None

    def __post_init__(self, length, graph):
        self.length = length

        if length is None and (self.node_from is not None or self.node_to is not None):
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

        return Row(**params)



def preprocess_fill_missing_times(df, graph, speed=1, fps=25):

    node_counts = defaultdict(lambda: 0)

    interval = speed / fps

    start = datetime.now()

    # change datetime to int
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**6 # resolution in milliseconds

    df['timestamp'] = df['timestamp'].div(1000 * interval).round().astype(np.int64)
    df = df.groupby(['timestamp','vehicle_id']).first().reset_index()

    rows = [Row(interval=interval, **kwargs) for kwargs in df.to_dict(orient='records')]
    rows = sorted(rows, key=lambda x: (x.vehicle_id, x.timestamp))

    new_rows = []
    for i, (row, next_row) in enumerate(zip(rows[:], rows[1:] + [None])):  # TODO: row => record; processing_record & next_record
        if next_row is None or row.vehicle_id != next_row.vehicle_id:
            new_rows.append(row)
            if(row.start_offset_m < row.length / 2):
                closer_node = row.node_from
                row.count_from = 1
            else:
                closer_node =  row.node_to
                row.count_to = 1

            node_counts[(row.timestamp, closer_node)] += 1
        else:  # fill missing records
            new_timestamps = [*range(row.timestamp, next_row.timestamp)][:-1]
            new_params = []

            if row.segment_id == next_row.segment_id:
                new_offsets = np.linspace(row.start_offset_m, next_row.start_offset_m, len(new_timestamps), endpoint=False)
                processing_segments = [row] * len(new_offsets)
            else:
                # 1. fill missing offset between two consecutive records
                new_offsets = np.linspace(row.start_offset_m, next_row.start_offset_m + row.length, len(new_timestamps), endpoint=False)
                # 2. adjust offset where the segment change
                new_offsets = np.where(new_offsets > row.length, new_offsets - row.length, new_offsets)
                processing_segments = np.where(new_offsets < row.length, row, next_row)

            new_params = zip(new_timestamps, new_offsets)

            closer_node = None
            half_length_current = row.length / 2
            half_length_next = next_row.length / 2
            for segment, (timestamp, start_offset_m) in zip(processing_segments, new_params):
                    timestamp, start_offset_m = param
                    new_row = Row.create_from_existing(timestamp, start_offset_m, segment)
                    new_rows.append(row_new)

                    if start_offset_m < half_length_first: # TODO: think of finer division
                        closer_node = row.node_from
                        # row_new.count_from = 1  # TODO: rename to 'attach_to' wihich is a number from division paramter
                    else:
                        closer_node = row.node_to
                    #     row_new.count_to = 1
                    new_row.attach_to = None # TODO: compute based on division number
                # else:
                #     new_row = Row.create_from_existing(timestamp, start_offset_m, next_row)
                #     rows_new.append(row_new)

                #     if start_offset_m < half_length_second:  # TODO: think of finer division
                #         closer_node = new_rows[-1].node_from
                #         row_new.count_from = 1
                #     else:
                #         closer_node =  new_rows[-1].node_to
                #         row_new.count_to = 1

                counts_dict[(new_rows[-1].timestamp, closer_node)] += 1

    return new_rows, counts_dict


def preprocess_add_counts(records, counts_dict):
    density_list = []
    records = sorted(records, key=lambda x: (x.timestamp, x.node_from, x.node_to))

    for (timestamp, node_from, node_to), _ in groupby(records, lambda x: (x.timestamp, x.node_from, x.node_to)):
        count_from = 0
        count_to = 0
        if (timestamp, node_from) in counts_dict:
            count_from = counts_dict[(timestamp, node_from)]
        if (timestamp, node_to) in counts_dict:
            count_to = counts_dict[(timestamp, node_to)]
        density_list.append(Density(timestamp,node_from, node_to, count_from, count_to))

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
