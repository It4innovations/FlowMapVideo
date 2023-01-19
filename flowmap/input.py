from pandas import read_parquet, to_datetime, cut, to_numeric, to_pickle, read_pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from dataclasses import dataclass

def get_counts_by_offset(offset_list, section_length, road_length, count_from, count_to):
    values_list = list(offset_list)
    values_list.append(road_length)
    values_list = np.bincount(np.floor_divide(values_list,section_length).astype(int))
    values_list[0] = count_from
    if len(values_list) > 1:
        values_list[-1] = count_to
    else:
        values_list[-1] += count_to
    return values_list

def get_counts_half(offset_list, length):
    length = length/2.0
    smaller_count = 0
    for num in offset_list:
        if num < length:
            smaller_count += 1
    return (smaller_count, len(offset_list) - smaller_count)

def get_length(node_from_to, g):
    node_from, node_to = node_from_to
    data = min(g.get_edge_data(node_from, node_to).values(), key=lambda data: data["length"])
    assert "length" in data
    return data["length"]

    # try:
    #     row['length'] = g[row['node_from']][row['node_to']][0]['length']
    #     return row
    # except KeyError:
    #     pass
    # try:
    #     path_nodes = dictionary.get((row['node_from'],row['node_to']))
    #     if path_nodes is None:
    #         _, path_nodes = nx.bidirectional_dijkstra(g, row['node_from'], row['node_to'], weight='length')
    #         dictionary[(row['node_from'],row['node_to'])] = path_nodes
    #     lengths = []
    #     total_length = 0
    #     old_length_sum = 0
    #     for node_index in range(len(path_nodes)-1):
    #         length = g[path_nodes[node_index]][path_nodes[node_index + 1]][0]['length']
    #         old_length_sum = total_length
    #         total_length += length
    #         lengths.append(length)
    #         if total_length >= row['start_offset_m']:
    #             row['node_from'] = path_nodes[node_index]
    #             row['node_to'] = path_nodes[node_index + 1]
    #             row['length'] = length
    #             row['start_offset_m'] = row['start_offset_m'] - old_length_sum
    #             return row

    #     row['length'] = length
    #     row['start_offset_m'] = length
    #     row['node_from'] = path_nodes[-2]
    #     row['node_to'] = path_nodes[-1]
    #     return row
    # except (nx.NetworkXNoPath, nx.NodeNotFound):
    #     pass
    # row['length'] = np.nan
    # return row
    #


@dataclass
class Row: # (Record)
    datetime: datetime # maybe timestamp as an integer
    vehicle_id: int
    segment_id: str
    start_offset_m: float
    speed_mps: float
    status: str
    node_from: int
    node_to: int

def test():
    abc = [(1,2,3), (3,4,5), (5, 6, 7)]
    rows = list(map(lambda args: Row(*args), abc))

def fill_missing_timestamps(row, fps):
    start, end = row[["timestamp", "next_timestamp"]]
    if end == 0:  # 0 means a change of <WHAT>?
        return np.int64(start),
    return tuple(range(np.int64(start),np.int64(end), 1000 / fps)) # NOTE: 1000ms = 1s


def fill_missing_offset(row, timestamps):
    start, end, is_last = row[["start_offste_m", "next_offset", "last_in_segment"]]
    if np.isnan(end):
        return start,
    if is_last_in_segment:
        return np.linspace(start, end, num=len(timestamps))
    return np.linspace(start, end, num=len(timestamps)+1)[:-1]


def preprocess_history_records(simulation, g, fps=1):
    # TODO: document this method; maybe refactor by spliting to separate functions

    # load file
    df = simulation.global_view.to_dataframe()  # NOTE: this method has some non-trivial overhead

    # df.reset_index(inplace=True)
    # df = df[['timestamp','node_from','node_to','vehicle_id','start_offset_m']].copy()
    # df.insert(0, 'index', range(0, len(df)))

    df['node_from'] = df['node_from'].astype(str).astype(np.uint64)
    df['node_to'] = df['node_to'].astype(str).astype(np.uint64)
    df['vehicle_id'] = df['vehicle_id'].astype(str).astype(np.uint16)
    df['start_offset_m'] = df['start_offset_m'].astype(str).astype(np.float32)
    df['length'] = pd.Series(dtype='float32')
    # change datetime to int
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**6 # resolution in milliseconds

    # add column with length
    # dictionary = {}
    df["length"] = df[["node_from", "node_to"]].apply(get_length, axis=1, g=g)

    # drop rows where path hasn't been found in the graph
    # df.dropna(subset=['length'], inplace=True)

    # >>> change of segment
    # a new function - fill missing position of vehicles between frames
    # add missing times
    df.sort_values(['vehicle_id', 'timestamp'], inplace=True)
    df['next_timestamp'] = df['timestamp'].shift(-1).fillna(0).astype('int')

    df['next_offset'] = df['start_offset_m'].shift(-1).astype('float')

    # true if a vehicle change the semgent
    mask = (df.node_from != df.node_from.shift(-1)) | (df.node_to != df.node_to.shift(-1))
    df.loc[mask, 'next_offset'] = df['length']  # floor the the offset when the segment has changed

    df['last_in_segment'] = False
    df.loc[mask, 'last_in_segment'] = True

    # >>> change of vehicle
    #
    mask = df.vehicle_id != df.vehicle_id.shift(-1)
    df.loc[mask,'next_timestamp'] = 0
    df.loc[mask,'next_offset'] = np.nan

    # now timestamp contains a list of timestamps which will be unrolled later
    # df['timestamp'] = df.apply(lambda row: fill_missing_timestamps(row['timestamp'], row['next_timestamp'], fps), axis=1)
    # df['start_offset_m'] = df.apply(lambda row: fill_missing_offsets(row['start_offset_m'], row['next_offset'], row['timestamp'], row['last_in_segment']), axis=1)
    #
    # reimplemented two lines before
    new_rows = [] # TODO: check if works
    for row in df.iterrows():
        missing_timestamps = fill_missing_timestamps(row)
        missing_offsets = fill_missing_offsets(row, missing_timestamps)
        new_rows.append(row.to_list().extend(zip(missing_timestamps, missing_offset)))


    # # drop unnecessary columns
    # df.drop('next_timestamp', axis=1, inplace=True)
    # df.drop('next_offset', axis=1, inplace=True)
    # df.drop('last_in_segment', axis=1, inplace=True)

    # df.set_index(['node_from','node_to','vehicle_id','length'], inplace=True)

    # df = df.apply(pd.Series.explode)  # TODO: maybe try to use explode on dataframe dirctly
    # <<< end of change vehilce

    # df.reset_index(inplace=True)
    # df.drop('index', axis=1, inplace=True)

    # find out which node is the vehicle closer to
    # code for splitting each edge in half
    #
    # >>> NOTE: decide whether to use finer divison or not as it influences the Pavla's work
    # NOTE: dividing the segment into halfs; TODO: onsider finer division
    df['count_from'] = np.where((df['start_offset_m'] < df['length'] / 2), 1, 0)
    df['count_to'] = np.where((df['start_offset_m'] > df['length'] / 2), 1, 0)

    # code for splitting edge into more segments
#     df['count_from'] = np.where(((df['length'] > 2 * segment_length) & (df['start_offset_m'] < segment_length)) | ((df['length'] < 2 * segment_length) & (df['start_offset_m'] < df['length'] / 2)), 1, 0)
#     df['count_to'] = np.where(((df['length'] > 2 * segment_length) & (df['start_offset_m'] > df['length'] - segment_length)) | ((df['length'] < 2 * segment_length) & (df['count_from'] != 1)), 1, 0)

    # create dataframe with number of vehicles for each node and timestamp
    df2 = df.groupby(['timestamp',"node_from","node_to"]).agg({'count_from': 'sum', 'count_to': 'sum'})
    df2.reset_index(inplace=True)
    df.drop('count_from', axis=1, inplace=True)
    df.drop('count_to', axis=1, inplace=True)

    # dataframe for "from" nodes
    df_from = df2[['timestamp','node_from', 'count_from']].copy()

    # dataframe for "to" nodes
    df_to = df2[['timestamp','node_to', 'count_to']].copy()

    # renaming for concatenation
    df_to.rename(columns = {'node_to':'node_from','count_to':'count_from'}, inplace = True)

    # concatenate both dataframes and get the total vehicle count for each node and timestamp
    df_from = pd.concat([df_from, df_to])
    df_from = df_from.groupby(["timestamp", "node_from"]).agg({'count_from': 'sum'}, inplace=True)

    # group by time, nodes and segment
    df = df.groupby(["timestamp", "node_from","node_to"]).count()

    df = df.join(df_from)
    df_from.rename_axis(index=["timestamp", "node_to"], inplace=True)
    df_from.rename(columns = {'count_from':'count_to'}, inplace = True)

    df = df.join(df_from)

    # code for splitting edge into more segments
#     df['count_list'] = df.apply(lambda x: get_counts_by_offset(x['start_offset_m'], segment_length, x['length'], x['count_from'], x['count_to']), axis=1)
#     df['count_list'] = df.apply(lambda x: get_counts_half(x['start_offset_m'], x['length']), axis=1)

    # code for creating list from the count columns
#     df[['count_from','count_to']] = pd.DataFrame(df.count_list.tolist(), index= df.index)
#     df['count_from'] = df['count_list'][0]
#     df['count_to'] = df['count_list'][1]
#     df['count_list'] = list(zip(df['count_from'], df['count_to']))

    df.drop(['length','start_offset_m','vehicle_id'], axis=1, inplace=True)
    df.reset_index(level=['node_from', 'node_to'], inplace=True)
    return df
