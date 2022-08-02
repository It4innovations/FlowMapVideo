from pandas import read_parquet, to_datetime, cut
from numpy import int64, arange, nan, floor_divide, bincount
from datetime import datetime, timedelta
from networkx import shortest_path_length

def get_counts_by_offset(offset_list, section_length):
    return bincount(floor_divide(offset_list,section_length).astype(int))

def get_length(g, node_from_id, node_to_id):
    try:
        return shortest_path_length(g, node_from_id, node_to_id, weight='length')
    except Exception:
        return nan

def make_list(start,end):
    if end == 0:
        return start,
    return tuple(range(start,end))

def load_input(path, g, segment_length):
    # load file
    df = read_parquet(path, engine="fastparquet")

    # add column with length
    df['length'] = df.apply(lambda row: get_length(g, row['node_from'], row['node_to']), axis=1)

    # drop rows where path hasn't been found in the graph
    df = df.dropna(subset=['length'])

    # change datetime to int
    # TODO casting datetime64[ns] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
    df['timestamp']= to_datetime(df['timestamp']).astype(int64)//10**9

    # add missing times
    df.sort_values(['vehicle_id', 'timestamp'], inplace=True)
    df['diff'] = df['timestamp'].shift(-1).fillna(0).astype('int')
    mask = df.vehicle_id != df.vehicle_id.shift(-1)
    df['diff'][mask] = 0
    df['timestamp'] = df.apply(lambda x: make_list(x['timestamp'], x['diff']), axis=1)
    df.drop('diff', axis=1, inplace=True)
    df = df.explode('timestamp')
    df = df.dropna(subset=['timestamp'])

    # group cars near to nodes
    mask = df['start_offset_m'] < segment_length
    df.loc[mask,['node_to','segment_id']] = -1

    # TODO handle situations when edge len < segment_length better (add to closer node)
    mask = (df['segment_id'] != -1) & (df['start_offset_m'] > df['length'] - segment_length)
    df.loc[mask,['node_from','segment_id']] = -1

    # TODO switch node_from and node_to columns
    # df.loc[mask,'node_from'] = df['node_to']

    # group by time, nodes and segment, add vehicle count and list of offsets
    df = df.groupby(["timestamp","node_from","node_to","segment_id"]).agg({'vehicle_id': 'count',
                         'start_offset_m': lambda x: list(x)})
    df.rename(columns = {'vehicle_id':'vehicle_count'}, inplace = True)

    # TODO is there better alternative for apply?
    df['count_list'] = df.apply(lambda x: get_counts_by_offset(x['start_offset_m'], segment_length), axis=1)

    return df


