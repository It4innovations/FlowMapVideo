from pandas import read_parquet, to_datetime, cut, to_numeric, to_pickle, read_pickle
import pandas as pd
from numpy import int64, arange, nan, floor_divide, bincount, where
from datetime import datetime, timedelta
from networkx import shortest_path_length

def get_counts_by_offset(offset_list, section_length, road_length):
    values_list = list(offset_list)
    values_list.append(road_length)
    values_list = bincount(floor_divide(values_list,section_length).astype(int))
    values_list[-1] -= 1
    return values_list

def get_counts_half(offset_list, length):
    length = length/2.0
    smaller_count = 0
    for num in offset_list:
        if num < length:
            smaller_count += 1
    return (smaller_count, len(offset_list) - smaller_count)

def get_length(g, node_from_id, node_to_id):
    try:
        return shortest_path_length(g, node_from_id, node_to_id, weight='length')
    except Exception:
        return nan

def make_list(start,end):
    if end == 0:
        return int64(start),
    return tuple(range(int64(start),int64(end)))


# def load_input(path, g, segment_length):
def load_input(path, g):

    # load file
    df = read_parquet(path, engine="fastparquet")
    df.reset_index(inplace=True)

#     df.drop('status', axis=1, inplace=True)

    df['node_from'] = df['node_from'].astype(str).astype(int64)
    df['node_to'] = df['node_to'].astype(str).astype(int64)

    df['node_from'],df['node_to']=where(df['node_from']>df['node_to'],(df['node_to'],df['node_from']),(df['node_from'],df['node_to']))

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
#     mask = df['start_offset_m'] < segment_length
#     df.loc[mask,['node_to','segment_id']] = -1

    # TODO handle situations when edge len < segment_length better (add to closer node)
#     mask = (df['segment_id'] != -1) & (df['start_offset_m'] > df['length'] - segment_length)
#     df.loc[mask,['node_from','segment_id']] = -1

    # TODO switch node_from and node_to columns
    # df.loc[mask,'node_from'] = df['node_to']

#     df['id'] = df.index

    # group by time, nodes and segment, add vehicle count and list of offsets
    df = df.groupby(["timestamp", "node_from","node_to",'length']).agg({'vehicle_id': 'count', 'start_offset_m': lambda x: list(x)}, inplace=True)
    df.reset_index(level=['node_to','node_from','length'], inplace=True)
    df.rename(columns = {'vehicle_id':'vehicle_count'}, inplace = True)

    # TODO is there better alternative for apply?
    # apply has better performance than iterating the dataframe and calling the function for each row,
    # but is it worth the bigger memory usage? how much does the memory demand increase?
    # maybe the list with offsets is not necessary
#     df['count_list'] = df.apply(lambda x: get_counts_by_offset(x['start_offset_m'], segment_length, x['length']), axis=1)
    df['count_list'] = df.apply(lambda x: get_counts_half(x['start_offset_m'], x['length']), axis=1)
#     df[['count_from','count_to']] = pd.DataFrame(df.count_list.tolist(), index= df.index)
#     df.drop('count_list', axis=1, inplace=True)

#     print(df.to_string(index=True,max_rows=100))

    return df