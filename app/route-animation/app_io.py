from pandas import read_parquet, to_datetime, cut, to_numeric, to_pickle, read_pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from networkx import shortest_path_length

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

def get_length(g, node_from_id, node_to_id):
    try:
        return g[node_from_id][node_to_id][0]['length']
    except KeyError:
        pass
    try:
        return shortest_path_length(g, node_from_id, node_to_id, weight='length')
    except Exception:
        return np.nan

def make_list(start,end):
    if end == 0:
        return np.int64(start),
    return tuple(range(np.int64(start),np.int64(end)))

def make_list_of_offsets(start, end, timestamps, is_last_in_segment):
    if np.isnan(end):
        return start,
    if is_last_in_segment:
        return np.linspace(start, end, num=len(timestamps))
    return np.linspace(start, end, num=len(timestamps)+1)[:-1]

def load_input(path, g, segment_length):

    # load file
    df = read_parquet(path, engine="fastparquet")

    df.reset_index(inplace=True)

#     df.drop('status', axis=1, inplace=True)
    df.drop('speed_mps', axis=1, inplace=True)

    df['node_from'] = df['node_from'].astype(str).astype(np.int64)
    df['node_to'] = df['node_to'].astype(str).astype(np.int64)

    # add column with length
    df['length'] = df.apply(lambda row: get_length(g, row['node_from'], row['node_to']), axis=1)
#     return

    # drop rows where path hasn't been found in the graph
    df = df.dropna(subset=['length'])

    # change datetime to int
    # TODO casting datetime64[ns] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**9
#     df['timestamp'] = to_datetime(df['timestamp']).view(int)//10**9

    # add missing times
    df.sort_values(['vehicle_id', 'timestamp'], inplace=True)
    df['diff'] = df['timestamp'].shift(-1).fillna(0).astype('int')
    df['next_offset'] = df['start_offset_m'].shift(-1).astype('float')

    mask = (df.node_from != df.node_from.shift(-1)) | (df.node_to != df.node_to.shift(-1))
    df['next_offset'][mask] = df['length']
    df['last_in_segment'] = False
    df['last_in_segment'][mask] = True

    mask = df.vehicle_id != df.vehicle_id.shift(-1)
    df['diff'][mask] = 0
    df['next_offset'][mask] = np.nan

    df['timestamp'] = df.apply(lambda x: make_list(x['timestamp'], x['diff']), axis=1)
    df['start_offset_m'] = df.apply(lambda row: make_list_of_offsets(row['start_offset_m'], row['next_offset'], row['timestamp'], row['last_in_segment']), axis=1)

    df.drop('diff', axis=1, inplace=True)
    df.drop('next_offset', axis=1, inplace=True)
    df.drop('last_in_segment', axis=1, inplace=True)
    df.set_index(['node_from','node_to','vehicle_id','segment_id','length','index'], inplace=True)
    df = df.apply(pd.Series.explode)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    df['count_from'] = np.where(((df['length'] > 2 * segment_length) & (df['start_offset_m'] < segment_length)) | ((df['length'] < 2 * segment_length) & (df['start_offset_m'] < df['length'] / 2)), 1, 0)
    df['count_to'] = np.where(((df['length'] > 2 * segment_length) & (df['start_offset_m'] > df['length'] - segment_length)) | ((df['length'] < 2 * segment_length) & (df['count_from'] != 1)), 1, 0)

    df2 = df.groupby(['timestamp',"node_from","node_to"]).agg({'count_from': 'sum', 'count_to': 'sum'})
    df2.reset_index(inplace=True)
    df.drop('count_from', axis=1, inplace=True)
    df.drop('count_to', axis=1, inplace=True)

    df_from = df2[['timestamp','node_from', 'count_from']].copy()

    df_to = df2[['timestamp','node_to', 'count_to']].copy()
    df_to.rename(columns = {'node_to':'node_from','count_to':'count_from'}, inplace = True)

    df_from = df_from.append(df_to)
    df_from.sort_values(['timestamp'], inplace=True)

    df_from = df_from.groupby(["timestamp", "node_from"]).agg({'count_from': 'sum'}, inplace=True)

    # group by time, nodes and segment, add vehicle count and list of offsets
    df = df.groupby(["timestamp", "node_from","node_to",'length']).agg({'vehicle_id': 'count', 'start_offset_m': lambda x: list(x)}, inplace=True)
    df.reset_index(level=['length'], inplace=True)
    df.rename(columns = {'vehicle_id':'vehicle_count'}, inplace = True)


    df = df.join(df_from)
    df_from.rename_axis(index=["timestamp", "node_to"], inplace=True)
    df_from.rename(columns = {'count_from':'count_to'}, inplace = True)
#     print(df.to_string(index=True,max_rows=1))

    df = df.join(df_from)

    df['count_list'] = df.apply(lambda x: get_counts_by_offset(x['start_offset_m'], segment_length, x['length'], x['count_from'], x['count_to']), axis=1)
#     df['count_list'] = df.apply(lambda x: get_counts_half(x['start_offset_m'], x['length']), axis=1)

    print(df.to_string(index=True,max_rows=100))

    return df