from pandas import read_parquet, to_datetime, cut, to_numeric, to_pickle, read_pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx

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

def get_length(row, map, dict):
#     print(row.dtype)
    try:
        row['length'] = map[row['node_from']][row['node_to']][0]['length']
        return row
    except KeyError:
        pass
    try:
#         print('puvodni: ', row)
        path_nodes = dict.get((row['node_from'],row['node_to']))
        if path_nodes is None:
            _, path_nodes = nx.bidirectional_dijkstra(map, row['node_from'], row['node_to'], weight='length')
            dict[(row['node_from'],row['node_to'])] = path_nodes
#         values_from_dict = dict.get((row['node_from'],row['node_to']))
#         len_of_values_from_dict = 0
#             len_of_values_from_dict = len(values_from_dict)
#         dict[(row['node_from'],row['node_to'])] = 1
        lengths = []
        total_length = 0
        old_length_sum = 0
        for node_index in range(len(path_nodes)-1):
            length = map[path_nodes[node_index]][path_nodes[node_index + 1]][0]['length']
            old_length_sum = total_length
            total_length += length
            lengths.append(length)
            if total_length >= row['start_offset_m']:
                row['node_from'] = path_nodes[node_index]
                row['node_to'] = path_nodes[node_index + 1]
                row['length'] = length
                row['start_offset_m'] = row['start_offset_m'] - old_length_sum
#                 print(lengths)
#                 print('novy: ', row)
                return row

        row['length'] = length
        row['start_offset_m'] = length
        row['node_from'] = path_nodes[-2]
        row['node_to'] = path_nodes[-1]
#         print(lengths)
#         print('novy: ', row)
        return row
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    row['length'] = np.nan
    return row

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
    df = read_parquet(path, engine="fastparquet", columns=('timestamp','node_from','node_to','vehicle_id','start_offset_m'))
    df.reset_index(inplace=True)

#     df.drop('status', axis=1, inplace=True)
#     df.drop('speed_mps', axis=1, inplace=True)

    df['node_from'] = df['node_from'].astype(str).astype(np.uint64)
    df['node_to'] = df['node_to'].astype(str).astype(np.uint64)
    df['vehicle_id'] = df['vehicle_id'].astype(str).astype(np.uint16)
    df['start_offset_m'] = df['start_offset_m'].astype(str).astype(np.float32)
    df['length'] = pd.Series(dtype='float32')
    # change datetime to int
    # TODO casting datetime64[ns] values to int64 with .astype(...) is deprecated and will raise in a future version. Use .view(...) instead.
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**9

    # add column with length
    dict = {}
    df = df.apply(get_length, axis=1, map=g, dict=dict)

    # drop rows where path hasn't been found in the graph
    df.dropna(subset=['length'], inplace=True)

    # add missing times
    df.sort_values(['vehicle_id', 'timestamp'], inplace=True)
    df['diff'] = df['timestamp'].shift(-1).fillna(0).astype('int')
    df['next_offset'] = df['start_offset_m'].shift(-1).astype('float')

    mask = (df.node_from != df.node_from.shift(-1)) | (df.node_to != df.node_to.shift(-1))

    df.loc[mask, 'next_offset'] = df['length']

    df['last_in_segment'] = False
    df.loc[mask, 'last_in_segment'] = True

    mask = df.vehicle_id != df.vehicle_id.shift(-1)
    df.loc[mask,'diff'] = 0
    df.loc[mask,'next_offset'] = np.nan

    df['timestamp'] = df.apply(lambda x: make_list(x['timestamp'], x['diff']), axis=1)
    df['start_offset_m'] = df.apply(lambda row: make_list_of_offsets(row['start_offset_m'], row['next_offset'], row['timestamp'], row['last_in_segment']), axis=1)

    df.drop('diff', axis=1, inplace=True)
    df.drop('next_offset', axis=1, inplace=True)
    df.drop('last_in_segment', axis=1, inplace=True)

    df.set_index(['node_from','node_to','vehicle_id','length','index'], inplace=True)

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

    df_from = pd.concat([df_from, df_to])
    df_from = df_from.groupby(["timestamp", "node_from"]).agg({'count_from': 'sum'}, inplace=True)

    # group by time, nodes and segment, add vehicle count and list of offsets
    df = df.groupby(["timestamp", "node_from","node_to",'length']).agg({'start_offset_m': lambda x: list(x)}, inplace=True)
    df.reset_index(level=['length'], inplace=True)

    df = df.join(df_from)
    df_from.rename_axis(index=["timestamp", "node_to"], inplace=True)
    df_from.rename(columns = {'count_from':'count_to'}, inplace = True)

    df = df.join(df_from)

    df['count_list'] = df.apply(lambda x: get_counts_by_offset(x['start_offset_m'], segment_length, x['length'], x['count_from'], x['count_to']), axis=1)
#     df['count_list'] = df.apply(lambda x: get_counts_half(x['start_offset_m'], x['length']), axis=1)
#     df[['count_from','count_to']] = pd.DataFrame(df.count_list.tolist(), index= df.index)
#     df['count_from'] = df['count_list'][0]
#     df['count_to'] = df['count_list'][1]
    df.drop('count_list', axis=1, inplace=True)
    df.reset_index(level=['node_from', 'node_to'], inplace=True)
    return df