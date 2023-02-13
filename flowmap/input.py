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

def get_length(node_from_to, g):
    node_from, node_to = node_from_to
    data = g.get_edge_data(node_from, node_to)
    # NOTE: uncomment assert, comment returning nan
#     assert data
    if data is None:
        return np.nan
    data = data[0]
    assert "length" in data
    return data['length']


def fill_missing_timestamps(row, interval: int):
    start, end = row[["timestamp", "next_timestamp"]]
    if end == 0:  # if change of vehicle, just keep one row
        return np.int64(start),
    return tuple(range(np.int64(start),np.int64(end) - 1))


def fill_missing_offsets(row):
    start, end, is_last_in_segment, timestamps, length = row[["start_offset_m", "next_offset", "last_in_segment","timestamp","length"]]
    # if change of vehicle, just keep the row untouched
    if np.isnan(end):
        return start,
    if is_last_in_segment:
        return np.linspace(start, end + length, num=len(timestamps))
    return np.linspace(start, end, num=len(timestamps)+1)[:-1]


def fill_missing(row, interval):
    start, end = row[["timestamp", "next_timestamp"]]
    if end == 0:  # if change of vehicle, just keep one row
        timestamps = np.int64(start),
    else:
        timestamps = tuple(range(np.int64(start),np.int64(end) - 1))

    start, end, is_last_in_segment, length = row[["start_offset_m", "next_offset", "last_in_segment","length"]]
    # if change of vehicle, just keep the row untouched
    if np.isnan(end):
        offsets = start,
        return timestamps, offsets
    if is_last_in_segment:
        return timestamps, np.linspace(start, end + length, num=len(timestamps))
    return timestamps, np.linspace(start, end, num=len(timestamps)+1)[:-1]

def fill_missing_rows(df, interval):
    # add missing times
    df.sort_values(['vehicle_id', 'timestamp'], inplace=True)
    df['next_timestamp'] = df['timestamp'].shift(-1, fill_value=0)
    df['next_offset'] = df['start_offset_m'].shift(-1).astype('float')

    # true if a vehicle changes the semgent
    mask = (df.node_from != df.node_from.shift(-1)) | (df.node_to != df.node_to.shift(-1))

    df['last_in_segment'] = False
    df.loc[mask, 'last_in_segment'] = True

    # >>> change of vehicle
    mask = df.vehicle_id != df.vehicle_id.shift(-1)
    df.loc[mask,'next_timestamp'] = 0
    df.loc[mask,'next_offset'] = np.nan

    # now timestamp contains a list of timestamps which will be unrolled later
    df['timestamp'] = df.apply(lambda row: fill_missing_timestamps(row, interval), axis=1)

    df['start_offset_m'] = df.apply(lambda row: fill_missing_offsets(row), axis=1)

#     df['timestamp','start_offset_m'] = df.apply(lambda row: fill_missing(row, fps), axis=1)


    df['next_node_from'] = df['node_from'].shift(-1, fill_value=0)
    df['next_node_to'] = df['node_to'].shift(-1, fill_value=0)
    df['next_length'] = df['length'].shift(-1)
    df['next_vehicle_id'] = df['vehicle_id'].shift(-1)

    # reimplemented two lines before

#     new_rows = [] # TODO: check if works
#     for row in df.iterrows():
#         missing_timestamps = fill_missing_timestamps(row)
#         missing_offsets = fill_missing_offsets(row, missing_timestamps)
#         new_rows.append(row.to_list().extend(zip(missing_timestamps, missing_offset)))


    # drop unnecessary columns
    df.drop('next_timestamp', axis=1, inplace=True)
    df.drop('next_offset', axis=1, inplace=True)
    df.drop('last_in_segment', axis=1, inplace=True)

     #         <<< end of change vehicle
    df = df.explode(column=['timestamp','start_offset_m'])

    df.reset_index(inplace=True)

    df['new_node_to'] =  df['node_to']
    df['new_node_from'] =  df['node_from']
    df['new_start_offset_m'] =  df['start_offset_m']
    df['new_length'] =  df['length']

    mask = (df['start_offset_m'] > df['length']) & (df['next_vehicle_id'] == df['vehicle_id'])
    df.loc[mask ,'new_node_from'] = df['next_node_from']
    df.loc[mask ,'new_node_to'] = df['next_node_to']
    df.loc[mask, 'new_start_offset_m'] = df['start_offset_m'] - df['length']
    df.loc[mask, 'new_length'] = df['next_length']

    df.drop(['node_to','node_from','start_offset_m','length','next_node_from','next_node_to','next_length','next_vehicle_id'], axis=1, inplace=True)
    df.rename(columns = {'new_node_to':'node_to','new_node_from':'node_from','new_start_offset_m':'start_offset_m','new_length':'length'}, inplace = True)

    return df

def add_counts(df):
    # find out which node is the vehicle closer to
    # code for splitting each edge in half
    #
    # >>> NOTE: decide whether to use finer division or not as it influences the Pavla's work
    # NOTE: dividing the segment into halves; TODO: consider finer division
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
    return df


def preprocess_history_records(simulation, g, speed=1, fps=25):
    # load file
    df = simulation.history.to_dataframe()  # NOTE: this method has some non-trivial overhead
#     df = simulation.global_view.to_dataframe()  # NOTE: this method has some non-trivial overhead

#     print(df)
#     print(df.dtypes)
#     print(df.loc[(df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (df['timestamp'] < np.datetime64('2021-06-16T08:01:00.000')),:])
#     print(df.loc[(df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (df['timestamp'] < np.datetime64('2021-06-16T08:30:00.000')),:])
#     print(df.loc[(df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (df['timestamp'] < np.datetime64('2021-06-16T09:00:00.000')),:])
    interval = speed / fps
    df = df.loc[(df['timestamp'] > np.datetime64('2021-06-16T08:00:00.000')) & (df['timestamp'] < np.datetime64('2021-06-16T09:00:00.000')),:]
    print('data loaded')
    start = datetime.now()

    start_n = datetime.now()
    df = df[['timestamp','node_from','node_to','vehicle_id','start_offset_m']].copy()

    df['node_from'] = df['node_from'].astype(str).astype(np.uint64)
    df['node_to'] = df['node_to'].astype(str).astype(np.uint64)
    df['vehicle_id'] = df['vehicle_id'].astype(str).astype(np.uint16)
    df['start_offset_m'] = df['start_offset_m'].astype(str).astype(np.float32)
    df['length'] = pd.Series(dtype='float32')
    # change datetime to int
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**6 # resolution in milliseconds
    print(df)

    df['timestamp'] = df['timestamp'].div(1000 * interval).round()
    df = df.groupby(['timestamp','vehicle_id']).first().reset_index()

    print('data types set in ')
    print(datetime.now() - start_n)
    start_n = datetime.now()

    # add column with length
    df["length"] = df[["node_from", "node_to"]].apply(get_length, axis=1, g=g)

    # NOTE: comment this later - there are segments not found in the map in the data

    # drop rows where path hasn't been found in the graph
    df.dropna(subset=['length'], inplace=True)
    print('length added in ')
    print(datetime.now() - start_n)
    start_n = datetime.now()

    print(df.shape)
    df = fill_missing_rows(df, interval)
    print(df.shape)

    print('missing rows filled in in ')
    print(datetime.now() - start)
    start_n = datetime.now()

    df.sort_values(['timestamp', 'vehicle_id'], inplace=True)

    df = add_counts(df)

    df.drop(['length','start_offset_m','vehicle_id'], axis=1, inplace=True)
    df.reset_index(level=['node_from', 'node_to'], inplace=True)

    print('counts added in ')
    print(datetime.now() - start_n)

    finish = datetime.now()
    print('doba trvani: ', finish - start)

    return df
