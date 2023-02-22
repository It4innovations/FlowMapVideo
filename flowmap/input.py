from pandas import read_parquet, to_datetime, cut, to_numeric, to_pickle, read_pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from math import floor


def get_segment_length(node_from_to, g):
    node_from, node_to = node_from_to
    data = g.get_edge_data(node_from, node_to)
    # NOTE: uncomment assert, comment returning nan
    # assert data
    if data is None:
        return np.nan
    data = data[0]
    assert "length" in data
    return data['length']


def fill_missing_timestamps(row, interval: int):
    start, end = row[['timestamp', 'next_timestamp']]
    if end == -1:  # if change of vehicle, just keep one row
        return np.int64(start),
    return tuple(range(np.int64(start),np.int64(end) - 1))


def fill_missing_offsets(row):
    start, end, is_last_in_segment, timestamps, length = row[['start_offset_m',
                                                              'next_offset',
                                                              'last_in_segment',
                                                              'timestamp',
                                                              'length']]
    # if change of vehicle, just keep the row untouched
    if end == -1:
        return start,
    if is_last_in_segment:
        return np.linspace(start, end + length, num=len(timestamps))
    return np.linspace(start, end, num=len(timestamps)+1)[:-1]


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
    df.loc[mask,'next_timestamp'] = -1
    df.loc[mask,'next_offset'] = -1

    # now timestamp contains a list of timestamps which will be unrolled later
    df['timestamp'] = df.apply(lambda row: fill_missing_timestamps(row, interval), axis=1)
    df['start_offset_m'] = df.apply(lambda row: fill_missing_offsets(row), axis=1)

    df[['next_node_from', 'next_node_to', 'next_length', 'next_vehicle_id']] = df[['node_from', 'node_to', 'length', 'vehicle_id']].shift(-1, fill_value=0)

    # drop unnecessary columns
    df.drop(['next_timestamp', 'next_offset', 'last_in_segment'], axis=1, inplace=True)

    # <<< end of change vehicle
    df = df.explode(column=['timestamp', 'start_offset_m'])

    df.reset_index(inplace=True)

    df[['new_node_from', 'new_node_to', 'new_start_offset_m', 'new_length']] = df[['node_from', 'node_to', 'start_offset_m', 'length']]

    # the change of segment
    mask = (df['start_offset_m'] > df['length']) & (df['next_vehicle_id'] == df['vehicle_id'])
    df.loc[mask, ['new_node_from', 'new_node_to', 'new_length']] = df.loc[mask, ['next_node_from', 'next_node_to', 'next_length']]
    df.loc[mask, 'new_start_offset_m'] = df.loc[mask, 'start_offset_m'] - df.loc[mask, 'length']

    df.drop([
        'node_to',
        'node_from',
        'start_offset_m',
        'length',
        'next_node_from',
        'next_node_to',
        'next_length',
        'next_vehicle_id'
    ], axis=1, inplace=True)

    df.rename(columns = {
        'new_node_to': 'node_to',
        'new_node_from': 'node_from',
        'new_start_offset_m': 'start_offset_m',
        'new_length': 'length'
    }, inplace = True)

    df.dropna(inplace=True)

    return df


def add_counts(df, divide=2):
    # find out which node is the vehicle closer to
    # code for splitting each edge in half
    #
    # >>> NOTE: decide whether to use finer division or not as it influences the Pavla's work
    # NOTE: dividing the segment into halves; TODO: consider finer division
    step = df['length'] / divide
    df['counts'] = [0] * divide
    df['counts'][floor(df['start_offset_m'] / step)] = 1
    df[['count_from', 'count_to']] = df['counts'][0], df['counts'][-1]

    # create dataframe with number of vehicles for each node and timestamp
    df2 = df.groupby(['timestamp', 'node_from', 'node_to']).agg({'count_from': 'sum', 'count_to': 'sum'})
    df2.reset_index(inplace=True)
    df.drop('count_from', axis=1, inplace=True)
    df.drop('count_to', axis=1, inplace=True)

    # dataframe for "from" nodes
    df_from = df2[['timestamp', 'node_from', 'count_from']].copy()

    # dataframe for "to" nodes
    df_to = df2[['timestamp', 'node_to', 'count_to']].copy()

    # renaming for concatenation
    df_to.rename(columns = {'node_to': 'node_from', 'count_to': 'count_from'}, inplace = True)

    # concatenate both dataframes and get the total vehicle count for each node and timestamp
    df_from = pd.concat([df_from, df_to])
    df_from = df_from.groupby(['timestamp', 'node_from']).agg({'count_from': 'sum'}, inplace=True)

    # group by time, nodes and segment
    df = df.groupby(['timestamp', 'node_from', 'node_to']).count()

    df = df.join(df_from)
    df_from.rename_axis(index=['timestamp', 'node_to'], inplace=True)
    df_from.rename(columns = {'count_from': 'count_to'}, inplace = True)

    df = df.join(df_from)

    return df


def preprocess_history_records(df, g, speed=1, fps=25, version=1):
    start = datetime.now()
    df = preprocess_fill_missing_times(df, g, speed, fps)
    print(df.shape)
    print("rows filled in: ", datetime.now() - start)

    start2 = datetime.now()
    df = preprocess_add_counts(df)
    print(df.shape)
    print("counts added in: ", datetime.now() - start2)
    print("total time: ", datetime.now() - start)

    return df


def preprocess_fill_missing_times(df, g, speed=1, fps=25):
    interval = speed / fps

    df = df[['timestamp', 'node_from', 'node_to', 'vehicle_id', 'start_offset_m']].copy()

    df['node_from'] = df['node_from'].astype(str).astype(np.uint64)
    df['node_to'] = df['node_to'].astype(str).astype(np.uint64)
    df['vehicle_id'] = df['vehicle_id'].astype(str).astype(np.uint16)
    df['start_offset_m'] = df['start_offset_m'].astype(str).astype(np.float32)
    df['length'] = pd.Series(dtype='float32')
    # change datetime to int
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**6 # resolution in milliseconds

    df['timestamp'] = df['timestamp'].div(1000 * interval).round()
    df = df.groupby(['timestamp', 'vehicle_id']).first().reset_index()

    # add column with length
    df['length'] = df[['node_from', 'node_to']].apply(get_segment_length, axis=1, g=g)

    # NOTE: comment this later - there are segments not found in the map in the data

    # drop rows where path hasn't been found in the graph
    df.dropna(subset=['length'], inplace=True)

    df = fill_missing_rows(df, interval)
    return df


def preprocess_add_counts(df):

    df.sort_values(['timestamp', 'vehicle_id'], inplace=True)
    df = add_counts(df)

    df.drop(['length', 'start_offset_m', 'vehicle_id'], axis=1, inplace=True)
    df.reset_index(level=['node_from', 'node_to'], inplace=True)

    return df
