from pandas import to_datetime
import pandas as pd
import numpy as np


def fill_missing_timestamps(row):
    start, end = row[['timestamp', 'next_timestamp']]
    if end == -1:  # if change of vehicle, just keep one row
        return np.int64(start),
    return tuple(range(np.int64(start),np.int64(end)))


def fill_missing_offsets(row):
    start, end, is_last_in_segment, timestamps, length = row[['start_offset_m',
                                                              'next_offset',
                                                              'last_in_segment',
                                                              'timestamp',
                                                              'segment_length']]
    # if change of vehicle, just keep the row untouched
    if end == -1:
        return start,
    if is_last_in_segment:
        return np.linspace(start, end + length, num=len(timestamps), endpoint=False)
    return np.linspace(start, end, num=len(timestamps), endpoint=False)


def fill_missing_rows(df):
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
    df['timestamp'] = df.apply(lambda row: fill_missing_timestamps(row), axis=1)
    df['start_offset_m'] = df.apply(lambda row: fill_missing_offsets(row), axis=1)

    df[['next_node_from', 'next_node_to', 'next_length', 'next_vehicle_id']] = df[['node_from', 'node_to', 'segment_length', 'vehicle_id']].shift(-1, fill_value=0)

    # drop unnecessary columns
    df.drop(['next_timestamp', 'next_offset', 'last_in_segment'], axis=1, inplace=True)

    # <<< end of change vehicle
    df = df.explode(column=['timestamp', 'start_offset_m'])
    df.reset_index(inplace=True)

    df[['new_node_from', 'new_node_to', 'new_start_offset_m', 'new_length']] = df[['node_from', 'node_to', 'start_offset_m', 'segment_length']]

    # the change of segment
    mask = (df['start_offset_m'] > df['segment_length']) & (df['next_vehicle_id'] == df['vehicle_id'])
    column_pairs = zip(
        ['new_node_from', 'new_node_to', 'new_length'],
        ['next_node_from', 'next_node_to', 'next_length'])

    for column_new, column_next in column_pairs:
        df.loc[mask, column_new] = df.loc[mask, column_next]

    df.loc[mask, 'new_start_offset_m'] = df.loc[mask, 'start_offset_m'] - df.loc[mask, 'segment_length']

    df.drop([
        'node_to',
        'node_from',
        'start_offset_m',
        'segment_length',
        'next_node_from',
        'next_node_to',
        'next_length',
        'next_vehicle_id'
    ], axis=1, inplace=True)

    df.rename(columns = {
        'new_node_to': 'node_to',
        'new_node_from': 'node_from',
        'new_start_offset_m': 'start_offset_m',
        'new_length': 'segment_length'
    }, inplace = True)

    return df


def add_counts(df, divide=2):
    # find out which node is the vehicle closer to
    # code for splitting each edge in half

    # NOTE: dividing the segment into parts

    if(divide < 2):
        print('Division smaller than 2 is not possible, setting division to 2.')
        divide = 2

    # find out which part of the segment is the vehicle in
    df['part'] = df['start_offset_m'] // (df['segment_length'] / divide)
    df.loc[df['part'] >= divide, 'part'] = divide - 1
    count_columns = ['counts_' + str(x) for x in range(divide - 2)]
    count_columns[:0] = ['count_from']
    count_columns.append('count_to')
    df[count_columns] = pd.DataFrame([[0] * divide], index=df.index)

    for i in range(divide):
        df.loc[df['part'] == i, count_columns[i]] = 1

    # create dataframe with number of vehicles for each node and timestamp
    d = {k:'sum' for k in count_columns}
    df = df.groupby(['timestamp', 'node_from', 'node_to']).agg(d)
    df.reset_index(inplace=True)

    # dataframe for "from" nodes
    df_from = df[['timestamp', 'node_from', 'count_from']].copy()

    # dataframe for "to" nodes
    df_to = df[['timestamp', 'node_to', 'count_to']].copy()

    # renaming for concatenation
    df_to.rename(columns = {'node_to': 'node_from', 'count_to': 'count_from'}, inplace = True)

    # concatenate both dataframes and get the total vehicle count for each node and timestamp
    df_from = pd.concat([df_from, df_to])
    df_from = df_from.groupby(['timestamp', 'node_from']).agg({'count_from': 'sum'}, inplace=True)

    df.drop(['count_from', 'count_to'], axis=1, inplace=True)
    df.set_index(['timestamp', 'node_from', 'node_to'], inplace=True)

    # connect node counts with segments
    df = df.join(df_from)
    df_from.rename_axis(index=['timestamp', 'node_to'], inplace=True)
    df_from.rename(columns = {'count_from': 'count_to'}, inplace = True)
    df = df.join(df_from)

    return df


def preprocess_history_records(df, g, speed=1, fps=25, divide=2):
    df = preprocess_fill_missing_times(df, g, speed, fps)
    df = preprocess_add_counts(df, divide)
    return df


def preprocess_fill_missing_times(df, g, speed=1, fps=25):
    interval = speed / fps

    df = df[['timestamp', 'node_from', 'node_to', 'vehicle_id', 'start_offset_m', 'segment_length']].copy()

    df['node_from'] = df['node_from'].astype(str).astype(np.uint64)
    df['node_to'] = df['node_to'].astype(str).astype(np.uint64)
    df['vehicle_id'] = df['vehicle_id'].astype(str).astype(np.uint16)
    df['start_offset_m'] = df['start_offset_m'].astype(str).astype(np.float32)
    # change datetime to int
    df['timestamp']= to_datetime(df['timestamp']).astype(np.int64)//10**6 # resolution in milliseconds

    df['timestamp'] = df['timestamp'].div(1000 * interval).round().astype(np.int64)
    df = df.groupby(['timestamp', 'vehicle_id']).first().reset_index()

    df = fill_missing_rows(df)

    return df


def preprocess_add_counts(df, divide=2):
    df.sort_values(['timestamp', 'vehicle_id'], inplace=True)
    df = add_counts(df, divide)

    df.reset_index(level=['node_from', 'node_to'], inplace=True)

    return df