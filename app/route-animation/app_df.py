def get_max_vehicle_count(df):
    return df['vehicle_count'].max()

def get_min_time(df):
    return df.index.min()

def get_max_time(df):
    return df.index.max()