import numpy as np

def fill_missing_timestamps_cython(long long start, long long end):
    if end == -1:  # if change of vehicle, just keep one row
        return start,
    return tuple(range(start,end))

def fill_missing_offsets_cython(double start, double end, bint is_last_in_segment, int timestamps_len, float length):
    # if change of vehicle, just keep the row untouched
    if end == -1:
        return start,
    if is_last_in_segment:
        return np.linspace(start, end + length, num=timestamps_len)
    return np.linspace(start, end, num=timestamps_len+1)[:-1]