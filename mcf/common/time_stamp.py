from datetime import datetime
import numpy as np

TS_PATTERN = "%Y-%m-%d%H:%M:%S.%f"

class TimeStamp:

    @classmethod
    def make(cls) -> str:
        timestamp = datetime.now()
        timestamp = timestamp.strftime(TS_PATTERN)
        return timestamp

    @classmethod
    def get_earliest(cls, time_stamps):
        time_stamps_num = []
        for ts in time_stamps:
            time_stamps_num.append(datetime.strptime(ts, TS_PATTERN))
        time_stamp_min = time_stamps[np.argmin(np.array(time_stamps_num))]
        return time_stamp_min
