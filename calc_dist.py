#!/usr/bin/python3
import matplotlib.pyplot as plt
import re
import datetime
from typing import List, Dict


def test(n: int):
    return n + 1


def main():
    dates_T = Dict[float, Dict[float, int]]
    dates: dates_T = dict()

    with open("./LABELS/all.txt", "r") as f:
        for line in f.readlines():
            line_splt: List[str] = line.split()
            img, label = line_splt[0], int(line_splt[1])
            m = re.match("[A-Z]+/[\d|-]+/camera\d+/[A-Z]_([\d|-]+)_([\d|.]+)_", img)
            date_str, time_str = m.groups()
            time = datetime.datetime.strptime(time_str, "%H.%M")
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            times = dates.get(date.timestamp(), dict())
            times[time.timestamp()] = times.get(time.timestamp(), 0) + label
            dates[date.timestamp()] = times
    for date, times in dates.items():
        pairs = sorted(times.items())
        timestamps, ns = zip(*pairs)
        datetimes = [datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M") for timestamp in timestamps]
        [print(dt) for dt in datetimes]
        plt.xticks(timestamps, labels=datetimes, rotation=90)
        plt.plot(timestamps, ns)
        plt.show()


if __name__ == "__main__":
    main()
