"""
.. module:: TimeDiscretizer

TimeDiscretizer
*************

:Description: TimeDiscretizer

    

:Authors: bejar
    

:Version: 

:Created on: 15/09/2014 12:29 

"""

__author__ = 'bejar'

from datetime import date, timedelta
import time


class TimeDiscretizer():
    intervals = None

    def __init__(self, intervals):
        """
        Assigns a list to the intervals attribute or generates a list using intervals as a step
        """
        if type(intervals) is int:
            l = []
            for v in range(0, 24, intervals):
                l.append(v)
            self.intervals = l
        else:
            self.intervals = intervals

    def discretize(self, timestamp):
        stime = date.fromtimestamp(timestamp)
        htime = time.localtime(timestamp)
        hour = htime[3]
        disc = 0
        if hour < self.intervals[0] and self.intervals[0] != 0:
            disc = len(self.intervals) - 1
            stime = stime - timedelta(1)
        for i in range(len(self.intervals) - 1):
            if self.intervals[i] <= hour < self.intervals[i + 1]:
                disc = i
        if hour >= self.intervals[-1]:
            disc = len(self.intervals) - 1
        return '%d%d%d' % (stime.year, stime.month, stime.day), disc

