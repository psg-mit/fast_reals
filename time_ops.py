from timeit import default_timer as timer
import bigfloat as bf
from datetime import timedelta


class Time():

    def __init__(self):
        self.agg_time, self.count = timedelta(seconds=0), 0

    def _timed_op(self, x, y, c, op):
        start = timer()
        result = op(x, y, c)
        end = timer()
        self.count += 1
        self.agg_time += timedelta(seconds=end - start)
        return result

    def add(self, x, y, c):
        return self._timed_op(x, y, c, bf.add)

    def sub(self, x, y, c):
        return self._timed_op(x, y, c, bf.sub)

    def mul(self, x, y, c):
        return self._timed_op(x, y, c, bf.mul)

    def div(self, x, y, c):
        return self._timed_op(x, y, c, bf.div)
