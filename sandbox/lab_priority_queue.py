import heapq


class Strategy(object):
    def __init__(self, expression, statistics, comp=None):
        self._expression = expression
        self._stats = statistics
        self._comp  = comp

        if not 'fitness' in self._stats and comp is None:
            raise ValueError("fitness and compare function are missing")



    def __lt__(self, other):
        if self._comp:
            return self._comp(self, other)
        else:
            return self._stats['fitness'] < other._stats['fitness']

    def __str__(self):
        return 'expression: {}, fitness: {}'.format(self._expression, self._stats['fitness'])





s_1 = Strategy('a1', {'fitness': -1})
s_2 = Strategy('a2', {'fitness': -2})
s_3 = Strategy('a3', {'fitness': -3})

list_strategy = [s_1, s_2, s_3]

heap_strategy = []

for item in list_strategy:
    heapq.heappush(heap_strategy, item)


while heap_strategy:
    print heapq.heappop(heap_strategy)


