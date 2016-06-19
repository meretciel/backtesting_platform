
"""
Script: strategy.py

In this script, we will define the Strategy class. It serves as a wrapper of all the information about a strategy.

"""


import heapq
import pandas as pd
import numpy as np
import random
import stoperator.GA as GA
import logging

logger = logging.getLogger('main.strategy')
logger.setLevel(logging.INFO)
logger.propagate = True


class Strategy(object):
    def __init__(self, expression=None, fitness=None, summary=None, simulation_summary=None):
        self._expression = expression
        self._fitness    = fitness
        self._summary    = summary
        self._simulation_summary = simulation_summary

    def __lt__(self, other):
        if other._fitness is None or np.isnan(other._fitness):
            return False

        if self._fitness is None or np.isnan(self._fitness):
            return True

        return self._fitness < other._fitness


    def __str__(self):
        return self._simulation_summary


class StrategyPool(object):
    """
    The StrategyPool is a collection of trading strategies. It represents the Population in the GA framework. It is a
    wrapper on top of the heapq package. The underlying data structure is priority queue or heap. Note that it is a
    min-heap because we would pop the strategy with the lowest fitness.

    The StrategyPool class provides API similar as other collection class. In addition, it has a evolve function which
    will trigger the Genetic Algorithm.

    """
    def __init__(self, size=None, simulator=None):
        self._limit_size = size
        self._simulator  = simulator
        self._heap = []
        self._total_fitness = 0.


    def resize(self, size):
        self._limit_size = size

    def push(self, strategy):
        if len(self._heap) == self._limit_size:
            raise ValueError("The pool is full")
        assert isinstance(strategy, Strategy)
        logger.info("Add new strategy:")
        logger.info("expression: {}".format(strategy._expression))
        logger.info("fitness: {}".format(strategy._fitness))
        self._total_fitness += strategy._fitness if self.isValidFitness(strategy._fitness) else 0.
        heapq.heappush(self._heap, strategy)

    def pop(self):
        worst = self.top()
        self._total_fitness -= worst._fitness if self.isValidFitness(worst._fitness) else 0.
        return heapq.heappop(self._heap)

    def top(self):
        if not self._heap:
            raise ValueError("The pool is empty.")
        return self._heap[0]

    def empty(self):
        return len(self._heap) == 0

    def full(self):
        return len(self._heap) == self._limit_size

    def nsmallest(self, n):
        return heapq.nsmallest(n, self._heap)

    @staticmethod
    def isValidFitness(fitness):
        return fitness is not None and not np.isnan(fitness)

    def pushIfBetter(self, strategy):
        if len(self._heap) < self._limit_size:
            self.push(strategy)
        else:
            worst = self._heap[0]
            if strategy > worst:
                logger.info("Add new strategy:")
                logger.info("expression: {}".format(strategy._expression))
                logger.info("fitness: {}".format(strategy._fitness))
                a = strategy._fitness if self.isValidFitness(strategy._fitness) else 0.
                b = worst._fitness if self.isValidFitness(worst._fitness)  else 0.
                self._total_fitness += a - b
                heapq.heapreplace(self._heap, strategy)

    def evolve(self, n_generation=100, p_crossover=0.6, p_mutate=0.4, p_opSwap=0.05, p_var=0.02, p_op=0.02, p_num=0.05, p_div=0.02, p_addOp=0.05):
        assert len(self._heap) != 0

        logger.info('Start evolution.')
        for n in xrange(n_generation):
            logger.info("================ {}-th generation ==============".format(n))

            rand_crossover = np.random.uniform()
            rand_mutate    = np.random.uniform()

            strategy_1 = random.choice(self._heap)
            strategy_2 = random.choice(self._heap)
            expression_1 = strategy_1._expression
            expression_2 = strategy_2._expression

            logger.info("Expression selected: ")
            logger.info(expression_1)
            logger.info(expression_2)

            if rand_crossover < p_crossover and expression_1 != expression_2:
                logger.info("Appplying crossover operator...")

                new_expression_1, new_expression_2 = GA.crossover(expression_1, expression_2, p_opSwap=p_opSwap)


                logger.info("New expression generated:")
                logger.info(new_expression_1)
                logger.info(new_expression_2)

                if new_expression_1 != expression_1:
                    self._simulator.simulate(expression=new_expression_1, use_expression=True)
                    self._simulator.analyze()
                    self.pushIfBetter(self._simulator.output_strategy())

                if new_expression_2 != expression_2:
                    self._simulator.simulate(expression=new_expression_2, use_expression=True)
                    self._simulator.analyze()
                    self.pushIfBetter(self._simulator.output_strategy())

            if rand_mutate < p_mutate:
                logger.info("Applying mutate operator...")

                new_expression = GA.mutate(expression_1, p_var=p_var, p_op=p_op, p_num=p_num, p_div=p_div, p_addOp=p_addOp)

                logger.info("New expression generated:")
                logger.info(new_expression)

                if new_expression != expression_1:
                    self._simulator.simulate(expression=new_expression, use_expression=True)
                    self._simulator.analyze()
                    self.pushIfBetter(self._simulator.output_strategy())

            if n % 10 == 0 and not self.isValidFitness(self._total_fitness):
                logger.warning("total fitness is None. Now update the value of the total_fitness.")
                s = 0
                for item in self._heap:
                    s += item._fitness if self.isValidFitness(item._fitness) else 0
                self._total_fitness = s

            logger.info("=========> The average fitness of {:>3}-th generation is: {:6.4F}".format(n, self._total_fitness / float(len(self._heap))))



    def summary(self):

        print("{:<15} {:10.4F}".format('the average fitness is:', self._total_fitness / float(len(self._heap))))


    def listStrategy(self, n=None):
        _n = n or min(len(self._heap), 10)
        for item in self._heap[:_n]:
            print("{:10<} {}".format('expressions:', item._expression))

    def add_from_expression(self, expressions):
        n = min(len(expressions), self._limit_size - len(self._heap))
        for expression in expressions[:n]:
            self.push(Strategy(expression=expression))

    def dump(self, output_path):
        list_expression = []
        list_fitness    = []

        for item in self._heap:
            list_expression.append(item._expression)
            list_fitness.append(item._fitness)

        df = pd.DataFrame({"expression": list_expression,
                           "fitness":    list_fitness})
        df.to_csv(output_path, index=False)

    def load(self, file):
        pass


    def update(self, func):
        # update the fitness of strategy in the pool
        map(func, self._heap)
        # heap sort the list
        heapq.heapify(self._heap)






