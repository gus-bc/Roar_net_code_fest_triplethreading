"""
python "Candle Race".py < candle_race_test.txt 
"""

from __future__ import annotations
from typing import final
from roar_net_api.operations import *



# ---------------------------------- Problem --------------------------------
@final
class Problem(SupportsEmptySolution, SupportsConstructionNeighbourhood):
    def __init__(self, villages):
        self.villages = villages
        self.distances = calc_distances(villages)
        self.neighbourhood = None

    def empty_solution(self):
        return Solution(self, [], 0, 0)

    @classmethod
    def from_textio(cls, f):
        num_villages = f.readline().strip()

        villages = []
        if num_villages.isdigit():
            num_villages = int(num_villages)

            for i in range(num_villages):
                s = f.readline().strip().split(" ")
                for j in range(len(s)):
                    
                    if s[j].isdigit():
                        s[j] = int(s[j])
                    else:
                        raise Exception("Invalid instance")
                villages.append(s)
        else:
            raise Exception("Invalid instance")
        return cls(villages)
    
    def construction_neighbourhood(self):
        if self.neighbourhood is None:
            self.neighbourhood = AddNeighbourhood(self)
        return self.neighbourhood

# ---------------------------------- Solution --------------------------------
@final
class Solution():
    def __init__(self, problem, sequence, total_distance, accumulated_candle_length):
        self.problem = problem
        self.sequence = sequence
        self.total_distance = total_distance
        self.accumulated_candle_length = accumulated_candle_length                      # Score
    
    # Not necessary probably
    def __str__(self):
        return f""
    
    @property
    def is_feasible(self):
        return True
    
    def copy_solution(self):
        return Solution(self.problem, self.sequence.copy(), self.total_distance, self.accumulated_candle_length)


# ------------------------------- Neighbourhood ------------------------------
@final
class AddNeighbourhood(SupportsMoves[Solution, "AddMove"]):
    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution):
        assert self.problem == solution.problem
        l = len(solution.sequence)
        for i in range(l):
            i = solution.sequence[i - 1] if i > 0 else 0
            j = solution.sequence[i]
            yield AddMove(self, i, j)

# ----------------------------------- Moves -----------------------------------
@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
    def __init__(self, neighbourhood, i, j):
        self.neighbourhood = neighbourhood
        self.i = i                              # From village
        self.j = j                              # To village

    def apply_move(self, solution):
        solution.sequence.append(self.j)

        prob = solution.problem
        solution.total_distance += prob.distances[self.i][self.j]
        solution.accumulated_candle_length += min(0, prob.villages[self.j][2] - solution.total_distance * prob.villages[self.j][3])

    def lower_bound_increment(self, solution):
        prob = solution.problem

        incr = min(0, prob.villages[self.j][2] - (prob.distances[self.i][self.j] + solution.total_distance) * prob.villages[self.j][3])



# ------------------------------- Helpers ------------------------------
def calc_distances(villages):
    """
    input: [[0, 0], [16, 25, 464, 2], [10, 34, 696, 6], [28, 17, 302, 5], [19, 57, 523, 10]]
    output: [[0, 41, 44, 45, 76], [41, 0, 15, 20, 35], [44, 15, 0, 35, 32], [45, 20, 35, 0, 49], [76, 35, 32, 49, 0]]
    """
    l = len(villages)
    return [[abs(villages[i][0] - villages[j][0]) + abs(villages[i][1] - villages[j][1]) for j in range(l)] for i in range(l)]


if __name__ == "__main__":
    import roar_net_api.algorithms as alg
    import sys


    problem = Problem.from_textio(sys.stdin)
    solution = alg.greedy_construction(problem)
    print(solution.sequence)
