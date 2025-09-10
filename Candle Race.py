"""
python "Candle Race".py < candle_race_test.txt 
"""
from typing import final
from roar_net_api.operations import *
from tabulate import tabulate


# ---------------------------------- Problem --------------------------------



@final
class Problem(SupportsEmptySolution, SupportsConstructionNeighbourhood):
    def __init__(self, villages):
        self.villages = villages
        self.num_villages = len(self.villages)
        self.travel_times = calc_travel_times(villages)

    def __str__(self):
        table = tabulate(
            self.travel_times,
            headers=[f"Village {i}" for i in range(len(self.villages))],
            showindex=[f"Village {i}" for i in range(len(self.villages))],
            tablefmt="grid"
        )
        return (
            f"Village Network with {self.num_villages} villages\n"
            f"Villages: {self.villages}\n"
            f"Travel Times:\n{table}"
        )

    def empty_solution(self):
        return Solution(self, [], 0, 0, get_available_candle_length())

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
    def __init__(self, problem, sequence, total_travel_time, accumulated_candle_length, upper_bound, not_visited_villages):
        self.problem = problem
        self.sequence = sequence
        self.not_visited_villages = not_visited_villages        # A set of village indexes not visited
        self.total_travel_time = total_travel_time
        self.accumulated_candle_length = accumulated_candle_length                      # Score
        self.upper_bound = upper_bound          # Is the accumulated_candle_length + the length of the remaining candels not blown out

    # Not necessary probably
    def __str__(self):
        return f""
    
    @property
    def is_feasible(self):
        return True
    
    def copy_solution(self):
        return Solution(self.problem, self.sequence.copy(), self.total_travel_time, self.accumulated_candle_length, self.upper_bound)

    def objective_value(self):
        return self.accumulated_candle_length

    def lower_bound(self):
        return -self.upper_bound


# ------------------------------- Neighbourhood ------------------------------
@final
class AddNeighbourhood(SupportsMoves[Solution, "AddMove"]):
    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution):
        assert self.problem == solution.problem
        for i in solution.not_visited_villages:
            yield AddMove(self, solution.sequence[-1], i)

# ----------------------------------- Moves -----------------------------------
@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
    def __init__(self, neighbourhood, i, j):
        self.neighbourhood = neighbourhood
        self.i = i                              # From village
        self.j = j                              # To village

    def apply_move(self, solution):
        solution.sequence.append(self.j)

        solution.total_travel_time += solution.problem.travel_times[self.i][self.j]
        solution.accumulated_candle_length += get_candle_length(solution.total_travel_time, solution.problem.villages[self.j])

    def lower_bound_increment(self, solution):
        """ Return accumulated_candle_length after traveling to village + get_available_candle_length()"""


        #incr = min(0, solution.problem.villages[self.j][2] - (solution.problem.travel_times[self.i][self.j] + solution.total_travel_time) * solution.problem.villages[self.j][3])



# ------------------------------- Helpers ------------------------------
def calc_travel_times(villages):
    """
    input: [[0, 0], [16, 25, 464, 2], [10, 34, 696, 6], [28, 17, 302, 5], [19, 57, 523, 10]]
    output: [[0, 41, 44, 45, 76], [41, 0, 15, 20, 35], [44, 15, 0, 35, 32], [45, 20, 35, 0, 49], [76, 35, 32, 49, 0]]
    """
    l = len(villages)
    return [[abs(villages[i][0] - villages[j][0]) + abs(villages[i][1] - villages[j][1]) for j in range(l)] for i in range(l)]

def get_candle_length(total_travel_time, village):
    return max(0, (village[2] - village[3]*total_travel_time))

def get_available_candle_length():
    """ Return the sum of the candle lengths of the not_visited_villages"""
    NotImplemented


if __name__ == "__main__":
    import roar_net_api.algorithms as alg
    import sys

    problem = Problem.from_textio(sys.stdin)
    print(problem)

    #solution = alg.greedy_construction(problem)
    #print(solution.sequence)
