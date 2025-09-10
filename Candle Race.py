"""
python "Candle Race".py < candle_race_test.txt 
"""
from typing import final

from roar_net_api.algorithms import greedy_construction
from roar_net_api.operations import *
from tabulate import tabulate
from typing import TextIO, final


# ---------------------------------- Problem --------------------------------



@final
class Problem(SupportsEmptySolution, SupportsConstructionNeighbourhood):
    def __init__(self, villages):
        self.villages = villages
        self.num_villages = len(self.villages)
        self.travel_times = calc_travel_times(villages)

    def __eq__(self, other):
        if not isinstance(other, Problem):
            return NotImplemented
        return (
                self.villages == other.villages
                and self.num_villages == other.num_villages
                and self.travel_times == other.travel_times
        )

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
        return Solution(self, [0], 0, 0, {i for i in range(1, self.num_villages)})

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
        return AddNeighbourhood(self)

# ---------------------------------- Solution --------------------------------
@final
class Solution():
    def __init__(self, problem, sequence, total_travel_time, accumulated_candle_length, not_visited_villages):
        self.problem = problem
        self.sequence = sequence
        self.not_visited_villages = not_visited_villages
        self.total_travel_time = total_travel_time
        self.accumulated_candle_length = accumulated_candle_length
        self.ub = self.upper_bound()
        self.available_candle_length = get_available_candle_length(self.total_travel_time, [self.problem.villages[i] for i in self.not_visited_villages])

    # Not necessary probably
    def __str__(self):
        return (f"Sequence: {self.sequence}\n"
                f"Accumulated_candle_length: {self.accumulated_candle_length}\n"
                f"Total_travel_time: {self.total_travel_time}\n"
                f"Not_visited_villages: {self.not_visited_villages}\n")

    def to_textio(self, f: TextIO) -> None:
        st = ""
        for s in range(1, len(self.sequence) - 1):
            st = st + str(self.sequence[s]) + "\n"
        st = st + str(self.sequence[-1])
        f.write(st)

    @property
    def is_feasible(self):
        return True

    def upper_bound(self):

        villages = [self.problem.villages[i] for i in self.not_visited_villages]
        ub = self.accumulated_candle_length + get_available_candle_length(self.total_travel_time, villages)
        self.ub = ub
        return ub

    def copy_solution(self):
        return Solution(self.problem, self.sequence.copy(), self.total_travel_time, self.accumulated_candle_length, self.not_visited_villages.copy() )

    def objective_value(self):
        return self.accumulated_candle_length

    def lower_bound(self):
        return -self.ub


# ------------------------------- Neighbourhood ------------------------------
@final
class AddNeighbourhood(SupportsMoves[Solution, "AddMove"]):
    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution):
        for i in solution.not_visited_villages:
            yield AddMove(self, solution.sequence[-1], i)

# ----------------------------------- Moves -----------------------------------
@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
    def __init__(self, neighbourhood, i, j):
        self.neighbourhood = neighbourhood
        self.i = i
        self.j = j

    def __str__(self):
        return f"AddMove: i={self.i}, j={self.j}"

    def apply_move(self, solution):
        solution.sequence.append(self.j)
        solution.not_visited_villages.remove(self.j)

        solution.total_travel_time += solution.problem.travel_times[self.i][self.j]
        solution.accumulated_candle_length += get_candle_length(solution.total_travel_time, solution.problem.villages[self.j])

        return solution

    def upper_bound_increment(self, solution):
        total_travel_time = solution.total_travel_time + solution.problem.travel_times[self.i][self.j]
        delta_accumulated_candle_length = get_candle_length(total_travel_time, solution.problem.villages[self.j])
        delta_available_candle_length = get_available_candle_length(total_travel_time, [solution.problem.villages[i] for i in solution.not_visited_villages if i != self.j]) - solution.available_candle_length
        return delta_accumulated_candle_length + delta_available_candle_length


    def lower_bound_increment(self, solution):
        return -self.upper_bound_increment(solution)


# ------------------------------- Helpers ------------------------------
def calc_travel_times(villages):
    """
    input: [[0, 0], [16, 25, 464, 2], [10, 34, 696, 6], [28, 17, 302, 5], [19, 57, 523, 10]]
    output: [[0, 41, 44, 45, 76], [41, 0, 15, 20, 35], [44, 15, 0, 35, 32], [45, 20, 35, 0, 49], [76, 35, 32, 49, 0]]
    """
    l = len(villages)
    return [[abs(villages[i][0] - villages[j][0]) + abs(villages[i][1] - villages[j][1]) for j in range(l)] for i in range(l)]

def get_candle_length(travel_time, village):
    return max(0, (village[2] - village[3]*travel_time))

def get_available_candle_length(travel_time, villages):
    """ Return the sum of the candle lengths of the not_visited_villages"""
    return sum([get_candle_length(travel_time, village) for village in villages])


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=argparse.FileType('r'), default=sys.stdin)

    args = parser.parse_args()
    filename = args.input_file.name
    problem = Problem.from_textio(args.input_file)

    solution = greedy_construction(problem)
    print(solution)

    file = open("output.txt", "w")
    solution.to_textio(file)
    file.close()

    #print(problem)
    # solution = problem.empty_solution()

    # print(solution)
    # construction_neighbourhood = problem.construction_neighbourhood()
    # moves = construction_neighbourhood.moves(solution)
    # init_objective_value = solution.objective_value()
    # init_upper_bound = solution.upper_bound()
    # print(f"init_objective_value: {init_objective_value}")
    # print(f"init_upper_bound: {init_upper_bound}\n")
    #
    # for move in moves:
    #     print(move)
    #     s = solution.copy_solution()
    #     print(f"move_upper_bound_increment: {move.upper_bound_increment(s)}")
    #     move.apply_move(s)
    #     s_objective_value = s.objective_value()
    #     s_upper_bound = s.upper_bound()
    #     print(f"s_objective_value: {s_objective_value}")
    #     print(f"s_upper_bound: {s_upper_bound}")
    #     print()


    #solution = alg.greedy_construction(problem)
    #print(solution.sequence)
