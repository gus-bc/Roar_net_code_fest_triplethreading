from random import choice, shuffle, random, randrange
from typing import final, Optional, Union

from roar_net_api.operations import *

# ---------------------------------- Problem --------------------------------
@final
class Problem(SupportsEmptySolution, SupportsConstructionNeighbourhood, SupportsRandomSolution):
    def __init__(self, villages):
        self.villages = villages
        self.num_villages = len(self.villages)
        self.travel_times = calc_travel_times(villages)

    def __eq__(self, other):
        return (
                self.villages == other.villages
                and self.num_villages == other.num_villages
                and self.travel_times == other.travel_times
        )

    def __str__(self):
        return (
                f"Village Network with {self.num_villages} villages\n"
                f"Villages: {self.villages}\n"
                f"Travel Times:\n{self.travel_times}"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(villages={(self.villages)})"

    def empty_solution(self):
        return Solution(self, [0], 0, 0, {i for i in range(1, self.num_villages)})

    def random_solution(self):
        sequence = self.villages.copy()
        shuffle(sequence)
        total_dist = 0
        for i in range(0, len(sequence)-1):
            total_dist += self.travel_times[i][i+1]

        return Solution(self, sequence, 3, 0, {i for i in range(1, self.num_villages)})

    @classmethod
    def from_textio(cls, f):
        num_villages = f.readline().strip()

        villages = []
        try:
            num_villages = int(num_villages)  # now works for negatives too
        except ValueError:
            raise Exception("Invalid instance")

        for i in range(num_villages):
            s = f.readline().strip().split(" ")
            row = []
            for val in s:
                try:
                    row.append(int(val))  # handles negative and positive integers
                except ValueError:
                    raise Exception("Invalid instance")
            villages.append(row)

        return cls(villages)
    
    def construction_neighbourhood(self):
        return AddNeighbourhood(self)

# ---------------------------------- Solution --------------------------------
@final
class Solution(SupportsLowerBoundIncrement, SupportsCopySolution, SupportsObjectiveValue, SupportsLowerBound):
    def __init__(self, problem, sequence, total_travel_time, accumulated_candle_length, not_visited_villages):
        self.problem = problem
        self.sequence = sequence
        self.not_visited_villages = not_visited_villages
        self.total_travel_time = total_travel_time
        self.accumulated_candle_length = accumulated_candle_length
        self.available_candle_length = get_available_candle_length(self.total_travel_time, [self.problem.villages[i] for i in self.not_visited_villages])

    def __str__(self):
        return (f"Sequence: {self.sequence}\n"
                f"Accumulated_candle_length: {self.accumulated_candle_length}\n"
                f"Total_travel_time: {self.total_travel_time}\n"
                f"Not_visited_villages: {self.not_visited_villages}\n")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"problem={repr(self.problem)}, "
            f"sequence={repr(self.sequence)}, "
            f"total_travel_time={self.total_travel_time}, "
            f"accumulated_candle_length={self.accumulated_candle_length}, "
            f"not_visited_villages={repr(self.not_visited_villages)})"
        )

    def __eq__(self, other):
        return (
            self.sequence == other.sequence and
            self.total_travel_time == other.total_travel_time and
            self.accumulated_candle_length == other.accumulated_candle_length and
            self.not_visited_villages == other.not_visited_villages
        )

    def to_textio(self, f) -> None:
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
        remaining = get_available_candle_length(self.total_travel_time, villages)
        return self.accumulated_candle_length + remaining

    def copy_solution(self):
        return Solution(self.problem, self.sequence.copy(), self.total_travel_time, self.accumulated_candle_length, self.not_visited_villages.copy() )

    def objective_value(self):
        return self.accumulated_candle_length

    def lower_bound(self):
        return -self.upper_bound()


# ------------------------------- Neighbourhood ------------------------------
@final
class AddNeighbourhood(SupportsMoves, SupportsRandomMove, SupportsRandomMovesWithoutReplacement):
    def __init__(self, problem):
        self.problem = problem

    def __repr__(self):
        return f"{self.__class__.__name__}(problem={repr(self.problem)})"

    def moves(self, solution):
        for i in solution.not_visited_villages.copy():
            # uncomment bellow to not add moves that goes to a city with burnt down candle
            # if get_candle_length(solution.total_travel_time + self.problem.travel_times[solution.sequence[-1]][i], self.problem.villages[i]) > 0:
            yield AddMove(self, solution.sequence[-1], i)

    def random_move(self, solution) :
        moves = list(self.moves(solution))  # reuse moves()
        return choice(moves) if moves else None

    def random_moves_without_replacement(self, solution):
        moves_gen = self.moves(solution)
        moves_dict = {}
        for idx, move in enumerate(moves_gen):
            moves_dict[idx] = move

        n = len(moves_dict)
        for i in sparse_fisher_yates_iter(n):
            yield moves_dict[i]



# class SwapNeighbourhood(SupportsMoves, SupportsRandomMove, SupportsRandomMovesWithoutReplacement):
#     def __init__(self, problem):
#         self.problem = problem
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(problem={repr(self.problem)})"
#
#     def moves(self, solution):
#         for i in solution.sequnce[1:]:
#             for j in solution.sequnce[1:]:
#                 if i < j:
#                     yield TwoOptMove(self, solution.sequence[-1], i)
#
#     def random_move(self, solution) :
#         moves = list(self.moves(solution))  # reuse moves()
#         return choice(moves) if moves else None
#
#     def random_moves_without_replacement(self, solution):
#         moves_gen = self.moves(solution)
#         moves_dict = {}
#         for idx, move in enumerate(moves_gen):
#             moves_dict[idx] = move
#
#         n = len(moves_dict)
#         for i in sparse_fisher_yates_iter(n):
#             yield moves_dict[i]


# ----------------------------------- Moves -----------------------------------
@final
class AddMove(SupportsApplyMove, SupportsLowerBoundIncrement):
    def __init__(self, neighbourhood, i, j):
        self.neighbourhood = neighbourhood
        self.i = i
        self.j = j

    def __str__(self):
        return f"AddMove: i={self.i}, j={self.j}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"neighbourhood={repr(self.neighbourhood)}, "
            f"i={self.i}, "
            f"j={self.j})"
        )

    def apply_move(self, solution):
        solution.sequence.append(self.j)
        solution.not_visited_villages.remove(self.j)
        solution.total_travel_time += solution.problem.travel_times[self.i][self.j]
        solution.accumulated_candle_length += get_candle_length(solution.total_travel_time, solution.problem.villages[self.j])

        solution.available_candle_length = get_available_candle_length(
            solution.total_travel_time,
            [solution.problem.villages[i] for i in solution.not_visited_villages]
        )
        return solution

    def upper_bound_increment(self, solution):
        new_total_travel_time = solution.total_travel_time + solution.problem.travel_times[self.i][self.j]

        delta_accumulated = get_candle_length(new_total_travel_time, solution.problem.villages[self.j])

        remaining_villages = [solution.problem.villages[i]
                              for i in solution.not_visited_villages
                              if i != self.j]

        new_remaining = get_available_candle_length(new_total_travel_time, remaining_villages)

        old_remaining = solution.available_candle_length

        delta_remaining = new_remaining - old_remaining
        return delta_accumulated + delta_remaining


    def lower_bound_increment(self, solution):
        return -self.upper_bound_increment(solution)

#
# @final
# class TwoOptMove(SupportsApplyMove, SupportsObjectiveValueIncrement):
#     def __init__(self, neighbourhood, i, j):
#         self.neighbourhood = neighbourhood
#         self.i = i
#         self.j = j
#
#     def __str__(self):
#         return f"AddMove: i={self.i}, j={self.j}"
#
#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}("
#             f"neighbourhood={repr(self.neighbourhood)}, "
#             f"i={self.i}, "
#             f"j={self.j})"
#         )
#
#     def apply_move(self, solution):
#         ...
#
#     def objective_value_increment(self, solution: Solution):
#         ...

# ------------------------------- Helpers ------------------------------

def sparse_fisher_yates_iter(n):
    p = dict()
    for i in range(n - 1, -1, -1):
        r = randrange(i + 1)
        yield p.get(r, r)
        if i != r:
            p[r] = p.get(i, i)


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
    from roar_net_api.algorithms import *
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=argparse.FileType('r'), default=sys.stdin)

    args = parser.parse_args()
    filename = args.input_file.name
    problem = Problem.from_textio(args.input_file)

    solution = greedy_construction(problem)

    file = open("output.txt", "w")
    solution.to_textio(file)
    file.close()

