from random import choice, shuffle, random, randrange
from typing import final, Optional, Union

from roar_net_api.operations import *


# ---------------------------------- Problem --------------------------------
@final
class Problem(SupportsEmptySolution, SupportsConstructionNeighbourhood, SupportsRandomSolution, SupportsLocalNeighbourhood):
    def __init__(self, villages):
        self.villages = villages
        self.num_villages = len(self.villages)

    def __eq__(self, other):
        return (
                self.villages == other.villages
                and self.num_villages == other.num_villages
                #and self.travel_times == other.travel_times
        )

    def __str__(self):
        return (
                f"Village Network with {self.num_villages} villages\n"
                f"Villages: {self.villages}"
                #f"Travel Times:\n{self.travel_times}"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(villages={(self.villages)})"

    def empty_solution(self):
        return Solution(self, [0], 0, 0, {i for i in range(1, self.num_villages)})

    def random_solution(self):
        sequence = [i for i in range(1, len(self.villages))]
        shuffle(sequence)
        sequence.insert(0, 0)
        total_travel_time = 0
        for i in range(0, len(sequence)-1):
            total_travel_time += calc_distance(self.villages[i], self.villages[i + 1])                       #self.travel_times[i][i+1]

        return Solution(self, sequence, total_travel_time, calc_accumulated_candle_length(sequence, self), {})

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

    def local_neighbourhood(self):
        return InsertNeighbourhood(self)

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


class InsertNeighbourhood(SupportsMoves, SupportsRandomMove, SupportsRandomMovesWithoutReplacement):
    def __init__(self, problem):
        self.problem = problem

    def __repr__(self):
        return f"{self.__class__.__name__}(problem={repr(self.problem)})"

    def moves(self, solution):
        for i in range(1, len(solution.sequence)):
            for j in range(1, len(solution.sequence)):
                if i != j and j != 0:
                    yield InsertMove(self, i, j)

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


@final
class InsertMove(SupportsApplyMove, SupportsObjectiveValueIncrement):
    def __init__(self, neighbourhood, i, j):
        self.neighbourhood = neighbourhood
        self.i = i      # Index of city in solution.sequnce
        self.j = j      # Index to insert city in solution.sequnce

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
        solution.total_travel_time += calc_delta_travel_time(self, solution)
        solution.accumulated_candle_length += calc_delta_candle_length(self, solution)
        v = solution.sequence.pop(self.i)
        solution.sequence.insert(self.j, v)
        return solution

    def objective_value_increment(self, solution: Solution):
        return calc_delta_candle_length(self, solution)

# ------------------------------- Helpers ------------------------------

def sparse_fisher_yates_iter(n):
    p = dict()
    for i in range(n - 1, -1, -1):
        r = randrange(i + 1)
        yield p.get(r, r)
        if i != r:
            p[r] = p.get(i, i)

def calc_delta_travel_time(move, solution):
    seq = solution.sequence
    tt = solution.problem.travel_times
    n = len(seq)
    i, j = move.i, move.j

    v = seq[i]

    new_seq = seq[:i] + seq[i+1:]
    new_seq = new_seq[:j] + [v] + new_seq[j:]

    old_tt = 0
    new_tt = 0
    for k in range(n - 1):
        old_tt += tt[seq[k]][seq[k+1]]
        new_tt += tt[new_seq[k]][new_seq[k+1]]

    return new_tt - old_tt

def calc_delta_candle_length(move, solution):

    new_seq = solution.sequence[: move.i] + solution.sequence[ move.i+1:]
    new_seq = new_seq[: move.j] + [solution.sequence[move.i]] + new_seq[ move.j:]

    return calc_accumulated_candle_length(new_seq, solution.problem) - solution.accumulated_candle_length


def calc_distance(vil1, vil2):
    return abs(vil1[0] - vil2[0]) + (vil1[1] - vil2[1])

def get_candle_length(travel_time, village):
    return max(0, (village[2] - village[3]*travel_time))

def calc_accumulated_candle_length(sequence, problem):
    accumulated_candle_length = 0
    travel_time = 0
    for idx in range(1,len(sequence)):
        travel_time += calc_distance(problem.villages[sequence[idx-1]], problem.villages[sequence[idx]])          #problem.travel_times[sequence[idx-1]][sequence[idx]]
        accumulated_candle_length += get_candle_length(travel_time, problem.villages[sequence[idx]])
    return accumulated_candle_length

def calc_total_travel_time(sequence, problem):
    total_travel_time = 0
    for i in range(0, len(sequence) - 1):
        total_travel_time += calc_distance(problem.villages[i], problem.villages[i + 1])                                                                                       #problem.travel_times[i][i + 1]
    return total_travel_time


def get_available_candle_length(travel_time, villages):
    """ Return the sum of the candle lengths of the not_visited_villages"""
    return sum([get_candle_length(travel_time, village) for village in villages])

if __name__ == "__main__":
    from roar_net_api.algorithms import *
    import argparse
    import sys


    
    """
    file = open("candle10000.txt", "w")
    import random
    n = 10000
    file.write(str(n) + "\n")
    file.write("0 0\n")
    for i in range(n - 1):
        file.write(str(random.randint(1, 90)) + " " + str(random.randint(1, 90)) + " " + str(random.randint(300, 985)) + " " + str(random.randint(2, 10)) + "\n")
    file.write(str(random.randint(1, 90)) + " " + str(random.randint(1, 90)) + " " + str(random.randint(300, 985)) + " " + str(random.randint(2, 10)))
    file.close()
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=argparse.FileType('r'), default=sys.stdin)

    args = parser.parse_args()
    filename = args.input_file.name
    problem = Problem.from_textio(args.input_file)


    local_neigbourhood = problem.local_neighbourhood()
    solution = problem.random_solution()
    solution = rls(problem, solution, 1)
    print(solution)
    
    # moves = local_neigbourhood.moves(solution)
    # for move in moves:
    #     s = solution.copy_solution()
    #     move.apply_move(s)
    #     print(f"move: {move}")
    #     print(f"Soltion after move:")
    #     print(s)
    #     print()

    # rand_moves_wr = local_neigbourhood.moves(solution)
    # for move in rand_moves_wr:
    #     s = solution.copy_solution()
    #     move.apply_move(s)
    #     print(f"move: {move}")
    #     print(f"Soltion after move:")
    #     print(s)
    #     print()

    # file = open("output.txt", "w")
    # solution.to_textio(file)
    # file.close()

