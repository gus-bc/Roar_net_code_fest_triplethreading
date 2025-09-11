import unittest
from Candle_Race import Problem, get_candle_length, Solution, InsertMove, calc_delta_travel_time


class TestDeltaTravelTime(unittest.TestCase):
    def setUp(self):
        # Small problem instance
        self.villages = [
            [0, 0],
            [16, 25, 464, 2],
            [10, 34, 696, 6],
            [28, 17, 302, 5],
            [19, 57, 523, 10]
        ]
        self.problem = Problem(self.villages)
        self.not_visited_villages = set()
        self.sequence = [0, 1, 2, 3, 4]
        self.total_travel_time = sum(
            self.problem.travel_times[self.sequence[i]][self.sequence[i + 1]]
            for i in range(len(self.sequence) - 1)
        )
        self.accumulated_candle_length = sum(
            get_candle_length(self.total_travel_time, v)
            for v in self.villages[1:]
        )
        self.solution = Solution(
            self.problem,
            self.sequence.copy(),
            self.total_travel_time,
            self.accumulated_candle_length,
            self.not_visited_villages
        )

    def _manual_delta(self, move):
        seq = self.sequence.copy()
        original_tt = sum(self.problem.travel_times[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))
        village = seq.pop(move.i)
        seq.insert(move.j, village)
        new_tt = sum(self.problem.travel_times[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))
        return new_tt - original_tt

    def _test_move(self, village_index, j):
        move = InsertMove(None, village_index, j)
        delta_tt = calc_delta_travel_time(move, self.solution)
        expected_delta = self._manual_delta(move)
        self.assertEqual(delta_tt, expected_delta,
                         f"Delta travel time mismatch for move {village_index}->{j}: {delta_tt} != {expected_delta}")

    def test_insert_end(self):
        self._test_move(1, 4)

    def test_insert_same_position(self):
        self._test_move(2, 2)

    def test_insert_adjacent_before(self):
        self._test_move(4, 3)


    def test_minimal_sequence(self):
        # Only two villages
        seq = [0, 1]
        total_tt = sum(self.problem.travel_times[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))
        sol = Solution(self.problem, seq.copy(), total_tt, 0, set())
        move = InsertMove(None, 1, 0)
        delta_tt = calc_delta_travel_time(move, sol)
        village = seq.pop(1)
        seq.insert(0, village)
        expected_delta = sum(self.problem.travel_times[seq[i]][seq[i + 1]] for i in range(len(seq) - 1)) - total_tt
        self.assertEqual(delta_tt, expected_delta)


if __name__ == "__main__":
    unittest.main()
