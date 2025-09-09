from __future__ import annotations

from typing import final

from roar_net_api.operations import *



# ---------------------------------- Problem --------------------------------
@final
class Problem(SupportsEmptySolution, SupportsConstructionNeighbourhood):
    def __init__(self, villages):
        self.villages = villages
        self.distances = calc_distances(villages)


# ---------------------------------- Solution --------------------------------
@final
class Solution():
    def __init__(self):
        NotImplemented


# ------------------------------- Neighbourhood ------------------------------
@final
class AddNeighbourhood(SupportsMoves[Solution, "AddMove"]):
    def __init__(self):
        NotImplemented

# ----------------------------------- Moves -----------------------------------
@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
    def __init__(self):
        NotImplemented





## Helpers ##

def calc_distances(villages):
    NotImplemented