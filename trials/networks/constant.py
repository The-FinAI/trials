from enum import IntEnum

FLOAT_MAX = 5e4


class Action(IntEnum):
    long = 0  # long A, short B
    short = 1  # short A, long B
    close = 2  # close position


class PositionState(IntEnum):
    long = 0  # long A, short B
    short = 1  # long B, short A
    bear = 2  # bear position
