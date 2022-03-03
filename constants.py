import enum


class Phase(enum.Enum):
    VALIDATION = 0
    TEST = 1


class BackBone(enum.Enum):
    GCN = 0
    DGAT = 1


class ModelType(enum.Enum):
    GNN = 0
    MATRIX_FRACTION = 1
    DOT_PRODUCT = 2


class GNN(enum.Enum):
    MMGCN = 0
    GAEMDA = 1
    AGAEMD = 2


INF = 99999
