import torch.nn as nn


def initCriterion(criterion, model):
    pass


def createCriterion(args, model):
    criterion = nn.MSELoss()
    return criterion
