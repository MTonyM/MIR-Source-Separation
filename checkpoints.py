import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
import torch.nn as nn


def latest(args):
    latestPath = os.path.join(args.resume, 'latest.pth.tar')
    if not os.path.exists(latestPath):
        return None
    print('=> Loading the latest checkpoint ' + latestPath)
    return torch.load(latestPath)


def best(args):
    bestPath = os.path.join(args.resume, 'best.pth.tar')
    if not os.path.exists(bestPath):
        return None
    print('=> Loading the best checkpoint ' + bestPath)
    return torch.load(bestPath)


def load(args):
    epoch = args.epochNum
    if epoch == 0:
        return None
    elif epoch == -1:
        return latest(args)
    elif epoch == -2:
        return best(args)
    else:
        modelFile = 'model_' + str(epoch) + '.pth.tar'
        criterionFile = 'criterion_' + str(epoch) + '.pth.tar'
        optimFile = 'optimState_' + str(epoch) + '.pth.tar'
        loaded = {'epoch': epoch, 'modelFile': modelFile, 'criterionFile': criterionFile, 'optimFile': optimFile}
        return loaded


def save(epoch, model, criterion, optimizer, bestModel, loss, args):
    if isinstance(model, nn.DataParallel):
        model = model.get(0)

    modelFile = 'model_' + str(epoch) + '.pth.tar'
    criterionFile = 'criterion_' + str(epoch) + '.pth.tar'
    optimFile = 'optimState_' + str(epoch) + '.pth.tar'

    if bestModel or (epoch % args.saveEpoch == 0):
        torch.save(model.state_dict(), os.path.join(args.resume, modelFile))
        torch.save(criterion, os.path.join(args.resume, criterionFile))
        torch.save(optimizer.state_dict(), os.path.join(args.resume, optimFile))
        info = {'epoch': epoch, 'modelFile': modelFile, 'criterionFile': criterionFile, 'optimFile': optimFile,
                'loss': loss}
        torch.save(info, os.path.join(args.resume, 'latest.pth.tar'))

    if bestModel:
        info = {'epoch': epoch, 'modelFile': modelFile, 'criterionFile': criterionFile, 'optimFile': optimFile,
                'loss': loss}
        torch.save(info, os.path.join(args.resume, 'best.pth.tar'))
        torch.save(model.state_dict(), os.path.join(args.resume, 'model_best.pth.tar'))
