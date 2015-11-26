import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import importlib

def setup(args, checkpoint, model):
    criterion = None
    criterionHandler = importlib.import_module('models.' + args.model + '-criterion')

    if checkpoint != None:
        criterionPath = os.path.join(args.resume, checkpoint['criterionFile'])
        assert os.path.exists(criterionPath), '=> WARNING: Saved criterion not found: ' + criterionPath
        print('=> Resuming criterion from ' + criterionPath)
        criterion = torch.load(criterionPath)
        criterionHandler.initCriterion(criterion, model)
    else:
        print('=> Creating criterion from file: models/' + args.model + '-criterion.py')
        criterion = criterionHandler.createCriterion(args, model)

    if args.GPU:
        criterion = criterion.cuda()

    return criterion
