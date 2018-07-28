import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel.data_parallel import DataParallel
import importlib

def setup(args, checkpoint):
    model = None
    optimState = None
    
    print('=> Creating model from file: models/' + args.model + '.py')
    models = importlib.import_module('models.' + args.model)
    model = models.get_model(args)

    if checkpoint != None:
        modelPath = os.path.join(args.resume, checkpoint['modelFile'])
        assert os.path.exists(modelPath), '=> WARNING: Saved model state not found: ' + modelPath
        print('=> Resuming model state from ' + modelPath)
        model.load_state_dict(torch.load(modelPath))
        optimPath = os.path.join(args.resume, checkpoint['optimFile'])
        assert os.path.exists(optimPath), '=> WARNING: Saved optimState not found: ' + optimPath
        print('=> Resuming optimizer state from ' + optimPath)
        optimState = torch.load(optimPath)

    if isinstance(model, nn.DataParallel):
        model = model.get(0)
    
    if args.GPU:
        model = model.cuda()
        
    return model, optimState
