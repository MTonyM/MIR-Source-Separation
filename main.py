from torch import optim
from torch.autograd import Variable
import numpy as np
from utils.audio_op import *
from options import *
from datasets.dataset import get_dataloader
import importlib
import math
from utils.progbar import progbar
import time
from utils.evaluate import *
import datetime


#############################
# Fix the error ' AttributeError: Can't get attribute '_rebuild_tensor_v2' on <module 'torch._utils' from '/root/miniconda3/lib/python3.6/site-packages/torch/_utils.py'> '  by fix version compatibility 
#Doesn't work so stop check point loading for now
# import torch._utils
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
#     print('Tried fix')
###########################



####################################################
models = importlib.import_module('models.init')
criterions = importlib.import_module('criterions.init')
checkpoints = importlib.import_module('checkpoints')
####################################################


args = get_args()
num_epoches = args.num_epoches
learning_rate = args.learning_rate
learning_rate = 1e-3
batch_size = args.batch_size
trainLoader, testLoader = get_dataloader(args)    
# print('=> Checking checkpoints')
# checkpoint = checkpoints.load(args)
checkpoint = None
# create model
model, optimState = models.setup(args, checkpoint)
if args.GPU:
    model = model.cuda()
criterion = criterions.setup(args, checkpoint, model)
# ************************* (need changed)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(args.momentum, 0.999), eps=1e-8,
                       weight_decay=args.weightDecay)

if optimState != None:
    optimizer.load_state_dict(optimState)
# **************************

nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
result_dir = '../results/' + args.dataset + '_' + args.model+ '_' + nowTime
bestLoss = 0.0
epoch_start = 0
if checkpoint != None:
    epoch_start = checkpoint['epoch'] + 1
    bestLoss = checkpoint['loss']
    print('Previous loss: \033[1;36m%1.4f\033[0m' % bestLoss)

    

if not os.path.exists(result_dir):
    os.makedirs(result_dir)    
    
logger = {'train': open(os.path.join(result_dir, 'train.log'), 'a+'),
       'val': open(os.path.join(result_dir, 'val.log'), 'a+'),
       'test': open(os.path.join(result_dir, 'test.log'), 'a+'),
       'test_mask': open(os.path.join(result_dir, 'test_mask.log'), 'a+'),
       'info':open(os.path.join(result_dir, 'info.log'), 'a+')}
# Initial Log File 
init_log = args.dataset + nowTime + '\n'  
logger['train'].write(init_log)
logger['test'].write(init_log)
logger['val'].write(init_log)
logger['info'].write(init_log)
logger['test_mask'].write(init_log)
info_log = nowTime + '\n' + 'DataSet: ' + args.dataset + '\n' + 'Model: ' + args.model + '\n' + 'Train ratio: ' + str(args.train_ratio) + '\n'      
logger['info'].write(info_log)




scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.LRDParam, gamma=0.1, last_epoch=epoch_start - 1)

# RNN
h_state = None
avg_loss = math.inf
best_loss = math.inf






for epoch in range(epoch_start, num_epoches):
    log = 'epoch Num:' + str(epoch)+'\n'
    print(log)
    logger['train'].write(log)
    logger['test'].write(log)
    logger['test_mask'].write(log)
    epoch_file = os.path.join( result_dir , 'epoch_%d' % epoch)
    
    if not os.path.exists(epoch_file):
        os.makedirs(epoch_file)
    pbar = progbar(len(trainLoader)*batch_size, width=50)
    running_loss = 0.0
    for i, (inputs, target, phase) in enumerate(trainLoader):
#         print(i,'done')
        if args.debug and i>1:
            break
        batch_file = epoch_file + '/train_batch_%d' % i
        if not os.path.exists(batch_file):
            os.makedirs(batch_file)
        model.train()
        # Variable 
        if args.GPU:
            inputs = Variable(inputs).cuda()
            target = Variable(target).cuda()
        else:
            inputs = Variable(inputs)
            target = Variable(target)
        out, h_state = model(inputs, h_state)
#         import pdb
#         pdb.set_trace()
        h_v = []
        if isinstance(h_state, tuple):
            for h_i in h_state:
                h_v.append(Variable(h_i.data))
            h_state = tuple(h_v)
        else:
            h_state = Variable(h_state.data) # useless.
        _, inputs, _ = torch.split(inputs, 513 , dim=2)
        # loss
        song_mag_out, voice_mag_out = torch.split(out, 513, dim=2)
        song_mag_tar, voice_mag_tar = torch.split(target, 513 , dim=2)

        # Apply mask 
        song_mag_mask, voice_mag_mask = soft_mask(song_mag_out, voice_mag_out, inputs)

        # 1 Disc_loss
        disc_lambda = 0.05
        
        # 1_1 loss without mask
        #         loss = 0.5 * criterion(song_mag_out,song_mag_tar) + 0.5 * criterion(voice_mag_out,voice_mag_tar) - disc_lambda * ( criterion(song_mag_out,voice_mag_tar) + criterion(voice_mag_out,song_mag_tar))
        
        
        # 1_2 loss with  mask
        loss = 0.5 * criterion(song_mag_mask, song_mag_tar) + 0.5 * criterion(voice_mag_mask,voice_mag_tar) - disc_lambda * (
                       criterion(song_mag_mask, voice_mag_tar) + criterion(voice_mag_mask, song_mag_tar))

        # 2  Standard_loss
        #             print('compute loss')
        #             loss = criterion(out,target)

        running_loss += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = running_loss / (batch_size * i + 1e-4) 
#         avg_loss = float(loss) / (batch_size + 1e-4) # change to moving average.
    
        phase = phase.numpy()
#         print(type(song_mag_out.cpu().detach().data))
        song_spec_out = merge_mag_phase(song_mag_out.cpu().detach().data.numpy(), phase)
        voice_spec_out = merge_mag_phase(voice_mag_out.cpu().detach().data.numpy(), phase)
        song_spec_mask = merge_mag_phase(song_mag_mask.cpu().detach().data.numpy(), phase)
        voice_spec_mask = merge_mag_phase(voice_mag_mask.cpu().detach().data.numpy(), phase)

        song_spec_tar = merge_mag_phase(song_mag_tar.cpu().detach().data.numpy(), phase)
        voice_spec_tar = merge_mag_phase(voice_mag_tar.cpu().detach().data.numpy(), phase)

        for batch_item in range(batch_size):
            song_audio = create_audio_from_spectrogram(song_spec_out[batch_item, :, :], args)
            voice_audio = create_audio_from_spectrogram(voice_spec_out[batch_item, :, :], args)
            song_audio_mask = create_audio_from_spectrogram(song_spec_mask[batch_item, :, :], args)
            voice_audio_mask = create_audio_from_spectrogram(voice_spec_mask[batch_item, :, :], args)

            song_audio_tar = create_audio_from_spectrogram(song_spec_tar[batch_item, :, :], args)
            voice_audio_tar = create_audio_from_spectrogram(voice_spec_tar[batch_item, :, :], args)

#             song_audio_tar = song_audio_tar[:, np.newaxis]
#             voice_audio_tar = voice_audio_tar[:, np.newaxis]
            mixed_audio = 0.5 * voice_audio_tar + 0.5 * song_audio_tar 
            writeWav(os.path.join(batch_file, '%d_%d_song_tar.wav' % (i, batch_item)),
                     args.sample_rate, song_audio_tar)
            writeWav(os.path.join(batch_file, '%d_%d_voice_tar.wav' % (i, batch_item)),
                     args.sample_rate, voice_audio_tar)        
        
            writeWav(os.path.join(batch_file, '%d_%d_song.wav' % (i, batch_item)),
                     args.sample_rate, song_audio)
            writeWav(os.path.join(batch_file, '%d_%d_voice.wav' % (i, batch_item)),
                     args.sample_rate, voice_audio)

            writeWav(os.path.join(batch_file, '%d_%d_song_mask.wav' % (i, batch_item)),
                     args.sample_rate, song_audio_mask)
            writeWav(os.path.join(batch_file, '%d_%d_voice_mask.wav' % (i, batch_item)),
                     args.sample_rate, voice_audio_mask)
            
            pbar.update(i*batch_size + batch_item + 1, [('avg_loss', float(loss) / (batch_size + 1e-4)),])

                
        del loss
        del song_mag_out
        del voice_mag_out
        del song_mag_tar
        del voice_mag_tar 
    log = '[{}/{}] Loss: {:.6f} \n Saved to {}\n'.format(epoch + 1, num_epoches, avg_loss, batch_file)
    logger['train'].write(log)
    print(log)

#     =======================================================
#                 soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_audio_tar,
#                                                                    voice_audio_tar, song_audio, voice_audio)
#                 log = '=> done write :' + '%d_%d' % (i, batch_item) + "|" +str(soft_gnsdr[0])+ "|" + \
#                         str(soft_gnsdr[1])+"|" + str(soft_gsir[0])+ "|" +str(soft_gsir[1])+"|" + \
#                         str(soft_gsar[0])+"|"+str(soft_gsar[1])+"\n"
#     ====================================================

    """ Test now """
    running_loss = 0.0
    pbar_val = progbar(len(testLoader) * batch_size, width=50)
    for i, (inputs, target, phase) in enumerate(testLoader):
        batch_file = epoch_file + '/test_batch_%d' % i

        if not os.path.exists(batch_file):
            os.makedirs(batch_file)
        model.eval()
        
        # Variable
        if args.GPU:
            inputs = Variable(inputs).cuda()
            target = Variable(target).cuda()
        else:
            inputs = Variable(inputs)
            target = Variable(target)

        out, h_state = model(inputs, h_state)
        h_v = []
        if isinstance(h_state, tuple):
            for h_i in h_state:
                h_v.append(Variable(h_i.data))
            h_state = tuple(h_v)
        else:
            h_state = Variable(h_state.data) # useless.
            
            
        _, inputs, _ = torch.split(inputs, 513 , dim=2)
        # loss
        song_mag_out, voice_mag_out = torch.split(out, 513 , dim=2)
        song_mag_tar, voice_mag_tar = torch.split(target, 513, dim=2)

        # Apply mask
        song_mag_mask, voice_mag_mask = soft_mask(song_mag_out, voice_mag_out, inputs)

        # 1 Disc_loss
        disc_lambda = 0.05
        loss = 0.5 * criterion(song_mag_mask, song_mag_tar) + 0.5 * criterion(voice_mag_mask,voice_mag_tar) - disc_lambda * (
                       criterion(song_mag_mask, voice_mag_tar) + criterion(voice_mag_mask, song_mag_tar))


        # 2  Standard_loss
        #             print('compute loss')
        #             loss = criterion(out,target)

        running_loss += float(loss)

        avg_loss = running_loss / (batch_size * i + 1e-4)
#         avg_loss = float(loss) / (batch_size + 1e-4)
        phase = phase.numpy()

        song_spec_out = merge_mag_phase(song_mag_out.cpu().detach().data.numpy(), phase)
        voice_spec_out = merge_mag_phase(voice_mag_out.cpu().detach().data.numpy(), phase)
        song_spec_mask = merge_mag_phase(song_mag_mask.cpu().detach().data.numpy(), phase)
        voice_spec_mask = merge_mag_phase(voice_mag_mask.cpu().detach().data.numpy(), phase)

        song_spec_tar = merge_mag_phase(song_mag_tar.cpu().detach().data.numpy(), phase)
        voice_spec_tar = merge_mag_phase(voice_mag_tar.cpu().detach().data.numpy(), phase)

        for batch_item in range(batch_size) :
            if args.debug and batch_item>1:
                break
            song_audio = create_audio_from_spectrogram(song_spec_out[batch_item, :, :], args)
            voice_audio = create_audio_from_spectrogram(voice_spec_out[batch_item, :, :], args)
            song_audio_mask = create_audio_from_spectrogram(song_spec_mask[batch_item, :, :], args)
            voice_audio_mask = create_audio_from_spectrogram(voice_spec_mask[batch_item, :, :], args)

            song_audio_tar = create_audio_from_spectrogram(song_spec_tar[batch_item, :, :], args)
            voice_audio_tar = create_audio_from_spectrogram(voice_spec_tar[batch_item, :, :], args)

#             song_audio_tar = song_audio_tar[:, np.newaxis]
#             voice_audio_tar = voice_audio_tar[:, np.newaxis]
            mixed_audio = 0.5 * voice_audio_tar + 0.5 * song_audio_tar

            writeWav(os.path.join(batch_file, '%d_%d_song_tar.wav' % (i, batch_item)),
                     args.sample_rate, song_audio_tar)
            writeWav(os.path.join(batch_file, '%d_%d_voice_tar.wav' % (i, batch_item)),
                     args.sample_rate, voice_audio_tar)        
        
            writeWav(os.path.join(batch_file, '%d_%d_song.wav' % (i, batch_item)),
                     args.sample_rate, song_audio)
            writeWav(os.path.join(batch_file, '%d_%d_voice.wav' % (i, batch_item)),
                     args.sample_rate, voice_audio)

            writeWav(os.path.join(batch_file, '%d_%d_song_mask.wav' % (i, batch_item)),
                     args.sample_rate, song_audio_mask)
            writeWav(os.path.join(batch_file, '%d_%d_voice_mask.wav' % (i, batch_item)),
                     args.sample_rate, voice_audio_mask)

            pbar_val.update(i * batch_size + batch_item + 1, [])
            soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_audio_tar, voice_audio_tar, song_audio_mask, voice_audio_mask)
            
            # with mask
            log = '=> done write :' + '%d_%d' % (i, batch_item) + "| GNSDR 0: " +str(soft_gnsdr[0])+ " | GNSDR 1: " + \
                    str(soft_gnsdr[1])+" | GSIR 0: " + str(soft_gsir[0])+ " | GSIR 1: " +str(soft_gsir[1])+" | GSAR 0: " + \
                    str(soft_gsar[0])+" | GSAR 1: "+str(soft_gsar[1]) + '\n'
            logger['test_mask'].write(log)
            # without mask usually disabled
#             soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_audio_tar, voice_audio_tar, song_audio, voice_audio)
#             log = '=> done write :' + '%d_%d' % (i, batch_item) + "| GNSDR 0: " +str(soft_gnsdr[0])+ " | GNSDR 1: " + \
#                     str(soft_gnsdr[1])+" | GSIR 0: " + str(soft_gsir[0])+ " | GSIR 1: " +str(soft_gsir[1])+" | GSAR 0: " + \
#                     str(soft_gsar[0])+" | GSAR 1: "+str(soft_gsar[1]) + '\n'
#             logger['test'].write(log)
            
        del loss
        del song_mag_out
        del voice_mag_out
        del song_mag_tar
        del voice_mag_tar 
    log = '[{}/{}] Loss: {:.6f}\n'.format(epoch + 1, num_epoches, avg_loss)
    logger['val'].write(log)
    print(log)

    bestModel = False
    if avg_loss < best_loss:
        bestModel = True
        best_loss = avg_loss
        print(' * Best model: \033[1;36m%1.4f\033[0m * ' % best_loss)

    scheduler.step()

    checkpoints.save(epoch, model, criterion, optimizer, bestModel, avg_loss, args)
