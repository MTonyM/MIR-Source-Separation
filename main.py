import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os

from scipy.io import wavfile
from audio_op import *
from config import *
from evaluate import bss_eval_global
from dataset import get_dataloader
from models.rnn import get_model
import time
import importlib
import math
####################################################
models = importlib.import_module('models.init')
criterions = importlib.import_module('criterions.init')
checkpoints = importlib.import_module('checkpoints')
####################################################


args = get_args()
num_epoches = args.num_epoches
learning_rate = args.learning_rate
batch_size = args.batch_size
result_wav_dir = './results'

train_dataset = get_dataloader(args)
trainLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


print('=> Checking checkpoints')
checkpoint = checkpoints.load(args)

# create model
model, optimState = models.setup(args, checkpoint)
if args.GPU:
    model = model.cuda()
criterion = criterions.setup(args, checkpoint, model)
# ************************* (need changed)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if optimState != None:
    optimizer.load_state_dict(optimState)
# **************************
if checkpoint != None:
    startEpoch = checkpoint['epoch'] + 1
    bestLoss = checkpoint['loss']
    print('Previous loss: \033[1;36m%1.4f\033[0m' % bestLoss)

logger = {'train' : open(os.path.join(args.resume, 'train.log'), 'a+'),
            'val' : open(os.path.join(args.resume, 'test.log'), 'a+')}    
    
# RNN
h_state = None
for epoch in range(num_epoches):
    epoch_file = './results/epoch_%d'% epoch
    if not os.path.exists(epoch_file):
        os.makedirs(epoch_file)
    best_loss = math.inf
    running_loss = 0.0
    for i, (inputs, target, phase) in enumerate(trainLoader):
        batch_file = epoch_file + '/batch_%d'%i 
        if not os.path.exists(batch_file):
            os.makedirs(batch_file)
        model.train()
        t0=time.time()
        # Variable
        if args.GPU:
            inputs= Variable(inputs).cuda()
            target = Variable(target).cuda()
        else:
            inputs= Variable(inputs)
            target = Variable(target)

        try:   
            out, h_state= model(inputs, h_state)
        except:
            print("h_state is error")
            break


        h_state = Variable(h_state.data)

        pre_win,inputs,next_win = torch.split(inputs, (513,513,513), dim = 2)
        # loss
        song_mag_out, voice_mag_out = torch.split(out, (513,513), dim = 2)
        song_mag_tar, voice_mag_tar = torch.split(target, (513,513), dim = 2)

        # Apply mask 
        song_mag_mask, voice_mag_mask = soft_mask(song_mag_out, voice_mag_out, inputs)

        # 1 Disc_loss
        disc_lambda = 0.05
        # 1_1 loss without mask
#         loss = 0.5 * criterion(song_mag_out,song_mag_tar) + 0.5 * criterion(voice_mag_out,voice_mag_tar) - disc_lambda * ( criterion(song_mag_out,voice_mag_tar) + criterion(voice_mag_out,song_mag_tar))
        # 1_2 loss with  mask
        loss = 0.5 * criterion(song_mag_mask,song_mag_tar) + 0.5 * criterion(voice_mag_mask,voice_mag_tar) - disc_lambda * ( criterion(song_mag_mask,voice_mag_tar) + criterion(voice_mag_mask,song_mag_tar))
    
    
        # 2  Standard_loss
#             print('compute loss')
#             loss = criterion(out,target)


        running_loss += loss
    
        t1 = time.time()
        print('input-loss compute time:' ,t1-t0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = running_loss / (batch_size * i + 1e-4)
        log = '[{}/{}] Loss: {:.6f}\n'.format(epoch + 1, num_epoches, avg_loss)
        logger['train'].write(log)
        print(log)
        t2 = time.time()
        print('backward compute time:' ,t2-t1)
        
        phase = phase.numpy()
        mixed_spec = merge_mag_phase(inputs.data.cpu().numpy(), phase)

        song_spec_out = merge_mag_phase(song_mag_out.cpu().detach().numpy(),phase)
        voice_spec_out = merge_mag_phase(voice_mag_out.cpu().detach().numpy(),phase)
        song_spec_mask = merge_mag_phase(song_mag_mask.cpu().detach().numpy(),phase)
        voice_spec_mask = merge_mag_phase(voice_mag_mask.cpu().detach().numpy(),phase)            

        song_spec_tar = merge_mag_phase(song_mag_tar.cpu().detach().numpy(), phase)
        voice_spec_tar = merge_mag_phase(voice_mag_tar.cpu().detach().numpy(), phase)
        t3 = time.time()
        print('combine phase compute time:' ,t3-t2)

        for batch_item in range(batch_size):
            t3_1 = time.time()
            song_audio = create_audio_from_spectrogram(song_spec_out[batch_item,:,:], args)
            voice_audio = create_audio_from_spectrogram(voice_spec_out[batch_item,:,:], args)
            song_audio_mask = create_audio_from_spectrogram(song_spec_mask[batch_item,:,:], args)
            voice_audio_mask = create_audio_from_spectrogram(voice_spec_mask[batch_item,:,:], args)                

            song_audio_tar = create_audio_from_spectrogram(song_spec_tar[batch_item,:,:], args)
            voice_audio_tar = create_audio_from_spectrogram(voice_spec_tar[batch_item,:,:], args) 
            

            song_audio_tar = song_audio_tar[:,np.newaxis]
            voice_audio_tar =voice_audio_tar[:,np.newaxis]
            mixed_audio = np.concatenate([voice_audio_tar, song_audio_tar], axis=1) 
            
            
            t3_2 = time.time()
            print('creat audio time:',t3_2-t3_1)
            
            writeWav(os.path.join(batch_file, '%d_%d_song.wav' % (i, batch_item)), 
                     args.sample_rate, song_audio)
            writeWav(os.path.join(batch_file, '%d_%d_voice.wav' % (i, batch_item)), 
                     args.sample_rate, voice_audio)
            writeWav(os.path.join(batch_file, '%d_%d_mixed.wav' % (i, batch_item)), 
                     args.sample_rate, mixed_audio)
            writeWav(os.path.join(batch_file, '%d_%d_song_mask.wav' % (i, batch_item)), 
                     args.sample_rate, song_audio_mask)
            writeWav(os.path.join(batch_file, '%d_%d_voice_mask.wav' % (i, batch_item)), 
                     args.sample_rate, voice_audio_mask)
            
            
            t3_3 = time.time()
            print('writeWav time:',t3_3-t3_2)
            
#=======================================================
#             soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_audio_tar, 
#                                                                voice_audio_tar, song_audio, voice_audio)
#             t3_4 = time.time()
#             print('Evaluation time:',t3_4-t3_3)
#             log = '=> done write :' + '%d_%d' % (i, batch_item) + "|" +str(soft_gnsdr[0])+ "|" + \
#                     str(soft_gnsdr[1])+"|" + str(soft_gsir[0])+ "|" +str(soft_gsir[1])+"|" + \
#                     str(soft_gsar[0])+"|"+str(soft_gsar[1])+"\n"

#====================================================
            log='=> done write :' + '%d_%d' % (i, batch_item) + " | "+ 'avg_loss: %d' % (avg_loss)
            print(log)
            logger['train'].write(log)
            if args.debug:
                break
        
        t4 = time.time()
        print('write_file compute time:' ,t4-t3)
        
        if args.debug:
            break
        
    bestModel = False
    if avg_loss < best_loss:
        bestModel = True
        best_loss = avg_loss
        print(' * Best model: \033[1;36m%1.4f\033[0m * ' % best_loss)
    
    checkpoints.save(epoch, model, criterion, optimizer, bestModel, avg_loss ,args)        
    
print(' * Finished Err: \033[1;36m%1.4f\033[0m * ' % bestLoss)
