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

####################################################
models = importlib.import_module('models.init')
criterions = importlib.import_module('criterions.init')
checkpoints = importlib.import_module('checkpoints')
####################################################


args = get_args()
num_epoches = args.num_epoches
learning_rate = args.learning_rate
batch_size = args.batch_size
trainLoader, testLoader = get_dataloader(args)
if args.new:
    checkpoint = None
else:    
    print('=> Checking checkpoints')
    checkpoint = checkpoints.load(args)

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
bestLoss = 0.0
epoch_start = 0
if checkpoint != None:
    epoch_start = checkpoint['epoch'] + 1
    bestLoss = checkpoint['loss']
    print('Previous loss: \033[1;36m%1.4f\033[0m' % bestLoss)

logger = {'train': open(os.path.join(args.resume, 'train.log'), 'a+'),
          'val': open(os.path.join(args.resume, 'test.log'), 'a+')}

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.LRDParam, gamma=0.1, last_epoch=epoch_start - 1)

# RNN
h_state = None
avg_loss = math.inf
best_loss = math.inf

for epoch in range(epoch_start, num_epoches):
    epoch_file = os.path.join('../results_'+ args.dataset,'epoch_%d' % epoch)
    if not os.path.exists(epoch_file):
        os.makedirs(epoch_file)
    pbar = progbar(len(trainLoader)*batch_size, width=50)
    running_loss = 0.0
    for i, (inputs, target, phase) in enumerate(trainLoader):
#         print(i,'done')
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
        h_state = Variable(h_state.data)

        _, inputs, _ = torch.split(inputs, (513, 513, 513), dim=2)
        # loss
        song_mag_out, voice_mag_out = torch.split(out, (513, 513), dim=2)
        song_mag_tar, voice_mag_tar = torch.split(target, (513, 513), dim=2)

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

        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = running_loss / (batch_size * i + 1e-4)

        phase = phase.numpy()

        song_spec_out = merge_mag_phase(song_mag_out.cpu().detach().numpy(), phase)
        voice_spec_out = merge_mag_phase(voice_mag_out.cpu().detach().numpy(), phase)
        song_spec_mask = merge_mag_phase(song_mag_mask.cpu().detach().numpy(), phase)
        voice_spec_mask = merge_mag_phase(voice_mag_mask.cpu().detach().numpy(), phase)

        song_spec_tar = merge_mag_phase(song_mag_tar.cpu().detach().numpy(), phase)
        voice_spec_tar = merge_mag_phase(voice_mag_tar.cpu().detach().numpy(), phase)

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
            pbar.update(i*batch_size + batch_item + 1, [])
    log = '[{}/{}] Loss: {:.6f}\n'.format(epoch + 1, num_epoches, avg_loss)
    logger['train'].write(log)
    print(log)

    # =======================================================
    #             soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_audio_tar,
    #                                                                voice_audio_tar, song_audio, voice_audio)
    #             log = '=> done write :' + '%d_%d' % (i, batch_item) + "|" +str(soft_gnsdr[0])+ "|" + \
    #                     str(soft_gnsdr[1])+"|" + str(soft_gsir[0])+ "|" +str(soft_gsir[1])+"|" + \
    #                     str(soft_gsar[0])+"|"+str(soft_gsar[1])+"\n"
    # ====================================================

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
        try:
            out, h_state = model(inputs, h_state)
        except:
            break
        h_state = Variable(h_state.data)

        _, inputs, _ = torch.split(inputs, (513, 513, 513), dim=2)
        # loss
        song_mag_out, voice_mag_out = torch.split(out, (513, 513), dim=2)
        song_mag_tar, voice_mag_tar = torch.split(target, (513, 513), dim=2)

        # Apply mask
        song_mag_mask, voice_mag_mask = soft_mask(song_mag_out, voice_mag_out, inputs)

        # 1 Disc_loss
        disc_lambda = 0.05
        loss = 0.5 * criterion(song_mag_mask, song_mag_tar) + 0.5 * criterion(voice_mag_mask,voice_mag_tar) - disc_lambda * (
                       criterion(song_mag_mask, voice_mag_tar) + criterion(voice_mag_mask, song_mag_tar))

        # 2  Standard_loss
        #             print('compute loss')
        #             loss = criterion(out,target)

        running_loss += loss

        avg_loss = running_loss / (batch_size * i + 1e-4)

        phase = phase.numpy()

        song_spec_out = merge_mag_phase(song_mag_out.cpu().detach().numpy(), phase)
        voice_spec_out = merge_mag_phase(voice_mag_out.cpu().detach().numpy(), phase)
        song_spec_mask = merge_mag_phase(song_mag_mask.cpu().detach().numpy(), phase)
        voice_spec_mask = merge_mag_phase(voice_mag_mask.cpu().detach().numpy(), phase)

        song_spec_tar = merge_mag_phase(song_mag_tar.cpu().detach().numpy(), phase)
        voice_spec_tar = merge_mag_phase(voice_mag_tar.cpu().detach().numpy(), phase)

        for batch_item in range(batch_size) :
            song_audio = create_audio_from_spectrogram(song_spec_out[batch_item, :, :], args)
            voice_audio = create_audio_from_spectrogram(voice_spec_out[batch_item, :, :], args)
            song_audio_mask = create_audio_from_spectrogram(song_spec_mask[batch_item, :, :], args)
            voice_audio_mask = create_audio_from_spectrogram(voice_spec_mask[batch_item, :, :], args)

            song_audio_tar = create_audio_from_spectrogram(song_spec_tar[batch_item, :, :], args)
            voice_audio_tar = create_audio_from_spectrogram(voice_spec_tar[batch_item, :, :], args)

#             song_audio_tar = song_audio_tar[:, np.newaxis]
#             voice_audio_tar = voice_audio_tar[:, np.newaxis]
            mixed_audio = 0.5 * voice_audio_tar + 0.5 * song_audio_tar

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

            pbar_val.update(i * batch_size + batch_item + 1, [])
            soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_audio_tar, voice_audio_tar, song_audio_mask, voice_audio_mask)
            log = '=> done write :' + '%d_%d' % (i, batch_item) + "| GNSDR 0: " +str(soft_gnsdr[0])+ " | GNSDR 1: " + \
                    str(soft_gnsdr[1])+" | GSIR 0: " + str(soft_gsir[0])+ " | GSIR 1: " +str(soft_gsir[1])+" | GSAR 0: " + \
                    str(soft_gsar[0])+" | GSAR 1: "+str(soft_gsar[1])+"\n"
            print(log)
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
