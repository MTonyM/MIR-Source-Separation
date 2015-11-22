import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os

from scipy.io import wavfile
from dataset import *
from rnn import get_model
from audio_op import *
from config import *
from evaluate import bss_eval_sources

def writeWav(fn, fs, data):
    data = data * 1.5 / np.max(np.abs(data))
    wavfile.write(fn, fs, data)

def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:len_cropped]
    src2_wav = src2_wav[:len_cropped]
    mixed_wav = mixed_wav[:len_cropped]
    gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0

    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), False)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), False)
    nsdr = sdr - sdr_mixed
    gnsdr += len_cropped * nsdr
    gsir += len_cropped * sir
    gsar += len_cropped * sar
    total_len += len_cropped
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    return gnsdr, gsir, gsar


# TF example
#    def add_loss_op(self, target):
#         self.target = target  # for outputting later
#         real_target = abs(target)
#         delta = self.output - real_target 
#         squared_error = tf.reduce_mean(tf.pow(delta, 2)) 

#         l2_cost = tf.reduce_mean([tf.norm(v) for v in tf.trainable_variables() if len(v.get_shape().as_list()) == 3])
#         self.loss = Config.l2_lambda * l2_cost + squared_error
#         tf.summary.scalar("loss", self.loss)
#         masked_loss = tf.abs(self.soft_masked_output) - real_target
#         self.masked_loss = Config.l2_lambda * l2_cost + tf.reduce_mean(tf.pow(masked_loss, 2))
#         tf.summary.scalar('masked_loss', self.masked_loss)
#         tf.summary.scalar('regularization_cost', Config.l2_lambda * l2_cost)
 
    
    
def main(args):
    num_epoches = args.num_epoches
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    result_wav_dir = './results'
    
    train_dataset = iKala_aug(args.len_frame, args.len_hop)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size,shuffle=True)
    
    model = get_model(args)
    use_gpu = torch.cuda.is_available()  # 
    if use_gpu:
        model = model.cuda()
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # RNN
    h_state = None    
    
    for epoch in range(num_epoches):
        epoch_file = './results/epoch_%d'% epoch
        if not os.path.exists(epoch_file):
            os.makedirs(epoch_file)
        
            

        print('epoch {}'.format(epoch + 1))
        
        
        running_loss = 0.0
        for i, (inputs, target, phase) in enumerate(train_loader):
            
            batch_file = epoch_file + '/batch_%d'%i 
            if not os.path.exists(batch_file):
                os.makedirs(batch_file)

            model.train()
            
            
            #concatenate
#             if i == 0:
#                 last_win = train_loader[i][0]
#             else:
#                 last_win = train_loader[i-1][0]
                    
#             if i ==   train_loader.shape[0]:
#                 next_win = train_loader[i][0]
#             else:
#                 next_win == train_loader[i+1][0]
                
#             inputs = concatenate( (last_win, inputs, next_win), axis=1)   
            
                
                
            # Variable
            if use_gpu:
                print('use gpu')
                inputs= Variable(inputs).cuda()
#                 print('inputs',inputs.shape)
                target = Variable(target).cuda()
#                 print('target',target.shape)
            else:
                inputs= Variable(inputs)
                target = Variable(target)
            try:
                out, h_state= model(inputs, h_state)
            except:
                print('break')
                break
            
            h_state = Variable(h_state.data)
            # Soft Masking        
#             soft_song_mask = tf.abs(song_out) / (tf.abs(song_out) + tf.abs(voice_out) + 1e-10)
#             soft_voice_mask = 1 - soft_song_mask
#             input_spec_curr = self.input[:,:,1]  # current frame of input spec


#             soft_song_output = apply_mask(input_spec_curr, soft_song_mask)
#             soft_voice_output = apply_mask(input_spec_curr, soft_voice_mask)
#             self.soft_masked_output = tf.concat([soft_song_output, soft_voice_output], axis=1)
            
            
            pre_win,inputs,next_win = torch.split(inputs, (513,513,513), dim = 2)
            # loss
            print(out.shape)
            song_mag_out, voice_mag_out = torch.split(out, (513,513), dim = 2)
            song_mag_tar, voice_mag_tar = torch.split(target, (513,513), dim = 2)
            
            
            # Apply mask 
            
            song_mag_mask, voice_mag_mask = soft_mask(song_mag_out, voice_mag_out, inputs)
            
            # Disc_loss
            disc_lambda = 0.05
            loss = criterion(song_mag_out,song_mag_tar) + criterion(voice_mag_out,voice_mag_tar) - disc_lambda * ( criterion(song_mag_out,voice_mag_tar) + criterion(voice_mag_out,song_mag_tar))
    
    
#             # Standard_loss
#             print('compute loss')
#             loss = criterion(out,target)
            
            
            
            
            running_loss += loss
          
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('[{}/{}] Loss: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i + 1e-4)))

            
            phase = phase.numpy()
            mixed_spec = merge_mag_phase(inputs.data.cpu().numpy(), phase)

            song_spec_out = merge_mag_phase(song_mag_out.cpu().detach().numpy(),phase)
            voice_spec_out = merge_mag_phase(voice_mag_out.cpu().detach().numpy(),phase)
            song_spec_mask = merge_mag_phase(song_mag_mask.cpu().detach().numpy(),phase)
            voice_spec_mask = merge_mag_phase(voice_mag_mask.cpu().detach().numpy(),phase)            
            
            
            song_spec_tar = merge_mag_phase(song_mag_tar.cpu().detach().numpy(), phase)
            voice_spec_tar = merge_mag_phase(voice_mag_tar.cpu().detach().numpy(), phase)

            

#             song_spec_out, voice_spec_out = np.split(out.data.cpu().numpy(), 2, 2)   
#             song_spec_out = reals_to_complex_batch(song_spec_out)
#             voice_spec_out = reals_to_complex_batch(voice_spec_out)
            
#             song_spec_tar, voice_spec_tar = np.split(target.data.cpu().numpy(), 2, 2)
#             song_spec_tar = reals_to_complex_batch(song_spec_tar)
#             voice_spec_tar = reals_to_complex_batch(voice_spec_tar)
            
#             mixed_spec = reals_to_complex_batch(inputs.data.cpu().numpy())
            
            
            for batch_item in range(batch_size):
                song_audio = create_audio_from_spectrogram(song_spec_out[batch_item,:,:], args)
                voice_audio = create_audio_from_spectrogram(voice_spec_out[batch_item,:,:], args)
                song_audio_mask = create_audio_from_spectrogram(song_spec_mask[batch_item,:,:], args)
                voice_audio_mask = create_audio_from_spectrogram(voice_spec_mask[batch_item,:,:], args)                
                
                song_audio_tar = create_audio_from_spectrogram(song_spec_tar[batch_item,:,:], args)
                voice_audio_tar = create_audio_from_spectrogram(voice_spec_tar[batch_item,:,:], args)                
                mixed_audio = create_audio_from_spectrogram(mixed_spec[batch_item,:,:], args)   
                
                writeWav(os.path.join(batch_file, 'song_%d_%d.wav' % (i, batch_item)), 
                         args.sample_rate, song_audio)
                writeWav(os.path.join(batch_file, 'voice_%d_%d.wav' % (i, batch_item)), 
                         args.sample_rate, voice_audio)
                writeWav(os.path.join(batch_file, 'mixed_%d_%d.wav' % (i, batch_item)), 
                         args.sample_rate, mixed_audio)
                writeWav(os.path.join(batch_file, 'song_%d_%d_mask.wav' % (i, batch_item)), 
                         args.sample_rate, song_audio_mask)
                writeWav(os.path.join(batch_file, 'voice_%d_%d_mask.wav' % (i, batch_item)), 
                         args.sample_rate, voice_audio_mask)
                
                soft_gnsdr, soft_gsir, soft_gsar = bss_eval_global(mixed_audio, song_audio_tar, 
                                                                   voice_audio_tar, song_audio, voice_audio)
                
                
                print('=> done write :', '%d_%d' % (i, batch_item), soft_gnsdr[0], soft_gnsdr[1], soft_gsir[0], soft_gsir[1], soft_gsar[0], soft_gsar[1])
#                 print('=> done write :', '%d_%d' % (i, batch_item), 
#                       "GNSDR: %fdB "%soft_gnsdr, 
#                       "GSIR: %fdB"%soft_gsir, 
#                       "GSAR: %fdB "%soft_gsar)

            
            # soft masking
        
            # eval   
        
            
            
if __name__ == '__main__':
    args = get_args()
    main(args)
    