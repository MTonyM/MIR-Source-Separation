import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
# from cfgs.config import cfg
from audio_op import *

def transform_spec_from_raw(raw):
    import pdb
    pdb.set_trace()
    spec = np.reshape(raw, [-1, 513 * 2])
    real, imag = np.split(spec, 2)
    orig_spec = np.complex(real, imag)
    return orig_spec

class iKala(Dataset):
    def __init__(self, frame, hop):
        self.len_frame = frame
        self.len_hop = hop
        self.spec_list = torch.load("iKala_spec_f%d_h%d.pth" % (self.len_frame, self.len_hop))["iKala_specs"]
    def __getitem__(self, idx):
        # load file
        data = torch.load(self.spec_list[idx])
#         song_spec, voice_spec, mixed_spec = data['song'], data['voice'], data['mixed']
        song_mag, voice_mag, mixed_mag, phase = data['song_mag'], data['voice_mag'], data['mixed_mag'], data['mixed_phase']     
#         print('pahse_shape',phase.shape)
        # complex operation - ignore
#         song_spec = transform_spec_from_raw(data['song'])
#         voice_spec = transform_spec_from_raw(data['voice'])
#         mixed_spec = transform_spec_from_raw(data['mixed'])
        input_mag = mixed_mag
        target_mag = np.concatenate((song_mag, voice_mag), axis=1)
#         print('input_mag',input_mag.shape)
#         print('target_mag',target_mag.shape)
#         print('phase',phase.shape)
        return input_mag, target_mag, phase
    
    def __len__(self):
        return len(self.spec_list)
    
class iKala_aug(Dataset):
    def __init__(self, frame, hop):
        self.len_frame = frame
        self.len_hop = hop
        self.spec_list = torch.load("iKala_spec_f%d_h%d_aug.pth" % (self.len_frame, self.len_hop))["iKala_specs"]
        
    def __getitem__(self, idx):
        
        # load file with context
        data = torch.load(self.spec_list[idx])
#         data_pre = torch.load(self.spec_list[max(idx-1,0)])
#         data_next = torch.load(self.spec_list[min(idx,len(self.spec_list)-1)])
#         song_spec, voice_spec, mixed_spec = data['song'], data['voice'], data['mixed']
        song_mag, voice_mag, mixed_mag, phase = data['song_mag'], data['voice_mag'], data['mixed_mag'], data['mixed_phase']      
        mixed_mag_pre =  torch.load(self.spec_list[max(idx-1,0)])['mixed_mag']
        mixed_mag_next =  torch.load(self.spec_list[min(idx,len(self.spec_list)-1)])['mixed_mag']                    
        
        # complex operation - ignore
#         song_spec = transform_spec_from_raw(data['song'])
#         voice_spec = transform_spec_from_raw(data['voice'])
#         mixed_spec = transform_spec_from_raw(data['mixed'])

        input_mag = np.concatenate((mixed_mag_pre, mixed_mag, mixed_mag_next), axis=1)

        target_mag = np.concatenate((song_mag, voice_mag), axis=1)

        
        return input_mag, target_mag, phase
    
    def __len__(self):
        return len(self.spec_list)


class Wav_aug(Dataset):
    
    def __init__(self, frame, hop):
        self.len_frame = frame
        self.len_hop = hop
        self.spec_list = torch.load("Wav_spec_f%d_h%d_aug.pth" % (self.len_frame, self.len_hop))["Wav_specs"]
        
    def __getitem__(self, idx):
        
        # load file with context
        data = torch.load(self.spec_list[idx])

#         data_pre = torch.load(self.spec_list[max(idx-1,0)])
#         data_next = torch.load(self.spec_list[min(idx,len(self.spec_list))])
#         song_spec, voice_spec, mixed_spec = data['song'], data['voice'], data['mixed']
        song_mag, voice_mag, mixed_mag, phase = data['song_mag'], data['voice_mag'], data['mixed_mag'], data['mixed_phase']      
        mixed_mag_pre =  torch.load(self.spec_list[max(idx-1,0)])['mixed_mag']
        mixed_mag_next =  torch.load(self.spec_list[min(idx,len(self.spec_list)-1)])['mixed_mag']                    
        
        # complex operation - ignore
#         song_spec = transform_spec_from_raw(data['song'])
#         voice_spec = transform_spec_from_raw(data['voice'])
#         mixed_spec = transform_spec_from_raw(data['mixed'])

        input_mag = np.concatenate((mixed_mag_pre, mixed_mag, mixed_mag_next), axis=1)

        target_mag = np.concatenate((song_mag, voice_mag), axis=1)

        
        return input_mag, target_mag, phase
    
    def __len__(self):
        return len(self.spec_list)
    
if __name__ == '__main__':
    data = iKala(1024, 512)
    print(data.files)
    print(data.__len__())
#     for i in data:
#         print(i)