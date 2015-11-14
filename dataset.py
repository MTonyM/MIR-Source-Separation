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
        
        song_spec, voice_spec, mixed_spec = data['song'], data['voice'], data['mixed']
        
        # complex operation - ignore
#         song_spec = transform_spec_from_raw(data['song'])
#         voice_spec = transform_spec_from_raw(data['voice'])
#         mixed_spec = transform_spec_from_raw(data['mixed'])

        input_spec = mixed_spec
        target_spec = np.concatenate((voice_spec, song_spec), axis=1)
        
        
        return input_spec, target_spec
    
    def __len__(self):
        return len(self.spec_list)

    
if __name__ == '__main__':
    data = iKala(1024, 512)
    print(data.files)
    print(data.__len__())
#     for i in data:
#         print(i)