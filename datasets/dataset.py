import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from audio_op import *

data_root = "/home/maoym/MIR-separation/DL_monaural_separation/pyTorch_version/data"

class dataList(Dataset):
    def __init__(self, args):
        self.len_frame = args.len_frame
        self.len_hop = args.len_hop
        self.spec_list = torch.load(data_root + "/spec/" + args.dataset + "_spec_f%d_h%d.pth" % (self.len_frame, self.len_hop))
        if args.mode = 0:
            self.spec_list = self.spec_list [0:int(len(self.spec_list) * args.train_ratio)] 
    def __getitem__(self, idx):
        # load file
        data = torch.load(self.spec_list[idx])
        song_mag, voice_mag, mixed_mag, phase = data['song_mag'], data['voice_mag'], data['mixed_mag'], data['mixed_phase']
        input_mag = mixed_mag
        target_mag = np.concatenate((song_mag, voice_mag), axis=1)
        return input_mag, target_mag, phase

    def __len__(self):
        return len(self.spec_list)



# class Wav_aug(iKala_aug):
#     def __init__(self, frame, hop):
#         self.len_frame = frame
#         self.len_hop = hop
#         self.spec_list = torch.load("Wav_spec_f%d_h%d_aug.pth" % (self.len_frame, self.len_hop))["Wav_specs"]


def get_dataloader(args):
    return eval(data + '(args)')

