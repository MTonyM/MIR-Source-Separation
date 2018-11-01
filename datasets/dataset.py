from torch.utils.data import Dataset
from utils.audio_op import *
import numpy as np
from torch.utils.data import DataLoader
import re


class DataList(Dataset):
    def __init__(self, args, mode):
        self.len_frame = args.len_frame
        self.len_hop = args.len_hop
        pth_name = args.dataset + "_spec_f%d_h%d.pth" % (self.len_frame, self.len_hop)
        pth_path = os.path.join(args.data_root, "info", pth_name)
#         spec_list = torch.load(pth_path)[args.dataset + "_specs"]
        spec_list = torch.load(pth_path)["All_specs"]

        self.train_ratio = args.train_ratio
        total_batches = int(np.ceil(len(spec_list) / args.batch_size))
        split_index = int(np.floor(total_batches * float(self.train_ratio))) * args.batch_size

        if mode == 'train':
            self.spec_list = spec_list[0:split_index]
        else: #train mode

            self.spec_list = [ x for x in spec_list[split_index:] if int(re.split(r"[._/]",x)[-2]) == 0]
            self.spec_list = self.spec_list[-150:]

    def __getitem__(self, idx):
        data = torch.load(self.spec_list[idx])
        song_mag, voice_mag, mixed_mag, phase = data['song_mag'], data['voice_mag'], data['mixed_mag'], data[
            'mixed_phase']


        # 
#         print (mixed_mag.shape)
 
        length = mixed_mag.shape[0]
#         print(length)
        mixed_mag_pre = np.concatenate((mixed_mag[0][np.newaxis,:],mixed_mag[0:(length-1)][:]),axis=0)
        mixed_mag_next = np.concatenate((mixed_mag[1:length][:],mixed_mag[length-1][np.newaxis,:]),axis=0)
        
#         pad_mode = 'wrap'  # repeat pading
        pad_mode = 'constant' # zero pading
        input_mag = np.concatenate((mixed_mag_pre, mixed_mag, mixed_mag_next), axis=1)
        input_mag = np.pad(input_mag, ((0, 400-input_mag.shape[0]), (0, 0)), pad_mode)
        target_mag = np.concatenate((song_mag, voice_mag), axis=1)
        target_mag = np.pad(target_mag, ((0, 400-target_mag.shape[0]), (0, 0)), pad_mode)
        phase = np.pad(phase, ((0, 400-phase.shape[0]), (0, 0)), pad_mode)
        
#         print(input_mag.shape)
#         print(phase.shape)
        return input_mag, target_mag, phase 
    

    def __len__(self):
        return len(self.spec_list)


def get_dataloader(args):
    train_dataset, test_dataset = DataList(args, 'train'), DataList(args, 'test')
    trainLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle=True)
    testLoader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return trainLoader, testLoader
