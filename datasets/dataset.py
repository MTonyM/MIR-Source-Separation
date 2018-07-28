from torch.utils.data import Dataset
from utils.audio_op import *


class DataList(Dataset):
    def __init__(self, args):
        self.len_frame = args.len_frame
        self.len_hop = args.len_hop
        pth_name = args.dataset + "_spec_f%d_h%d.pth" % (self.len_frame, self.len_hop)
        self.spec_list = torch.load(os.path.join(args.data_root, "info", pth_name))["iKala_specs"]
#         print(self.spec_list)
        
    def __getitem__(self, idx):
        data = torch.load(self.spec_list[idx])
        song_mag, voice_mag, mixed_mag, phase = data['song_mag'], data['voice_mag'], data['mixed_mag'], data[
            'mixed_phase']
        
        # 
        mixed_mag_pre = torch.load(self.spec_list[max(idx-1,0)])['mixed_mag']
        mixed_mag_next = torch.load(self.spec_list[min(idx,len(self.spec_list)-1)])['mixed_mag']
        input_mag = np.concatenate((mixed_mag_pre, mixed_mag, mixed_mag_next), axis=1)
       
        target_mag = np.concatenate((song_mag, voice_mag), axis=1)
        return input_mag, target_mag, phase

    def __len__(self):
        return len(self.spec_list)


def get_dataloader(args):
    return DataList(args)
