import os
import random
import librosa
import numpy as np
import torch
from config import *
from audio_op import *


# import cfg from configs


def listGenerate(args):
    records = os.listdir(args.dir)
    records.remove('.ipynb_checkpoints')
    total_num = len(records)
    #     all_shuffle = torch.randperm(total_num)
    #     all_path = [records[i] for i in all_shuffle]
    info = {
        "base_dir": args.dir,
        "total_num": total_num,
        "file_name": records
    }
    torch.save(info, "iKala_origin.pth")
    return info


def stftGenerate(args, info):
    records = info["file_name"]
    base_dir = info["base_dir"]

    len_frame = args.len_frame  # block size 1024
    len_hop = args.len_hop  # hop size 512

    all_spec = []
    print("iKala has total number of %d" % len(records), " doing preprocessing ...")
    for idx in range(len(records)):
        # load records
        record_path = os.path.join(base_dir, records[idx])
        sound, sample_rate = librosa.load(record_path, mono=False, sr=44100)
        song, voice = sound[1, :], sound[0, :]

        # resample + mix
        song = librosa.resample(song, sample_rate, 16000)
        voice = librosa.resample(voice, sample_rate, 16000)
        mixed = song / 2 + voice / 2

        # STFT
        song_spec = librosa.stft(song, n_fft=len_frame, hop_length=len_hop).transpose()
        voice_spec = librosa.stft(voice, n_fft=len_frame, hop_length=len_hop).transpose()
        mixed_spec = librosa.stft(mixed, n_fft=len_frame, hop_length=len_hop).transpose()

        #         print(mixed_spec.shape)
        #         mixed_audio = create_audio_from_spectrogram(mixed_spec,args)
        #         print (mixed_audio.shape)

        #         writeWav(os.path.join('./test','mixed_%d.wav' % (idx)), args.sample_rate, mixed_audio)
        #         print('done')

        # find real part of spectrum
        #         song_spec, voice_spec, mixed_spec = spectrogram_split_real_imag(song_spec), \
        #                                             spectrogram_split_real_imag(voice_spec), \
        #                                             spectrogram_split_real_imag(mixed_spec)

        song_mag, voice_mag, mixed_mag = np.absolute(song_spec), np.absolute(voice_spec), np.absolute(mixed_spec)
        song_phase, voice_phase, mixed_phase = get_phase(song_spec), get_phase(voice_spec), get_phase(mixed_spec)

        data = {
            "song": song_spec,
            "song_mag": song_mag,
            "song_phase": song_phase,
            "voice": voice_spec,
            "voice_mag": voice_mag,
            "voice_phase": voice_phase,
            "mixed": mixed_spec,
            "mixed_mag": mixed_mag,
            "mixed_phase": mixed_phase,
        }

        file_path = os.path.join("./pre", records[idx] + "_spec.pth")
        torch.save(data, file_path)
        print("=> saved #%d: " % (idx), file_path)
        all_spec.append(file_path)

    info = {
        "iKala_specs": all_spec
    }
    torch.save(info, "iKala_spec_f%d_h%d.pth" % (len_frame, len_hop))


def stftGenerate_aug(args, info):
    records = info["file_name"]
    base_dir = info["base_dir"]

    len_frame = args.len_frame  # block size 1024
    len_hop = args.len_hop  # hop size 512

    all_spec = []
    print("iKala has total number of %d" % len(records), " doing preprocessing ...")
    for idx in range(len(records)):

        # load records
        record_path = os.path.join(base_dir, records[idx])
        sound, sample_rate = librosa.load(record_path, mono=False, sr=44100)
        song, voice = sound[0, :], sound[1, :]

        # resample + mix
        song = librosa.resample(song, sample_rate, 16000)
        voice = librosa.resample(voice, sample_rate, 16000)
        print(voice.shape)
        for shift_len in range(10):
            song = np.roll(song, 4000 * shift_len)
            mixed = song / 2 + voice / 2;

            # STFT
            song_spec = librosa.stft(song, n_fft=len_frame, hop_length=len_hop).transpose()
            voice_spec = librosa.stft(voice, n_fft=len_frame, hop_length=len_hop).transpose()
            mixed_spec = librosa.stft(mixed, n_fft=len_frame, hop_length=len_hop).transpose()

            #         print(mixed_spec.shape)
            #         mixed_audio = create_audio_from_spectrogram(mixed_spec,args)
            #         print (mixed_audio.shape)

            #         writeWav(os.path.join('./test','mixed_%d.wav' % (idx)), args.sample_rate, mixed_audio)
            #         print('done')

            # find real part of spectrum
            #         song_spec, voice_spec, mixed_spec = spectrogram_split_real_imag(song_spec), \
            #                                             spectrogram_split_real_imag(voice_spec), \
            #                                             spectrogram_split_real_imag(mixed_spec)
            song_mag, voice_mag, mixed_mag = np.absolute(song_spec), np.absolute(voice_spec), np.absolute(mixed_spec)
            song_phase, voice_phase, mixed_phase = get_phase(song_spec), get_phase(voice_spec), get_phase(mixed_spec)

            data = {
                "song": song_spec,
                "song_mag": song_mag,
                "song_phase": song_phase,
                "voice": voice_spec,
                "voice_mag": voice_mag,
                "voice_phase": voice_phase,
                "mixed": mixed_spec,
                "mixed_mag": mixed_mag,
                "mixed_phase": mixed_phase,
            }

            file_path = os.path.join("./pre_aug", records[idx] + "_spec_aug%d.pth" % (shift_len))
            torch.save(data, file_path)
            print("=> saved #%d_aug%d: " % (idx, shift_len), file_path)
            all_spec.append(file_path)

    info = {
        "iKala_specs": all_spec
    }
    torch.save(info, "iKala_spec_f%d_h%d_aug.pth" % (len_frame, len_hop))


if __name__ == '__main__':
    args = get_args()
    info = listGenerate(args)
    if args.augment == True:
        stftGenerate_aug(args, info)
    else:
        stftGenerate(args, info)
