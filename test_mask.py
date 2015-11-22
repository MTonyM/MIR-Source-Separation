import librosa
import numpy as np
from audio_op import *
import os

len_frame = 1024
len_hop =512
print('begin')
records = '/home/maoym/MIR-separation/DL_monaural_separation/pyTorch_version/separation_LSTM'
# song,sample1  = librosa.load(os.path.join(records, 'song_9_13.wav'), mono=False, sr=16000)
voice,sample2 = librosa.load('voice_9_13.wav', mono=True, sr=16000)
mixed,sample3 = librosa.load('mixed_9_13.wav', mono=True, sr=16000)
print('loaded')


# song , voice= sound[0, :] , sound[1, :]
song_spec = librosa.stft(song, n_fft=len_frame, hop_length=len_hop).transpose()
voice_spec = librosa.stft(voice, n_fft=len_frame, hop_length=len_hop).transpose()
mixed_spec = librosa.stft(mixed, n_fft=len_frame, hop_length=len_hop).transpose()
song_mag,voice_mag,mixed_mag = np.absolute(song_spec),np.absolute(voice_spec),np.absolute(mixed_spec)
song_phase,voice_phase,mixed_phase = get_phase(song_spec),get_phase(voice_spec),get_phase(mixed_spec)
song_mag_out, voice_mag_out = soft_mask(song_mag_out, voice_mag_out, mixed_mag)
song_spec_out = merge_mag_phase(song_mag_out,phase)
voice_spec_out = merge_mag_phase(voice_mag_out,phase)
song_audio = create_audio_from_spectrogram(song_spec_out[batch_item,:,:], args)
voice_audio = create_audio_from_spectrogram(voice_spec_out[batch_item,:,:], args)
writeWav('song.wav', 16000, song_audio)
writeWav('voice.wav', 16000, voice_audio)