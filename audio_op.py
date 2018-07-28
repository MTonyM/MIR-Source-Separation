import os
import torch
from scipy.io import wavfile
import librosa
import numpy as np


def soft_mask(voice_mag, song_mag, mix_mag):
    voice_mag = abs(voice_mag)
    song_mag = abs(song_mag)
    mask_value = voice_mag / (voice_mag + song_mag + 1e-5)
    voice_mag_new = mask_value * mix_mag
    song_mag_new = (1 - mask_value) * mix_mag
    return voice_mag_new, song_mag_new


def merge_mag_phase(mag, phase):
    spec = np.multiply(mag, np.cos(phase) + np.sin(phase) * 1j)
    return spec


def spectrogram_split_real_imag(spec):
    return np.concatenate([np.real(spec), np.imag(spec)], axis=1)


def get_phase(spec):
    return np.imag(np.log(spec + 1e-6))


def create_audio_from_spectrogram(spec, args):
    if not isinstance(spec, np.ndarray):
        spec = spec.cpu().data.numpy()
        spec_transposed = spec[0].transpose()
        return librosa.istft(spec_transposed, args.len_hop)
    else:
        spec_transposed = spec.transpose()
        return librosa.istft(spec_transposed, args.len_hop)


def writeWav(fn, fs, data):
    data = data * 1.5 / np.max(np.abs(data))
    wavfile.write(fn, fs, data)


def reals_to_complex_batch(spec_in):
    real, imag = np.split(spec_in, 2, 2)
    spec = np.array(real, dtype=complex)
    spec.imag = imag
    return spec


if __name__ == '__main__':
    pass
