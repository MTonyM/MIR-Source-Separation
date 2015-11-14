import os
from scipy.io import wavfile
import librosa
import numpy as np

def create_audio_from_spectrogram(spec,args):
#     print(type(spec))
     
    if not isinstance(spec, np.ndarray):
        spec = spec.cpu().data.numpy()
# #     print(type(spec))
#     print(spec.shape)
#     print(type(spec[0]))
#     print(spec[0].shape)
        spec_transposed = spec[0].transpose()
        return librosa.istft(spec_transposed, args.len_hop)
    else:
        spec_transposed = spec.transpose()
    
        return librosa.istft(spec_transposed, args.len_hop)

def writeWav(fn, fs, data):
    data = data * 1.5 / np.max(np.abs(data))
    wavfile.write(fn, fs, data)

    
#     def apply_mask(spec, mask)
#     mag_spec = tf.abs(spec)
#     phase_spec = get_phase(spec)
#     return tf.multiply(
#         tf.cast(tf.multiply(mag_spec, mask), tf.complex64), 
#         tf.exp(tf.complex(tf.zeros_like(mag_spec), phase_spec))
#     )
def get_phase(spec):
    return tf.imag(tf.log(spec))


def reals_to_complex_batch(spec_in):
    real, imag = np.split(spec_in, 2, 2)
    spec = np.array(real, dtype=complex)
    spec.imag = imag
    return spec
