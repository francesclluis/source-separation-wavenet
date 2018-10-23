# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Util.py
# Utility functions for dealing with audio signals and training a Source Separation Wavenet

import os
import numpy as np
import json
import warnings
import scipy.signal
import scipy.stats
import soundfile as sf
import keras
import glob


def l1_l2_loss(y_true, y_pred, l1_weight, l2_weight):

    loss = 0

    if l1_weight != 0:
        loss += l1_weight*keras.losses.mean_absolute_error(y_true, y_pred)

    if l2_weight != 0:
        loss += l2_weight * keras.losses.mean_squared_error(y_true, y_pred)

    return loss


def compute_receptive_field_length(stacks, dilations, filter_length, target_field_length):

    half_filter_length = (filter_length-1)/2
    length = 0
    for d in dilations:
        length += d*half_filter_length
    length = 2*length
    length = stacks * length
    length += target_field_length
    return length


def wav_to_float(x):

    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.finfo(x.dtype).min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x


def float_to_uint8(x):

    x += 1.
    x /= 2.
    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


def keras_float_to_uint8(x):

    x += 1.
    x /= 2.
    uint8_max_value = 255
    x *= uint8_max_value
    return x


def linear_to_ulaw(x, u=255):

    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x


def keras_linear_to_ulaw(x, u=255.0):

    x = keras.backend.sign(x) * (keras.backend.log(1 + u * keras.backend.abs(x)) / keras.backend.log(1 + u))
    return x


def uint8_to_float(x):

    max_value = np.iinfo('uint8').max
    min_value = np.iinfo('uint8').min
    x = x.astype('float32', casting='unsafe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x


def keras_uint8_to_float(x):

    max_value = 255
    min_value = 0
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x


def ulaw_to_linear(x, u=255.0):

    y = np.sign(x) * (1 / float(u)) * (((1 + float(u)) ** np.abs(x)) - 1)
    return y


def keras_ulaw_to_linear(x, u=255.0):

    y = keras.backend.sign(x) * (1 / u) * (((1 + u) ** keras.backend.abs(x)) - 1)
    return y


def one_hot_encode(x, num_values=256):

    if isinstance(x, int):
        x = np.array([x])
    if isinstance(x, list):
        x = np.array(x)
    return np.eye(num_values, dtype='uint8')[x.astype('uint8')]


def one_hot_decode(x):

    return np.argmax(x, axis=-1)


def preemphasis(signal, alpha=0.95):

    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def binary_encode(x, max_value):

    if isinstance(x, int):
        x = np.array([x])
    if isinstance(x, list):
        x = np.array(x)
    width = np.ceil(np.log2(max_value)).astype(int)
    return (((x[:, None] & (1 << np.arange(width)))) > 0).astype(int)


def get_condition_input_encode_func(representation):

        if representation == 'binary':
            return binary_encode
        else:
            return one_hot_encode


def ensure_keys_in_dict(keys, dictionary):

    if all (key in dictionary for key in keys):
        return True
    return False


def get_subdict_from_dict(keys, dictionary):

    return dict((k, dictionary[k]) for k in keys if k in dictionary)


def pretty_json_dump(values, file_path=None):

    if file_path is None:
        print json.dumps(values, sort_keys=True, indent=4, separators=(',', ': '))
    else:
        json.dump(values, open(file_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


def read_wav(filename):
    # Reads in a wav audio file, averages both if stereo, converts the signal to float64 representation

    audio_signal, sample_rate = sf.read(filename)

    if audio_signal.ndim > 1:
        audio_signal = (audio_signal[:, 0] + audio_signal[:, 1])/2.0

    if audio_signal.dtype != 'float64':
        audio_signal = wav_to_float(audio_signal)

    return audio_signal, sample_rate


def load_wav(wav_path, desired_sample_rate):

    sequence, sample_rate = read_wav(wav_path)
    sequence = ensure_sample_rate(sequence, desired_sample_rate, sample_rate)
    return sequence


def write_wav(x, filename, sample_rate):

    if type(x) != np.ndarray:
        x = np.array(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf.write(filename, x, sample_rate)


def ensure_sample_rate(x, desired_sample_rate, file_sample_rate):

    if file_sample_rate != desired_sample_rate:
        return scipy.signal.resample_poly(x, desired_sample_rate, file_sample_rate)
    return x


def normalize(x):
    max_peak = np.max(np.abs(x))
    return x / max_peak


def get_sequence_with_singing_indices(full_sequence):

    signal_magnitude = np.abs(full_sequence)

    chunk_length = 800

    chunks_energies = []
    for i in xrange(0, len(signal_magnitude), chunk_length):
        chunks_energies.append(np.mean(signal_magnitude[i:i + chunk_length]))

    threshold = np.max(chunks_energies) * .1
    chunks_energies = np.asarray(chunks_energies)
    chunks_energies[np.where(chunks_energies < threshold)] = 0
    onsets = np.zeros(len(chunks_energies))
    onsets[np.nonzero(chunks_energies)] = 1
    onsets = np.diff(onsets)

    start_ind = np.squeeze(np.where(onsets == 1))
    finish_ind = np.squeeze(np.where(onsets == -1))

    if finish_ind[0] < start_ind[0]:
        finish_ind = finish_ind[1:]

    if start_ind[-1] > finish_ind[-1]:
        start_ind = start_ind[:-1]

    indices_inici_final = np.insert(finish_ind, np.arange(len(start_ind)), start_ind)

    return np.squeeze((np.asarray(indices_inici_final) + 1) * chunk_length)


def get_indices_subsequence(indices):

    start_indice = 2 * np.random.randint(0, np.ceil(len(indices) / 2))
    vocals_indices = (indices[start_indice], indices[start_indice + 1])
    accompaniment_indices = vocals_indices

    return vocals_indices, accompaniment_indices


def contains_voice(fragment, sequence):

    signal_fragment_magnitude = np.abs(fragment)
    signal_sequence_magnitude = np.abs(sequence)

    chunk_length = 800

    chunks_fragment_energies = []
    for i in xrange(0, len(signal_fragment_magnitude), chunk_length):
        chunks_fragment_energies.append(np.mean(signal_fragment_magnitude[i:i + chunk_length]))

    chunks_sequence_energies = []
    for i in xrange(0, len(signal_sequence_magnitude), chunk_length):
        chunks_sequence_energies.append(np.mean(signal_sequence_magnitude[i:i + chunk_length]))

    threshold = np.max(chunks_sequence_energies) * .1
    chunks_fragment_energies = np.asarray(chunks_fragment_energies)
    chunks_fragment_energies[np.where(chunks_fragment_energies < threshold)] = 0

    if np.count_nonzero(chunks_fragment_energies) > 0:
        return True
    else:
        return False


def dir_contains_files(path):

    for f in os.listdir(path):
        if not f.startswith('.'):
            return True
    return False
