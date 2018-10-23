# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Separate.py

from __future__ import division
import os
import util
import tqdm
import numpy as np


def separate_sample(model, input, batch_size, output_filename_prefix, sample_rate, output_path, target):

    if target == 'singing-voice':

        if len(input['mixture']) < model.receptive_field_length:
            raise ValueError('Input is not long enough to be used with this model.')

        num_output_samples = input['mixture'].shape[0] - (model.receptive_field_length - 1)
        num_fragments = int(np.ceil(num_output_samples / model.target_field_length))
        num_batches = int(np.ceil(num_fragments / batch_size))

        vocals_output = []
        num_pad_values = 0
        fragment_i = 0
        for batch_i in tqdm.tqdm(range(0, num_batches)):

            if batch_i == num_batches - 1:  # If its the last batch
                batch_size = num_fragments - batch_i * batch_size

            input_batch = np.zeros((batch_size, model.input_length))

            # Assemble batch
            for batch_fragment_i in range(0, batch_size):

                if fragment_i + model.target_field_length > num_output_samples:
                    remainder = input['mixture'][fragment_i:]
                    current_fragment = np.zeros((model.input_length,))
                    current_fragment[:remainder.shape[0]] = remainder
                    num_pad_values = model.input_length - remainder.shape[0]
                else:
                    current_fragment = input['mixture'][fragment_i:fragment_i + model.input_length]

                input_batch[batch_fragment_i, :] = current_fragment
                fragment_i += model.target_field_length

            separated_output_fragments = model.separate_batch({'data_input': input_batch})

            if type(separated_output_fragments) is list:
                vocals_output_fragment = separated_output_fragments[0]

            vocals_output_fragment = vocals_output_fragment[:,
                                     model.target_padding: model.target_padding + model.target_field_length]
            vocals_output_fragment = vocals_output_fragment.flatten().tolist()

            if type(separated_output_fragments) is float:
                vocals_output_fragment = [vocals_output_fragment]

            vocals_output = vocals_output + vocals_output_fragment

        vocals_output = np.array(vocals_output)

        if num_pad_values != 0:
            vocals_output = vocals_output[:-num_pad_values]

        mixture_valid_signal = input['mixture'][
                               model.half_receptive_field_length:model.half_receptive_field_length + len(vocals_output)]

        accompaniment_output = mixture_valid_signal - vocals_output

        output_vocals_filename = output_filename_prefix + '_vocals.wav'
        output_accompaniment_filename = output_filename_prefix + '_accompaniment.wav'

        output_vocals_filepath = os.path.join(output_path, output_vocals_filename)
        output_accompaniment_filepath = os.path.join(output_path, output_accompaniment_filename)

        util.write_wav(vocals_output, output_vocals_filepath, sample_rate)
        util.write_wav(accompaniment_output, output_accompaniment_filepath, sample_rate)

    if target == 'multi-instrument':

        if len(input['mixture']) < model.receptive_field_length:
            raise ValueError('Input is not long enough to be used with this model.')

        num_output_samples = input['mixture'].shape[0] - (model.receptive_field_length - 1)
        num_fragments = int(np.ceil(num_output_samples / model.target_field_length))
        num_batches = int(np.ceil(num_fragments / batch_size))

        vocals_output = []
        drums_output = []
        bass_output = []

        num_pad_values = 0
        fragment_i = 0
        for batch_i in tqdm.tqdm(range(0, num_batches)):

            if batch_i == num_batches - 1:  # If its the last batch
                batch_size = num_fragments - batch_i * batch_size

            input_batch = np.zeros((batch_size, model.input_length))

            # Assemble batch
            for batch_fragment_i in range(0, batch_size):

                if fragment_i + model.target_field_length > num_output_samples:
                    remainder = input['mixture'][fragment_i:]
                    current_fragment = np.zeros((model.input_length,))
                    current_fragment[:remainder.shape[0]] = remainder
                    num_pad_values = model.input_length - remainder.shape[0]
                else:
                    current_fragment = input['mixture'][fragment_i:fragment_i + model.input_length]

                input_batch[batch_fragment_i, :] = current_fragment
                fragment_i += model.target_field_length

            separated_output_fragments = model.separate_batch({'data_input': input_batch})

            if type(separated_output_fragments) is list:
                vocals_output_fragment = separated_output_fragments[0]
                drums_output_fragment = separated_output_fragments[1]
                bass_output_fragment = separated_output_fragments[2]

            vocals_output_fragment = vocals_output_fragment[:,
                                     model.target_padding: model.target_padding + model.target_field_length]
            vocals_output_fragment = vocals_output_fragment.flatten().tolist()

            drums_output_fragment = drums_output_fragment[:,
                                    model.target_padding: model.target_padding + model.target_field_length]
            drums_output_fragment = drums_output_fragment.flatten().tolist()

            bass_output_fragment = bass_output_fragment[:,
                                   model.target_padding: model.target_padding + model.target_field_length]
            bass_output_fragment = bass_output_fragment.flatten().tolist()

            if type(separated_output_fragments) is float:
                vocals_output_fragment = [vocals_output_fragment]
            if type(drums_output_fragment) is float:
                drums_output_fragment = [drums_output_fragment]
            if type(bass_output_fragment) is float:
                bass_output_fragment = [bass_output_fragment]

            vocals_output = vocals_output + vocals_output_fragment
            drums_output = drums_output + drums_output_fragment
            bass_output = bass_output + bass_output_fragment

        vocals_output = np.array(vocals_output)
        drums_output = np.array(drums_output)
        bass_output = np.array(bass_output)

        if num_pad_values != 0:
            vocals_output = vocals_output[:-num_pad_values]
            drums_output = drums_output[:-num_pad_values]
            bass_output = bass_output[:-num_pad_values]

        mixture_valid_signal = input['mixture'][
                               model.half_receptive_field_length:model.half_receptive_field_length + len(vocals_output)]

        other_output = mixture_valid_signal - vocals_output - drums_output - bass_output

        output_vocals_filename = output_filename_prefix + '_vocals.wav'
        output_drums_filename = output_filename_prefix + '_drums.wav'
        output_bass_filename = output_filename_prefix + '_bass.wav'
        output_other_filename = output_filename_prefix + '_other.wav'

        output_vocals_filepath = os.path.join(output_path, output_vocals_filename)
        output_drums_filepath = os.path.join(output_path, output_drums_filename)
        output_bass_filepath = os.path.join(output_path, output_bass_filename)
        output_other_filepath = os.path.join(output_path, output_other_filename)

        util.write_wav(vocals_output, output_vocals_filepath, sample_rate)
        util.write_wav(drums_output, output_drums_filepath, sample_rate)
        util.write_wav(bass_output, output_bass_filepath, sample_rate)
        util.write_wav(other_output, output_other_filepath, sample_rate)