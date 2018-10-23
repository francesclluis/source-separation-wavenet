# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Datasets.py

import util
import os
import numpy as np
import musdb
import logging


class SingingVoiceMUSDB18Dataset():

    def __init__(self, config, model):
        self.model = model
        self.path = config['dataset']['path']
        self.sample_rate = config['dataset']['sample_rate']
        self.file_paths = {'train': {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}, 'val':
            {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}}
        self.sequences = {'train': {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}, 'val':
            {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}}
        self.voice_indices = {'train': [], 'val': []}
        self.batch_size = config['training']['batch_size']
        self.extract_voice_percent = config['dataset']['extract_voice_percentage']
        self.in_memory_percentage = config['dataset']['in_memory_percentage']
        self.num_sequences_in_memory = 0
        self.condition_encode_function = util.get_condition_input_encode_func(config['model']['condition_encoding'])

    def load_dataset(self):

        print('Loading MUSDB18 dataset for singing voice separation...')

        mus = musdb.DB(root_dir=self.path, is_wav=True)
        tracks = mus.load_mus_tracks(subsets='train')
        np.random.seed(seed=1337)
        val_idx = np.random.choice(len(tracks), size=25, replace=False)
        train_idx = [i for i in range(len(tracks)) if i not in val_idx]
        val_tracks = [tracks[i] for i in val_idx]
        train_tracks = [tracks[i] for i in train_idx]
        for condition in ['mixture', 'vocals']:
            self.file_paths['val'][condition] = [track.path[:-11] + condition + '.wav' for track in val_tracks]
        for condition in ['mixture', 'vocals']:
            self.file_paths['train'][condition] = [track.path[:-11] + condition + '.wav' for track in train_tracks]
        self.load_songs()
        return self

    def load_songs(self):

        for set in ['train', 'val']:
            for condition in ['mixture', 'vocals']:
                for filepath in self.file_paths[set][condition]:

                    if condition == 'vocals':

                        sequence = util.load_wav(filepath, self.sample_rate)
                        self.sequences[set][condition].append(sequence)
                        self.num_sequences_in_memory += 1

                        if self.extract_voice_percent > 0:
                            self.voice_indices[set].append(util.get_sequence_with_singing_indices(sequence))
                    else:

                        if self.in_memory_percentage == 1 or np.random.uniform(0, 1) <= (
                                    self.in_memory_percentage - 0.5) * 2:
                            sequence = util.load_wav(filepath, self.sample_rate)
                            self.sequences[set][condition].append(sequence)
                            self.num_sequences_in_memory += 1
                        else:
                            self.sequences[set][condition].append([-1])

    def get_num_sequences_in_dataset(self):
        return len(self.sequences['train']['vocals']) + len(self.sequences['train']['mixture']) + len(
            self.sequences['val']['vocals']) + len(self.sequences['val']['mixture'])

    def retrieve_sequence(self, set, condition, sequence_num):

        if len(self.sequences[set][condition][sequence_num]) == 1:
            sequence = util.load_wav(self.file_paths[set][condition][sequence_num], self.sample_rate)

            if (float(self.num_sequences_in_memory) / self.get_num_sequences_in_dataset()) < self.in_memory_percentage:
                self.sequences[set][condition][sequence_num] = sequence
                self.num_sequences_in_memory += 1
        else:
            sequence = self.sequences[set][condition][sequence_num]

        return np.array(sequence)

    def get_random_batch_generator(self, set):

        if set not in ['train', 'val']:
            raise ValueError("Argument SET must be either 'train' or 'val'")

        while True:
            sample_indices = np.random.randint(0, len(self.sequences[set]['vocals']), self.batch_size)
            batch_inputs = []
            batch_outputs_1 = []
            batch_outputs_2 = []

            for i, sample_i in enumerate(sample_indices):

                while True:

                    starting_index = 0

                    mixture = self.retrieve_sequence(set, 'mixture', sample_i)
                    vocals = self.retrieve_sequence(set, 'vocals', sample_i)
                    accompaniment = mixture - vocals

                    if np.random.uniform(0, 1) < self.extract_voice_percent:
                        indices = self.voice_indices[set][sample_i]
                        vocals_indices, _ = util.get_indices_subsequence(indices)
                        vocals = vocals[vocals_indices[0]:vocals_indices[1]]
                        starting_index = vocals_indices[0]

                    if len(vocals) < self.model.input_length:
                        sample_i = np.random.randint(0, len(self.sequences[set]['vocals']))
                    else:
                        break

                offset_1 = np.squeeze(np.random.randint(0, len(vocals) - self.model.input_length + 1, 1))
                vocals_fragment = vocals[offset_1:offset_1 + self.model.input_length]
                offset_2 = offset_1 + starting_index
                accompaniment_fragment = accompaniment[offset_2:offset_2 + self.model.input_length]

                input = accompaniment_fragment + vocals_fragment
                output_vocals = vocals_fragment
                output_accompaniment = accompaniment_fragment

                batch_inputs.append(input)
                batch_outputs_1.append(output_vocals)
                batch_outputs_2.append(output_accompaniment)

            batch_inputs = np.array(batch_inputs, dtype='float32')
            batch_outputs_1 = np.array(batch_outputs_1, dtype='float32')
            batch_outputs_2 = np.array(batch_outputs_2, dtype='float32')
            batch_outputs_1 = batch_outputs_1[:, self.model.get_padded_target_field_indices()]
            batch_outputs_2 = batch_outputs_2[:, self.model.get_padded_target_field_indices()]

            batch = {'data_input': batch_inputs}, {'data_output_1': batch_outputs_1,
                                                   'data_output_2': batch_outputs_2}

            yield batch

    def get_condition_input_encode_func(self, representation):

        if representation == 'binary':
            return util.binary_encode
        else:
            return util.one_hot_encode

    def get_target_sample_index(self):
        return int(np.floor(self.fragment_length / 2.0))

    def get_samples_of_interest_indices(self, causal=False):

        if causal:
            return -1
        else:
            target_sample_index = self.get_target_sample_index()
            return range(target_sample_index - self.half_target_field_length - self.target_padding,
                         target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_sample_weight_vector_length(self):
        if self.samples_of_interest_only:
            return len(self.get_samples_of_interest_indices())
        else:
            return self.fragment_length


class MultiInstrumentMUSDB18Dataset():

    def __init__(self, config, model):
        self.model = model
        self.path = config['dataset']['path']
        self.sample_rate = config['dataset']['sample_rate']
        self.file_paths = {'train': {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}, 'val':
            {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}}
        self.sequences = {'train': {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}, 'val':
            {'vocals': [], 'mixture': [], 'drums': [], 'other': [], 'bass': []}}
        self.voice_indices = {'train': [], 'val': []}
        self.batch_size = config['training']['batch_size']
        self.extract_voice_percent = config['dataset']['extract_voice_percentage']
        self.in_memory_percentage = config['dataset']['in_memory_percentage']
        self.num_sequences_in_memory = 0
        self.condition_encode_function = util.get_condition_input_encode_func(config['model']['condition_encoding'])

    def load_dataset(self):

        print('Loading MUSDB18 dataset for multi-instrument separation...')

        mus = musdb.DB(root_dir=self.path, is_wav=True)
        tracks = mus.load_mus_tracks(subsets='train')
        np.random.seed(seed=1337)
        val_idx = np.random.choice(len(tracks), size=25, replace=False)
        train_idx = [i for i in range(len(tracks)) if i not in val_idx]
        val_tracks = [tracks[i] for i in val_idx]
        train_tracks = [tracks[i] for i in train_idx]
        for condition in ['mixture', 'vocals', 'drums', 'other', 'bass']:
            self.file_paths['val'][condition] = [track.path[:-11] + condition + '.wav' for track in val_tracks]
        for condition in ['mixture', 'vocals', 'drums', 'other', 'bass']:
            self.file_paths['train'][condition] = [track.path[:-11] + condition + '.wav' for track in train_tracks]
        self.load_songs()
        return self

    def load_songs(self):

        for set in ['train', 'val']:
            for condition in ['vocals', 'mixture', 'drums', 'other', 'bass']:
                for filepath in self.file_paths[set][condition]:

                    if condition == 'vocals':

                        sequence = util.load_wav(filepath, self.sample_rate)
                        self.sequences[set][condition].append(sequence)
                        self.num_sequences_in_memory += 1

                        if self.extract_voice_percent > 0:
                            self.voice_indices[set].append(util.get_sequence_with_singing_indices(sequence))
                    else:

                        if self.in_memory_percentage == 1 or np.random.uniform(0, 1) <= (
                                    self.in_memory_percentage - 0.5) * 2:
                            sequence = util.load_wav(filepath, self.sample_rate)
                            self.sequences[set][condition].append(sequence)
                            self.num_sequences_in_memory += 1
                        else:
                            self.sequences[set][condition].append([-1])

    def get_num_sequences_in_dataset(self):
        return len(self.sequences['train']['vocals']) + len(self.sequences['train']['mixture']) + len(
            self.sequences['val']['vocals']) + len(self.sequences['val']['mixture'])

    def retrieve_sequence(self, set, condition, sequence_num):

        if len(self.sequences[set][condition][sequence_num]) == 1:
            sequence = util.load_wav(self.file_paths[set][condition][sequence_num], self.sample_rate)

            if (float(self.num_sequences_in_memory) / self.get_num_sequences_in_dataset()) < self.in_memory_percentage:
                self.sequences[set][condition][sequence_num] = sequence
                self.num_sequences_in_memory += 1
        else:
            sequence = self.sequences[set][condition][sequence_num]

        return np.array(sequence)

    def get_random_batch_generator(self, set):

        if set not in ['train', 'val']:
            raise ValueError("Argument SET must be either 'train' or 'val'")

        while True:
            sample_indices = np.random.randint(0, len(self.sequences[set]['vocals']), self.batch_size)
            batch_inputs = []
            batch_outputs_1 = []
            batch_outputs_2 = []
            batch_outputs_3 = []

            for i, sample_i in enumerate(sample_indices):

                while True:

                    starting_index = 0

                    vocals = self.retrieve_sequence(set, 'vocals', sample_i)
                    bass = self.retrieve_sequence(set, 'bass', sample_i)
                    drums = self.retrieve_sequence(set, 'drums', sample_i)
                    other = self.retrieve_sequence(set, 'other', sample_i)

                    if np.random.uniform(0, 1) < self.extract_voice_percent:
                        indices = self.voice_indices[set][sample_i]
                        vocals_indices, _ = util.get_indices_subsequence(indices)
                        vocals = vocals[vocals_indices[0]:vocals_indices[1]]
                        starting_index = vocals_indices[0]

                    if len(vocals) < self.model.input_length:
                        sample_i = np.random.randint(0, len(self.sequences[set]['vocals']))
                    else:
                        break

                offset_1 = np.squeeze(np.random.randint(0, len(vocals) - self.model.input_length + 1, 1))
                vocals_fragment = vocals[offset_1:offset_1 + self.model.input_length]
                offset_2 = offset_1 + starting_index
                bass_fragment = bass[offset_2:offset_2 + self.model.input_length]
                drums_fragment = drums[offset_2:offset_2 + self.model.input_length]
                other_fragment = other[offset_2:offset_2 + self.model.input_length]

                input = vocals_fragment + bass_fragment + drums_fragment + other_fragment
                output_vocals = vocals_fragment
                output_drums = drums_fragment
                output_bass = bass_fragment

                batch_inputs.append(input)
                batch_outputs_1.append(output_vocals)
                batch_outputs_2.append(output_drums)
                batch_outputs_3.append(output_bass)

            batch_inputs = np.array(batch_inputs, dtype='float32')
            batch_outputs_1 = np.array(batch_outputs_1, dtype='float32')
            batch_outputs_2 = np.array(batch_outputs_2, dtype='float32')
            batch_outputs_3 = np.array(batch_outputs_3, dtype='float32')

            batch_outputs_1 = batch_outputs_1[:, self.model.get_padded_target_field_indices()]
            batch_outputs_2 = batch_outputs_2[:, self.model.get_padded_target_field_indices()]
            batch_outputs_3 = batch_outputs_3[:, self.model.get_padded_target_field_indices()]

            batch = {'data_input': batch_inputs}, {'data_output_1': batch_outputs_1,
                                                   'data_output_2': batch_outputs_2,
                                                   'data_output_3': batch_outputs_3}

            yield batch

    def get_condition_input_encode_func(self, representation):

        if representation == 'binary':
            return util.binary_encode
        else:
            return util.one_hot_encode

    def get_target_sample_index(self):
        return int(np.floor(self.fragment_length / 2.0))

    def get_samples_of_interest_indices(self, causal=False):

        if causal:
            return -1
        else:
            target_sample_index = self.get_target_sample_index()
            return range(target_sample_index - self.half_target_field_length - self.target_padding,
                         target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_sample_weight_vector_length(self):
        if self.samples_of_interest_only:
            return len(self.get_samples_of_interest_indices())
        else:
            return self.fragment_length
