# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Main.py

import sys
import logging
import optparse
import json
import os
import models
import datasets
import util
import separate


def set_system_settings():
    sys.setrecursionlimit(50000)
    logging.getLogger().setLevel(logging.INFO)


def get_command_line_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(config='sessions/multi-instrument/config.json')
    parser.set_defaults(mode='training')
    parser.set_defaults(target='multi-instrument')
    parser.set_defaults(load_checkpoint=None)
    parser.set_defaults(condition_value=0)
    parser.set_defaults(batch_size=None)
    parser.set_defaults(one_shot=False)
    parser.set_defaults(mixture_input_path=None)
    parser.set_defaults(print_model_summary=False)
    parser.set_defaults(target_field_length=None)

    parser.add_option('--mode', dest='mode')
    parser.add_option('--target', dest='target')
    parser.add_option('--print_model_summary', dest='print_model_summary')
    parser.add_option('--config', dest='config')
    parser.add_option('--load_checkpoint', dest='load_checkpoint')
    parser.add_option('--condition_value', dest='condition_value')
    parser.add_option('--batch_size', dest='batch_size')
    parser.add_option('--one_shot', dest='one_shot')
    parser.add_option('--mixture_input_path', dest='mixture_input_path')
    parser.add_option('--target_field_length', dest='target_field_length')

    (options, args) = parser.parse_args()

    return options


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)


def get_dataset(config, cla, model):

    if config['dataset']['type'] == 'musdb18':
        if cla.target == 'singing-voice':
            return datasets.SingingVoiceMUSDB18Dataset(config, model).load_dataset()
        elif cla.target == 'multi-instrument':
            return datasets.MultiInstrumentMUSDB18Dataset(config, model).load_dataset()


def training(config, cla):

    # Instantiate Model

    if cla.target == 'singing-voice':
        model = models.SingingVoiceSeparationWavenet(config, load_checkpoint=cla.load_checkpoint,
                                                     print_model_summary=cla.print_model_summary)
    elif cla.target == 'multi-instrument':
        model = models.MultiInstrumentSeparationWavenet(config, load_checkpoint=cla.load_checkpoint,
                                                        print_model_summary=cla.print_model_summary)
    else:
        raise Exception("Argument target must be either 'singing-voice' or 'multi-instrument'")

    dataset = get_dataset(config, cla, model)
    num_steps_train = config['training']['num_steps_train']
    num_steps_val = config['training']['num_steps_test']
    train_set_generator = dataset.get_random_batch_generator('train')
    val_set_generator = dataset.get_random_batch_generator('val')

    model.fit_model(train_set_generator, num_steps_train, val_set_generator, num_steps_val,
                    config['training']['num_epochs'])


def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
            break
        j += 1
    return output_folder_path


def inference(config, cla):

    if cla.batch_size is not None:
        batch_size = int(cla.batch_size)
    else:
        batch_size = config['training']['batch_size']

    if cla.target_field_length is not None:
        cla.target_field_length = int(cla.target_field_length)

    if not bool(cla.one_shot):

        if config['model']['type'] == 'singing-voice':
            model = models.SingingVoiceSeparationWavenet(config, target_field_length=cla.target_field_length,
                                                         load_checkpoint=cla.load_checkpoint,
                                                         print_model_summary=cla.print_model_summary)

        elif config['model']['type'] == 'multi-instrument':
            model = models.MultiInstrumentSeparationWavenet(config, target_field_length=cla.target_field_length,
                                                            load_checkpoint=cla.load_checkpoint,
                                                            print_model_summary=cla.print_model_summary)

        print 'Performing inference..'

    else:
        print 'Performing one-shot inference..'

    samples_folder_path = os.path.join(config['training']['path'], 'samples')
    output_folder_path = get_valid_output_folder_path(samples_folder_path)

    #If input_path is a single wav file, then set filenames to single element with wav filename
    if cla.mixture_input_path.endswith('.wav'):
        filenames = [cla.mixture_input_path.rsplit('/', 1)[-1]]
        cla.mixture_input_path = cla.mixture_input_path.rsplit('/', 1)[0] + '/'

    else:
        if not cla.mixture_input_path.endswith('/'):
            cla.mixture_input_path += '/'
        filenames = [filename for filename in os.listdir(cla.mixture_input_path) if filename.endswith('.wav')]

    for filename in filenames:
        mixture_input = util.load_wav(cla.mixture_input_path + filename, config['dataset']['sample_rate'])

        input = {'mixture': mixture_input}

        output_filename_prefix = filename[0:-4]

        if bool(cla.one_shot):
            if len(input['mixture']) % 2 == 0:  # If input length is even, remove one sample
                input['mixture'] = input['mixture'][:-1]

            if config['model']['type'] == 'singing-voice':
                model = models.SingingVoiceSeparationWavenet(config, target_field_length=cla.target_field_length,
                                                             load_checkpoint=cla.load_checkpoint,
                                                             print_model_summary=cla.print_model_summary)

            elif config['model']['type'] == 'multi-instrument':
                model = models.MultiInstrumentSeparationWavenet(config, target_field_length=cla.target_field_length,
                                                                load_checkpoint=cla.load_checkpoint,
                                                                print_model_summary=cla.print_model_summary)

        print "Separating: " + filename
        separate.separate_sample(model, input, batch_size, output_filename_prefix,
                                 config['dataset']['sample_rate'], output_folder_path, config['model']['type'])


def main():

    set_system_settings()
    cla = get_command_line_arguments()
    config = load_config(cla.config)

    if cla.mode == 'training':
        training(config, cla)
    elif cla.mode == 'inference':
        inference(config, cla)


if __name__ == "__main__":
    main()
