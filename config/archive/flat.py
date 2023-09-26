from configparser import ConfigParser
import glob
#Get the configparser object
config = ConfigParser()

DATA_DIR = '/eos/user/c/cristova/DUNE/AlternateGenerators/'
MODE = 12
GENERATOR_A = 'GENIEv2'
GENERATOR_B = 'GENIEv3_G18_10b'

modes = {
    'train': {
        'n_files': 1,
        'decimal_index': '1',
    },
    'test': {
        'n_files': 1,
        'decimal_index': '2',
    }
}


def get_filepaths(generator, mode, n_files, decimal_index, data_dir=DATA_DIR,):
    high = n_files - 1
    wildcard = f'{data_dir}flat_*_{mode}_{generator}*_1M_0{decimal_index}[0-{high}]_NUISFLAT.root'
    print(wildcard)
    return glob.glob(wildcard)


print('Creating config file based on wildcards: ')

for mode in ['train', 'test']:

    config[mode] = {
        'data_dir': DATA_DIR,
        'generator_a': GENERATOR_A,
        'generator_b': GENERATOR_B,
        'filepaths_a': get_filepaths(GENERATOR_A, MODE, modes[mode]['n_files'], modes[mode]['decimal_index']),
        'filepaths_b': get_filepaths(GENERATOR_B, MODE, modes[mode]['n_files'], modes[mode]['decimal_index']),
    }

  
#Write the above sections to config.ini file
with open('config/files.ini', 'w') as conf:
    config.write(conf)