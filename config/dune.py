from configparser import ConfigParser
import glob
#Get the configparser object
config = ConfigParser()

DATA_DIR = '/eos/project/n/neutrino-generators/generatorOutput/GENIE/DIRT_II_Studies/DUNE_Ar_numu/'
MODE = 14
GENERATOR_A = 'G1810a'
GENERATOR_B = 'G1810b'


def get_filepaths(generator, data_dir=DATA_DIR, mode='train'):
    wildcard = f'{data_dir}flat_vec*_{generator}*.root'
    paths = glob.glob(wildcard)
    print(len(paths))
    if mode == 'train':
        return paths[:8]
    else:
        return paths[8:14]


for mode in ['train', 'test']:
    config[mode] = {
        'data_dir': DATA_DIR,
        'generator_a': GENERATOR_A,
        'generator_b': GENERATOR_B,
        'filepaths_a': get_filepaths(GENERATOR_A, mode=mode),
        'filepaths_b': get_filepaths(GENERATOR_B, mode=mode),
        
    }
  
#Write the above sections to config.ini file
with open('config/files.ini', 'w') as conf:
    config.write(conf)