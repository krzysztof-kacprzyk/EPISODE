import os

def get_composition_scores_folder_path(model_name, dataset_name, benchmarks_dir='results'):
    return os.path.join(benchmarks_dir, 'composition_scores', model_name, dataset_name)

def get_composition_scores_file_name(seed, m):
    return f'composition_scores_train_val_{seed}_{m}.csv'

def get_inductive_bias_composition_libraries(dataset_symbol):
    if dataset_symbol == 'SIR':
        composition_libraries = {
            0: [
                ['--c','-+h']
            ],
            1: [
                ['++c','+-c','--c','-+h']
            ],
            2: [
                ['++c','+-h']
            ]
        }
    elif dataset_symbol == 'beta':
        composition_libraries = {
            0: [
                ['++c','+-c','--c','-+c']
            ]
        }
    elif dataset_symbol in ['pk','real_pharma']:
        composition_libraries = {
            0: [
                ['++c','+-c','--c','-+h']
            ]
        }
    elif dataset_symbol == 'tumor':
        composition_libraries = {
            0: [
                ['++f'],
                ['-+c','++f'],
                ['-+h'],
                ['--c','-+h']
            ]
        }
    elif dataset_symbol == 'bike':
        composition_libraries = {
            0: [
                ['++c', '+-c', '--c', '-+c', '++c', '+-c', '--c', '-+c'],
                ['++c', '+-c', '--c', '-+c']
            ]
        }
    elif dataset_symbol == 'HIV':
        composition_libraries = {
            0: [
                ['-+h'],
                ['--c','-+h'],
                ['--c','-+c','++c','+-h'],
                ['-+c','++c','+-h']
            ],
            1: [
                ['+-c','--c','-+h'],
                ['++c','+-c','--c','-+h'],
            ],
            2: [
                ['+-c','--c','-+h'],
                ['++c','+-c','--c','-+h'],
            ]
        }
    return composition_libraries

def get_dataset_symbol_from_name(core_dataset_name):
    dictionary = {
        'SIR': 'SIR',
        'PK': 'pk',
        'synthetic-tumor': 'tumor',
        'bike-sharing': 'bike',
        'tacrolimus-real': 'real_pharma',
        'HIV': 'HIV',
    }
    return dictionary[core_dataset_name]
