import os
import re
from collections import defaultdict
from typing import List

from modcma import evaluate_bbob

from ioh import ProblemClass
import json


def main(
        *,
        fid: int,
        dim: int,
        data_folder: str = 'output',
        exp_repetitions: int = 1,
        seed: int = 42,
        **parameters
):
    evaluate_bbob(
        fid,
        dim,
        logging=True,
        seed=seed,
        data_folder=data_folder,
        iterations=exp_repetitions,
        # selecting BBOB class of problems
        problem_class=ProblemClass.BBOB,
        **parameters
    )


def get_dirs_paths(path, filt=None):
    dirs = next(os.walk(path))[1]
    if filt is not None:
        pass
    return [os.path.join(path, x) for x in dirs]


def get_files_paths(path, filt=None):
    files = next(os.walk(path))[2]
    if filt is not None:
        pass
    return [os.path.join(path, x) for x in files]


def get_recursive_files(path):
    # [
    #   ('.', ['data_f2_Ellipsoid'], ['IOHprofiler_f2_Ellipsoid.json']),
    #   ('./data_f2_Ellipsoid', [], ['IOHprofiler_f2_DIM2.dat'])
    # ]
    m = []
    for root, _, files in os.walk(path):
        for file in files:
            m.append(os.path.join(root, file))
    return m


def merge_json(json1, json2):
    raise NotImplementedError('todo')


def merge_ioh_dirs(destination: str, sources: List[str]):
    name_of_function = re.compile(r'_f\d+_.*')

    fdat_paths, fjson_paths = [], []

    for source in sources:
        files = get_recursive_files(source)
        assert len(files) == 2
        fdat, fjson = files
        fdat_paths.append(fdat)
        fjson_paths.append(fjson)

    name_long = os.path.basename(fdat_paths[0])
    assert name_long.endswith('.dat')
    name_long = name_long[:-len('.dat')]
    assert name_long.startswith('IOHprofiler_')
    name = name_long[len('IOHprofiler_'):]

    # checking names equal
    for path in fdat_paths:
        assert os.path.basename(path) == f"{name_long}.dat"
    for path in fjson_paths:
        assert os.path.basename(path) == f"{name_long}.json"

    # data
    data_dir = os.path.join(destination, f'data_{name}')
    os.makedirs(data_dir)

    with open(os.path.join(data_dir, name_long + '.dat'), 'w') as f_write:
        for path in fdat_paths:
            with open(path, 'r') as f_read:
                f_write.write(f_read.read())
            f_write.write('\n')
        f_write.seek(-1, 2)
        f_write.truncate()

    with open(os.path.join(destination, name_long + '.json'), 'w') as f_write:
        data = None

        for path in fjson_paths:
            with open(path, 'r') as f_read:
                if data is None:
                    data = json.load(f_read)
                else:
                    data = merge_json(data, json.load(f_read))
        json.dump(data, f_write)


def normalize_output(input_path, output_path):
    '''
        situation:
            input_path -> output/<experiment_name>/
            in <input_path>:
                <job_2433>/F2_2D/IOHprofiler_f2_Ellipsoid.json
                <job_3434>/F2_2D/data_f2_Ellipsoid/IOHprofiler_f2_DIM2.dat
                ... job_x2
                ... job_x3
                ... job_x4

            result:
            in <output_path>
                F2_2D/IOHprofiler_f2_Ellipsoid.json
                F2_2D/data_f2_Ellipsoid/IOHprofiler_f2_DIM2.dat
                F1_2D ...
                F3_2D ...
                F1_1D ...
    '''
    filter_job_dir = None
    filter_comp_dir = None

    comp_dir_pattern = re.compile(r'(F\d+_\d+D)-?.*')
    comp_dir_full_name_pattern = re.compile(f'_.*_.*',)

    tomerge = defaultdict(list)

    # 1. find all folders that computes the same thing +- seed
    for job_dir in get_dirs_paths(input_path, filt=filter_job_dir):
        for comp_dir in get_dirs_paths(job_dir, filt=filter_comp_dir):
            match = comp_dir_pattern.match(os.path.basename(comp_dir))
            if match:
                tomerge[match.group(1)].append(comp_dir)
            else:
                print(f'Warning cannot match {comp_dir}')

    # 2. merge the directories into output
    for comp_dir_name, in_folders in tomerge.items():
        comp_dir = os.path.join(output_path, comp_dir_name)
        os.makedirs(comp_dir)
        merge_ioh_dirs(comp_dir, in_folders)




