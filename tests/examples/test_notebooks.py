#!/usr/bin/env python3
'''Test functionality of example notebooks.'''
import inspect
import os
import pathlib

import pytest


def clean_example(trash):
    '''Clean the example folder after running the script.'''
    for it in trash:
        path = pathlib.Path(it)
        if path.exists():
            path.unlink()
    return


@pytest.mark.slow
@pytest.mark.extras
@pytest.mark.parametrize('name, trash', [('08_visualizer_extra', []),
                                         ('10_domain_generation', []),
                                         ('13_wannier_localization', ['CH4_WO_0.cube',
                                                                      'CH4_WO_1.cube',
                                                                      'CH4_WO_2.cube',
                                                                      'CH4_WO_3.cube'])])
def test_notebooks(name, trash):
    '''Test the execution of a given Jupyter notebook.'''
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbformat import read

    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    os.chdir(file_path.joinpath(f'../../examples/{name}'))

    with open(f'{name}.ipynb', 'r') as fh:
        nb = read(fh, as_version=4)
        ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
        assert ep.preprocess(nb) is not None

    clean_example(trash)
    return


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
