# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import os
import unittest
import numpy as np

import dotdot
from experiment import Experiment


def session(exp):
    exp.model.setup()
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    return exp.task.records

def run_model():
    experiment = Experiment(model  = "../experiments/model-guthrie.json",
                            task   = "../experiments/task-guthrie.json",
                            result = "data/test-experiment-guthrie.npy",
                            report = "data/test-experiment-guthrie.txt",
                            n_session = 8, n_block = 1, seed = 1,
                            rootdir=os.path.dirname(__file__)) # for unittest and nosetests.
    records = experiment.run(session, save=True, force=True, parse=False)
    records = np.squeeze(records)
    mean = np.mean(records["best"], axis=0)[-1]
    assert mean >= 0.85

def result_filename(suffix='', ext='npy'):
    return os.path.join(os.path.dirname(__file__),
                        'data/test-experiment-guthrie{}.{}'.format(suffix, ext))


class DanaTests(unittest.TestCase):
    """Verifying that results can be reproduced exactly."""

    def test_reproducible(self):
        # removing existing results
        for ext in ['npy', 'txt']:
            for suffix in ['', '_ref']:
                if os.path.exists(result_filename(suffix=suffix, ext=ext)):
                    os.remove(result_filename(suffix=suffix, ext=ext))

        # first run of the model
        run_model()
        # moving the files out of the way
        for ext in ['npy', 'txt']:
            os.rename(result_filename(ext=ext), result_filename(suffix='_ref', ext=ext))
        # second run of the model: should be identical
        run_model()

        # comparing results
        for ext in ['npy', 'txt']:
            with open(result_filename(ext=ext), 'rb') as f:
                run0 = f.read()
            with open(result_filename(suffix='_ref', ext=ext), 'rb') as f:
                run1 = f.read()
            assert run0 == run1


if __name__ == '__main__':
    unittest.main()
