# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from experiment import Experiment

def session(exp):
    exp.model.setup()
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    return exp.task.records

def test_model():
    experiment = Experiment(model = "experiments/model-guthrie.json",
                            task = "experiments/task-guthrie.json",
                            result = "test-experiment-guthrie.npy",
                            report = "test-experiment-guthrie.txt",
                            n_session = 10, n_block = 1, seed = 1)
    records = experiment.run(session, save=False, force=True, parse=False)
    records = np.squeeze(records)    
    P_mean = np.mean(records["best"], axis=0)
    assert P_mean[-1] > 0.9
