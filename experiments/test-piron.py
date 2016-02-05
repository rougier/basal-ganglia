# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from experiment import Experiment


def session_1(exp):
    exp.model.setup()
    exp.task.setup()

    # Block 1 : GPi ON
    g1 = exp.model["GPi:cog → THL:cog"].gain
    g2 = exp.model["GPi:mot → THL:mot"].gain
    for trial in exp.task.block(0):
        exp.model.process(exp.task, trial)

    # Block 2 : GPi ON
    exp.model["GPi:cog → THL:cog"].gain = g1
    exp.model["GPi:mot → THL:mot"].gain = g2
    for trial in exp.task.block(1):
        exp.model.process(exp.task, trial)

    # Block 3 : GPi ON
    exp.model["GPi:cog → THL:cog"].gain = g1
    exp.model["GPi:mot → THL:mot"].gain = g2
    for trial in exp.task.block(2):
        exp.model.process(exp.task, trial)
        
    return exp.task.records

def session_2(exp):
    exp.model.setup()
    exp.task.setup()

    # Block 1 : GPi ON
    g1 = exp.model["GPi:cog → THL:cog"].gain
    g2 = exp.model["GPi:mot → THL:mot"].gain
    for trial in exp.task.block(0):
        exp.model.process(exp.task, trial)

    # Block 2 : GPi OFF
    exp.model["GPi:cog → THL:cog"].gain = 0
    exp.model["GPi:mot → THL:mot"].gain = 0
    for trial in exp.task.block(1):
        exp.model.process(exp.task, trial)

    # Block 3 : GPi OFF
    exp.model["GPi:cog → THL:cog"].gain = 0
    exp.model["GPi:mot → THL:mot"].gain = 0
    for trial in exp.task.block(2):
        exp.model.process(exp.task, trial)
       
    return exp.task.records

def test_protocol_1():
    experiment = Experiment(model  = "experiments/model-topalidou.json",
                            task   = "experiments/task-piron.json",
                            result = "data/tmp.npy",
                            report = "data/tmp.txt",
                            n_session = 25, n_block = 1, seed = 1)
    records = experiment.run(session_1, save=False, force=True, parse=False)
    records = np.squeeze(records)

    w = 5
    start,end = experiment.task.blocks[0]
    P = np.mean(records["best"][:,end-w:end],axis=1)
    mean,std = np.mean(P), np.std(P)
    print("Acquisition (HC):  %.2f ± %.2f" % (mean,std))
    assert mean > 0.95
    
    start,end = experiment.task.blocks[1]
    P = np.mean(records["best"][:,end-w:end],axis=1)
    mean,std = np.mean(P), np.std(P)
    print("Test (HC, GPi Off): %.2f ± %.2f" % (mean,std))
    assert mean > 0.95
    
    start,end = experiment.task.blocks[2]
    P = np.mean(records["best"][:,end-w:end],axis=1)
    mean,std = np.mean(P), np.std(P)
    assert mean > 0.95
    print("Test (NC, GPi Off): %.2f ± %.2f" % (mean,std))
    

