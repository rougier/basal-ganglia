# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from experiment import Experiment

def session(exp):
    exp.model.setup()
    records = np.zeros((exp.n_block, exp.n_trial), dtype=exp.task.records.dtype)

    # Day 1 : GPi OFF
    g1 = exp.model["GPi:cog → THL:cog"].gain
    g2 = exp.model["GPi:mot → THL:mot"].gain
    exp.model["GPi:cog → THL:cog"].gain = 0
    exp.model["GPi:mot → THL:mot"].gain = 0
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    records[0] = exp.task.records

    # Day 2: GPi ON
    exp.model["GPi:cog → THL:cog"].gain = g1
    exp.model["GPi:mot → THL:mot"].gain = g2
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    records[1] = exp.task.records

    # Day 1 : GPi OFF
    exp.model["GPi:cog → THL:cog"].gain = 0
    exp.model["GPi:mot → THL:mot"].gain = 0
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    records[2] = exp.task.records
        
    return records


experiment = Experiment(model = "model-topalidou.json",
                        task = "task-topalidou.json",
                        result = "experiment-topalidou-protocol-2.npy",
                        report = "experiment-topalidou-protocol-2.txt",
                        n_session = 100, n_block = 3, seed = None)
records = experiment.run(session, "Protocol 2")


# Analyze
P = np.squeeze(records["best"][:,0,:25])
P = P.mean(axis=len(P.shape)-1)
print("D1 start: %.3f ± %.3f" % (P.mean(), P.std()))
P = np.squeeze(records["best"][:,0,-25:])
P = P.mean(axis=len(P.shape)-1)
print("D1 end:   %.3f ± %.3f" % (P.mean(), P.std()))

print()

P = np.squeeze(records["best"][:,1,:25])
P = P.mean(axis=len(P.shape)-1)
print("D2 start: %.3f ± %.3f" % (P.mean(), P.std()))
P = np.squeeze(records["best"][:,1,-25:])
P = P.mean(axis=len(P.shape)-1)
print("D2 end:   %.3f ± %.3f" % (P.mean(), P.std()))

print()

P = np.squeeze(records["best"][:,2,:25])
P = P.mean(axis=len(P.shape)-1)
print("D3 start: %.3f ± %.3f" % (P.mean(), P.std()))
P = np.squeeze(records["best"][:,2,-25:])
P = P.mean(axis=len(P.shape)-1)
print("D3 end:   %.3f ± %.3f" % (P.mean(), P.std()))
print("-"*30)



