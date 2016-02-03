# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from experiment import Experiment


def session(exp):
    exp.model.setup()
    records = np.zeros((exp.n_block, exp.n_trial), dtype=exp.task.records.dtype)

    # Day 1 : GPi ON
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    records[0] = exp.task.records

    # Day 2: GPi ON
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    records[1] = exp.task.records

    return records


experiment = Experiment(model  = "model-topalidou.json",
                        task   = "task-topalidou.json",
                        result = "data/experiment-topalidou-control.npy",
                        report = "data/experiment-topalidou-control.txt",
                        n_session = 25, n_block = 2, seed = None)
records = experiment.run(session, "Control")


# Save performance (one column per session)
# -----------------------------------------------------------------------------
# P = np.squeeze(records["best"][:,0])
# np.savetxt("data/experiment-topalidou-control-D1-P.csv", P.T, fmt="%d", delimiter=",")
# P = np.squeeze(records["best"][:,1])
# np.savetxt("data/experiment-topalidou-control-D2-P.csv", P.T, fmt="%d", delimiter=",")
# P = np.squeeze(records["RT"][:,0])
# np.savetxt("data/experiment-topalidou-control-D1-RT.csv", P.T, fmt="%.4f", delimiter=",")
# P = np.squeeze(records["RT"][:,1])
# np.savetxt("data/experiment-topalidou-control-D2-RT.csv", P.T, fmt="%.4f", delimiter=",")


# Textual results
# -----------------------------------------------------------------------------
P = np.squeeze(records["best"][:,0,:25])
P = P.mean(axis=len(P.shape)-1)
print("D1 start: %.3f ± %.3f" % (P.mean(), P.std()))
P = np.squeeze(records["best"][:,0,-25:])
P = P.mean(axis=len(P.shape)-1)
print("D1 end:   %.3f ± %.3f" % (P.mean(), P.std()))

P = np.squeeze(records["RT"][:,0])
print("D1 mean RT: %.3f ± %.3f" % (P.mean(), P.std()))

print()

P = np.squeeze(records["best"][:,1,:25])
P = P.mean(axis=len(P.shape)-1)
print("D2 start: %.3f ± %.3f" % (P.mean(), P.std()))
P = np.squeeze(records["best"][:,1,-25:])
P = P.mean(axis=len(P.shape)-1)
print("D2 end:   %.3f ± %.3f" % (P.mean(), P.std()))
P = np.squeeze(records["RT"][:,1])
print("D2 mean RT: %.3f ± %.3f" % (P.mean(), P.std()))

print("-"*30)


# Graphical results
# -----------------------------------------------------------------------------
from figures import *
figure_H_P(records, [1,1], "Control", "data/experiment-topalidou-control-H-P.pdf")
figure_H_RT(records, [1,1], "Control", "data/experiment-topalidou-control-H-RT.pdf")
# figure_P(records, [1,1], "Control", "data/experiment-topalidou-control-P.pdf")
# figure_RT(records, [1,1], "Control", "data/experiment-topalidou-control-RT.pdf")

