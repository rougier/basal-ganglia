# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from experiment import Experiment

def session(exp):
    exp.model.setup()
    records = np.zeros((exp.n_block, exp.n_trial),
                       dtype=exp.task.records.dtype)

    # Day 1 : GPi ON
    g1 = exp.model["GPi:cog → THL:cog"].gain
    g2 = exp.model["GPi:mot → THL:mot"].gain
    for trial in exp.task:
        exp.model.process(exp.task, trial, model=exp.model)
    records[0] = exp.task.records

    # Day 2: GPi OFF
    exp.model["GPi:cog → THL:cog"].gain = 0
    exp.model["GPi:mot → THL:mot"].gain = 0
    for trial in exp.task:
        exp.model.process(exp.task, trial, model=exp.model)
    records[1] = exp.task.records

    return records


experiment = Experiment(model = "model-topalidou.json",
                        task = "task-topalidou.json",
                        result = "data/experiment-topalidou-protocol-1.npy",
                        report = "data/experiment-topalidou-protocol-1.txt",
                        n_session = 25, n_block = 2, seed = None)
records = experiment.run(session, "Protocol 1")



# Save performance (one column per session)
# -----------------------------------------------------------------------------
# P = np.squeeze(records["best"][:,0])
# np.savetxt("data/experiment-topalidou-protocol-1-D1-P.csv", P.T, fmt="%d", delimiter=",")
# P = np.squeeze(records["best"][:,1])
# np.savetxt("data/experiment-topalidou-protocol-1-D2-P.csv", P.T, fmt="%d", delimiter=",")
# P = np.squeeze(records["RT"][:,0])
# np.savetxt("data/experiment-topalidou-protocol-1-D1-RT.csv", P.T, fmt="%.4f", delimiter=",")
# P = np.squeeze(records["RT"][:,1])
# np.savetxt("data/experiment-topalidou-protocol-1-D2-RT.csv", P.T, fmt="%.4f", delimiter=",")

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

title = "Protocol 1"
filename = "data/experiment-topalidou-protocol-1-P-all.pdf"
figure_P_all(records, [1, 0], title=title, filename=filename)


#figure_H_P(records, [1,0], "Protocol 1", "data/experiment-topalidou-protocol-1-H-P.pdf")
#figure_H_RT(records, [1,0], "Protocol 1", "data/experiment-topalidou-protocol-1-H-RT.pdf")

#figure_P(records, [1,0], "Protocol 1", "data/experiment-topalidou-protocol-1-P.pdf")

# filename = "data/experiment-topalidou-protocol-1-P-individual.pdf"
# figure_P_individual(records, GPi=[1,0], 
#                     title="Protocol 1",
#                     filename=filename)

# filename = "data/experiment-topalidou-protocol-1-RT-individual.pdf"
# figure_RT_individual(records, GPi=[1,0], 
#                     title="Protocol 1",
#                     filename=filename)


# figure_RT(records, [1,0], "Protocol 1", "data/experiment-topalidou-protocol-1-RT.pdf")

