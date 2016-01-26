# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import random
import numpy as np
from tqdm import tqdm
from task import Task
from model import Model

seed      = random.randint(0,1000)
np.random.seed(seed), random.seed(seed)

model     = Model("model-topalidou.json")
task      = Task("task-topalidou.json")
filename  = "experiment-topalidou-protocol-2.npy"
n_session = 100
n_trial   = len(task)
debug     = False

print("-"*30)
print("Seed:     %d" % seed)
print("Model:    %s" % model.filename)
print("Task:     %s" % task.filename)
print("Sessions: %d (%d trials)" % (n_session, 2 * n_session*n_trial))
print("-"*30)


records = np.zeros((n_session, 2, n_trial), dtype=task.records.dtype)
total   = records.size

if 0:
    with tqdm(total=total, leave=True, desc="Protocol 2", unit="trial", disable=debug) as bar:
        for index in range(n_session):
            model.setup()

            # Day 1 : GPi OFF
            g1 = model["GPi:cog → THL:cog"].gain
            g2 = model["GPi:mot → THL:mot"].gain
            for trial in task:
                bar.update(1)
                model.process(task, trial, debug=debug)
            records[index,0] = task.records


            # Day 2: GPi ON
            model["GPi:cog → THL:cog"].gain = 0
            model["GPi:mot → THL:mot"].gain = 0
            for trial in task:
                bar.update(1)
                model.process(task, trial, debug=debug)
            records[index,1] = task.records

            # Day 3: GPi ON
            model["GPi:cog → THL:cog"].gain = g1
            model["GPi:mot → THL:mot"].gain = g2
    np.save(filename, records)
else:
    import os.path, time
    print("Loading previous results")
    print("('%s', last modified on %s)" % (filename,time.ctime(os.path.getmtime(filename))))
    records = np.load(filename)


print("-"*30)
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
print("-"*30)
