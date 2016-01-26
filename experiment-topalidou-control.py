# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import sys
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from task import Task
from model import Model

seed = random.randint(0,1000)
np.random.seed(seed), random.seed(seed)

model     = Model("model-topalidou.json")
task      = Task("task-topalidou.json")
filename  = "experiment-topalidou-control.npy"
n_session = 100
n_trial   = len(task)
debug     = False
records   = np.zeros((n_session, 2, n_trial), dtype=task.records.dtype)
total     = records.size


# -----------------------------------------------------------------------------
print("-"*30)
print("Seed:     %d" % seed)
print("Model:    %s" % model.filename)
print("Task:     %s" % task.filename)
print("Sessions: %d (%d trials)" % (n_session, 2 * n_session*n_trial))
print("-"*30)

def session(*args):
    model.setup()
    records = np.zeros((2, n_trial), dtype=task.records.dtype)

    # Day 1 : GPi ON
    for trial in task:
        model.process(task, trial, debug=debug)
    records[0] = task.records

    # Day 2: GPi ON
    for trial in task:
        model.process(task, trial, debug=debug)
    records[1] = task.records

    return records


if 1:
    index = 0
    records = np.zeros((n_session, 2, n_trial), dtype=task.records.dtype)
    pool = Pool(4)
    for result in tqdm(pool.imap_unordered(session, [1,]*n_session),
                       total=n_session, leave=True, desc="Control", unit="session",):
        records[index] = result
        index += 1
    pool.close()                       
    np.save(filename, records)
else:
    import os.path, time
    print("Loading previous results")
    print("('%s', last modified on %s)" % (filename,time.ctime(os.path.getmtime(filename))))
    records = np.load(filename)

print("-"*30)
# -----------------------------------------------------------------------------


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
