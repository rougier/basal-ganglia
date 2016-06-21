# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from task import Task
from model import Model
from experiment import Experiment

def session(exp):
    exp.model.setup()
    model = Model(exp.model_file)
    task = Task(exp.task_file)
    for trial in task:
        exp.model.process(task=task, trial=trial, model = model)
    return task.records

experiment = Experiment(model = "model-guthrie.json",
                        task = "task-economics.json",
                        result = "data/experiment-economics.npy",
                        report = "data/experiment-economics.txt",
                        n_session = 50, n_block = 1, seed = None)
records = experiment.run(session, "Progress")

records = np.squeeze(records)
records = records[:,1::2]
n_trial = len(experiment.task)/2


# -----------------------------------------------------------------------------
#P_mean = np.mean(records["reward"]*2, axis=0)
#P_std = np.std(records["reward"]*2, axis=0)
P_mean = np.mean(records["reward"], axis=0)
P_std = np.std(records["reward"], axis=0)

RT_mean = np.mean(records["RT"]*1000, axis=0)
RT_std = np.std(records["RT"]*1000, axis=0)



plt.figure(figsize=(16,6), facecolor="w")


ax = plt.subplot(111)
ax.patch.set_facecolor("w")
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_tick_params(direction="in")
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_tick_params(direction="in")

X = 1+np.arange(n_trial)

plt.plot(X, P_mean, c='b', lw=2)
plt.plot(X, P_mean + P_std, c='b', lw=.5)
plt.plot(X, P_mean - P_std, c='b', lw=.5)
plt.fill_between(X, P_mean + P_std, P_mean - P_std, color='b', alpha=.1)

plt.text(n_trial+1, P_mean[-1], "%.2f" % P_mean[-1],
         ha="left", va="center", color="b")

plt.ylabel("Performance\n", fontsize=16)
plt.xlim(1,n_trial)
plt.ylim(0,1.25)

plt.yticks([ 0.0,   0.2,   0.4,  0.6, 0.8,   1.0])
plt.text(0, P_mean[0], "%.2f" % P_mean[0],
         ha="right", va="center", color="b")


plt.savefig("data/experiment-guthrie.pdf")
plt.show()

