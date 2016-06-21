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

# Since the task is two steps, we take only one trial out of 2 (when reward is
# actually received, hence starting a trial 1)
records = records[:,1::2]
n_trial = len(experiment.task)/2


# -----------------------------------------------------------------------------

# Moving average over n trials
n = 10
R = records["reward"]
R = np.cumsum(R, axis=-1, dtype=float)
R[:,n:] = R[:,n:] - R[:,:-n]
R = R[:,n-1:]/n
n_trial = R.shape[1]

P_mean = np.mean(R, axis=0)
P_std = np.std(R, axis=0)


plt.figure(figsize=(16,6), facecolor="w")

ax = plt.subplot(111)
ax.patch.set_facecolor("w")
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_tick_params(direction="in")
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_tick_params(direction="in")

X = n+np.arange(n_trial)

plt.plot(X, P_mean, c='b', lw=2)
plt.plot(X, P_mean + P_std, c='b', lw=.5)
plt.plot(X, P_mean - P_std, c='b', lw=.5)
plt.fill_between(X, P_mean + P_std, P_mean - P_std, color='b', alpha=.1)

plt.text(n_trial+n, P_mean[-1], "%.2f" % P_mean[-1],
         ha="left", va="center", color="b")

plt.ylabel("Performance\n", fontsize=16)
plt.xlim(0,n+n_trial)
plt.ylim(0,1.1)

ax.axvspan(0,n, color='.975')
ax.axhline(1.0, color='.5', lw=.5, ls = "--")
ax.axhline(0.5, color='.5', lw=.5, ls = "--")

plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#plt.text(-1, P_mean[0], "%.2f" % P_mean[0],
#         ha="right", va="center", color="b")

plt.savefig("data/experiment-economics.pdf")
plt.show()

