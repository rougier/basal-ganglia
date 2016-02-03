# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from experiment import Experiment

def session(exp):
    exp.model.setup()
    for trial in exp.task:
        exp.model.process(exp.task, trial)
    return exp.task.records

experiment = Experiment(model = "model-guthrie.json",
                        task = "task-guthrie.json",
                        result = "data/experiment-guthrie.npy",
                        report = "data/experiment-guthrie.txt",
                        n_session = 250, n_block = 1, seed = None)
records = experiment.run(session, "Protocol 1")
records = np.squeeze(records)

# -----------------------------------------------------------------------------
P_mean = np.mean(records["best"], axis=0)
P_std = np.std(records["best"], axis=0)
RT_mean = np.mean(records["RT"]*1000, axis=0)
RT_std = np.std(records["RT"]*1000, axis=0)


plt.figure(figsize=(16,10), facecolor="w")
n_trial = len(experiment.task)

ax = plt.subplot(211)
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

ax = plt.subplot(212)

ax.patch.set_facecolor("w")
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_tick_params(direction="in")
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_tick_params(direction="in")

X = 1+np.arange(n_trial)
plt.plot(X, RT_mean, c='r', lw=2)
plt.plot(X, RT_mean + RT_std, c='r', lw=.5)
plt.plot(X, RT_mean - RT_std, c='r', lw=.5)
plt.fill_between(X, RT_mean + RT_std, RT_mean - RT_std, color='r', alpha=.1)
plt.xlabel("Trial number", fontsize=16)
plt.ylabel("Response time (ms)\n", fontsize=16)
plt.xlim(1,n_trial)
plt.yticks([400,500,600,700,800,1000])

plt.text(n_trial+1, RT_mean[-1], "%d ms" % RT_mean[-1],
         ha="left", va="center", color="r")
plt.text(0, RT_mean[0], "%d" % RT_mean[0],
         ha="right", va="center", color="r")

plt.savefig("data/experiment-guthrie.pdf")
plt.show()

