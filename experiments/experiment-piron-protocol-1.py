# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from experiment import Experiment

def session(exp):
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


experiment = Experiment(model  = "model-topalidou.json",
                        task   = "task-piron.json",
                        result = "data/experiment-piron-protocol-1.npy",
                        report = "data/experiment-piron-protocol-1.txt",
                        n_session = 100, n_block = 1, seed = 1)
records = experiment.run(session, "Protocol 1")
records = np.squeeze(records)

# -----------------------------------------------------------------------------
w = 5
start,end = experiment.task.blocks[0]
P = np.mean(records["best"][:,end-w:end],axis=1)
mean,std = np.mean(P), np.std(P)
print("Acquisition (HC):  %.2f ± %.2f" % (mean,std))

start,end = experiment.task.blocks[1]
P = np.mean(records["best"][:,end-w:end],axis=1)
mean,std = np.mean(P), np.std(P)
print("Test (HC, GPi On): %.2f ± %.2f" % (mean,std))

start,end = experiment.task.blocks[2]
P = np.mean(records["best"][:,end-w:end],axis=1)
mean,std = np.mean(P), np.std(P)
print("Test (NC, GPi On): %.2f ± %.2f" % (mean,std))

# -----------------------------------------------------------------------------
P_mean, P_std = [], []
RT_mean, RT_std = [], []
for i in range(records.shape[1]-w):
    P = np.mean(records["best"][:,i:i+w],axis=1)
    P_mean.append(np.mean(P))
    P_std.append(np.std(P))
    RT = np.mean(records["RT"][:,i:i+w],axis=1)
    RT_mean.append(np.mean(RT))
    RT_std.append(np.std(RT))
    
P_mean  = np.array(P_mean)
P_std   = np.array(P_std)
RT_mean = np.array(RT_mean)*1000
RT_std  = np.array(RT_std)*1000
X       = w + np.arange(len(P_mean))


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

# X = 1+np.arange(n_trial)


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

plt.savefig("data/experiment-piron-protocol-1.pdf")
plt.show()
