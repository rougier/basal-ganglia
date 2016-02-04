# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import random
import numpy as np
import matplotlib.pyplot as plt
from task import Task
from model import Model

seed = random.randint(0,1000)
np.random.seed(seed)
random.seed(seed)

model = Model("model-topalidou.json")
task  = Task("task-guthrie.json")

print("-"*30)
print("Seed:     %d" % seed)
print("Model:    %s" % model.filename)
print("Task:     %s" % task.filename)
print("-"*30)

model["GPi:cog → THL:cog"].gain = 0
model["GPi:mot → THL:mot"].gain = 0
trial = task[0]
model.process(task, trial, stop=False, debug=False)

cog = model["CTX"]["cog"].history[:3000]
mot = model["CTX"]["mot"].history[:3000]


fig = plt.figure(figsize=(12,5))
plt.subplots_adjust(bottom=0.15)

duration = 3.0
timesteps = np.linspace(0, duration, 3000)

fig.patch.set_facecolor('.9')
ax = plt.subplot(1,1,1)

plt.plot(timesteps, cog[:,0], c='r', label="Cognitive Cortex")
plt.plot(timesteps, cog[:,1], c='r')
plt.plot(timesteps, cog[:,2], c='r')
plt.plot(timesteps, cog[:,3], c='r')
plt.plot(timesteps, mot[:,0], c='b', label="Motor Cortex")
plt.plot(timesteps, mot[:,1], c='b')
plt.plot(timesteps, mot[:,2], c='b')
plt.plot(timesteps, mot[:,3], c='b')

plt.title("Single trial (GPi OFF)")
plt.xlabel("Time (seconds)")
plt.ylabel("Activity (Hz)")
plt.legend(frameon=False, loc='upper left')
plt.xlim(0.0,duration)
plt.ylim(-10.0,60.0)
plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
           ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5','3.0'])
plt.savefig("data/single-trial-GPi-OFF.pdf")
plt.show()
