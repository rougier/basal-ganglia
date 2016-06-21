# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Context dependent, two steps, two arms bandit task:

              (V=1.0)  +-- choice 1: P = 1.00 | strategy A/1: V=1.0
          +-- state A -|                      | 
          |            +-- choice 2: P = 0.00 | strategy A/2: V=0.0
       choice A
          |
state 0 --+
          |
       choice B
          |            +-- choice 1: P = 0.66 | strategy B/1: V=0.66
          +-- state B -|                      |
              (V=.66)  +-- choice 2: P = 0.33 | strategy B/2: V=0.33


Initial state
=============

      A B 1 2
     +-+-+-+-+
     |1|1| | |
     +-+-+-+-+

+-+  +-+-+-+-+
|1|  |1| |1| | -> Strategy A/1
+-+  +-+-+-+-+
|1|  |1| | |1| -> Strategy A/2 (*)
+-+  +-+-+-+-+
|1|  | |1|1| | -> Strategy B/1
+-+  +-+-+-+-+
|1|  | |1| |1| -> Strategy B/2
+-+  +-+-+-+-+


State A or B
============

      A B 1 2
     +-+-+-+-+
     | | |1|1|
     +-+-+-+-+

+-+  +-+-+-+-+
| |  |1| |1| | -> Strategy A/1
+-+  +-+-+-+-+
|1|  |1| | |1| -> Strategy A/2 (* working memory)
+-+  +-+-+-+-+
| |  | |1|1| | -> Strategy B/1
+-+  +-+-+-+-+
| |  | |1| |1| -> Strategy B/2
+-+  +-+-+-+-+
"""
import os
import json
import random
import numpy as np


class Task(object):

    def __init__(self, filename="task-guthrie.json"):
        self.index       = None
        self.index_start = None
        self.index_stop  = None

        self.filename = filename
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)

        # create data/ directory if it does not exists.
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)

        np.random.seed()
            
        self.setup()


    def block(self,index):
        self.index_start = self.blocks[index][0]-1
        self.index_stop  = self.blocks[index][1]
        self.index = self.index_start
        return self

    def setup(self):

        _ = self.parameters

        blocks = []
        for name in _["session"]:
            blocks.append(_[name])

        # Get total number of trials
        n = 0
        self.blocks = []
        start,stop = 0, 0
        for block in blocks:
            start = stop
            stop += block["n_trial"]
            self.blocks.append((start,stop))
            n += block["n_trial"]

        # Build corresponding arrays
        self.trials = np.zeros(n, [("mot",   float, 4),
                                   ("cog",   float, 4),
                                   ("ass",   float, (4,4)),
                                   ("rwd",   float, (2,2)),
                                   ("rnd",   float, 1) ] )
        self.records  = np.zeros(n, [("choice",  float, 1),
                                     ("best",    float, 1),
                                     ("valid",   float, 1),
                                     ("RT",      float, 1),
                                     ("state",   float, 1),
                                     ("reward",  float, 1),
                                     # These values must be collected from the model
                                     ("value", float, 4),
                                     ("CTX:cog -> CTX:ass", float, 4),
                                     ("CTX:cog -> STR:cog", float, 4)] )

        # We draw all random probabilities at once (faster)
        self.trials["rnd"] = np.random.uniform(0,1,n)

        # This does not change
        self.trials["ass"] = [[1,0, 1,0],
                              [1,0, 0,1],
                              [0,1, 1,0],
                              [0,1, 0,1]]

        # Build actual trials
        index = 0
        for block in blocks:
            n = block["n_trial"]
            for i in range(n):
                trial = self.trials[index]
                trial["rwd"] = block["rwd"]
                index += 1
        

    def __iter__(self):
        if self.index_start is None:
            self.setup()
            self.index_start = 0
            self.index_stop  = len(self)
            self.index = self.index_start
            self.state = 0
        return self      

    
    def __next__(self,n=1):
        index = self.index
        if index >= len(self):
            raise StopIteration

        trials = self.trials
        if self.state == 0:
            # Any strategy can be selected
            trials[index]["cog"] = [1,1,1,1]
            # First motor choice
            trials[index]["mot"] = [1,1,0,0]
        else:
            # Here we keep the last chosen strategy active (working memory)
            trials[index]["cog"] = [0,0,0,0]
            trials[index]["cog"][self.cog] = 1
            trials[index]["mot"] = [0,0,1,1]
            
        self.index += 1
        return self.trials[index]

    
    def process(self, trial, cog=-1, mot=-1, RT=0.0, model=None, debug=False):

        index = self.index-1
        self.records[index]["RT"] = RT
        
        # State 0 : motor choice is 0 or 1
        if self.state == 0:
            if mot == 0:
                self.state = 1
            else:
                self.state = 2
            self.cog = cog
            return None
        
        # State 1 or 2
        P = trial["rwd"][self.state-1, mot-2]
        reward = int(trial["rnd"] < P)
        self.state = 0

        # Record everything
        # self.records[index]["RT"] = RT
        # self.records[self.index]["best"] = best
        # self.records[self.index]["valid"] = valid
        # self.records[self.index]["choice"] = choice
        self.records[index]["reward"] = reward

        if model is not None:
            self.records[index]["value"] = model["value"]
            self.records[index]["CTX:cog -> CTX:ass"] = model["CTX:cog → CTX:ass"].weights
            self.records[index]["CTX:cog -> STR:cog"] = model["CTX:cog → STR:cog"].weights

        return reward
    
    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        return self.trials[index]



# -----------------------------------------------------------------------------
if __name__ == "__main__":
    task = Task()

    for trial in task:
        # Best choice
        if trial["mot"][0] == 1:
            choice = 0
        else:
            choice = 2
            
        # Random choice
        # if trial["mot"][0] == 1:
        #     choice = np.random.randint(0,2)
        # else:
        #     choice = np.random.randint(2,4)

        
        # Process choice
        reward = task.process(trial, choice, debug=True)
        print (reward)

