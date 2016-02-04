# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import io
import sys
import json
import time
import random
import os.path
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from task import Task
from model import Model


class Experiment(object):
    def __init__(self, model, task, result, report,
                       n_session, n_block, seed=None):
        self.model_file  = model
        self.task_file   = task
        self.result_file = result
        self.report_file = report
        self.n_session   = n_session
        self.n_block     = n_block
        self.seed        = seed

        if self.seed is None:
            self.seed = random.randint(0,1000)
        np.random.seed(seed), random.seed(seed)

        self.model = Model(self.model_file)
        self.task  = Task(self.task_file)
        self.n_trial = len(self.task)
            

    def run(self, session, desc=""):

        # Command line argument parsing for the --force switch
        parser = argparse.ArgumentParser()
        parser.add_argument("--force", action='store_true')
        args = parser.parse_args()
        force =  args.force


        if os.path.exists(self.result_file) and not force:
            print("Reading report (%s)" % self.report_file)
            self.read_report()
            
        print("-"*30)
        print("Seed:     %d" % self.seed)
        print("Model:    %s" % self.model_file)
        print("Task:     %s" % self.task_file)
        print("Result:   %s" % self.result_file)
        print("Report:   %s" % self.report_file)
        n = self.n_session * self.n_block * self.n_trial
        print("Sessions: %d (%d trials)" % (self.n_session, n))
        print("-"*30)

        if not os.path.exists(self.result_file) or force:
            index = 0
            records = np.zeros((self.n_session, self.n_block, self.n_trial),
                               dtype=self.task.records.dtype)
            pool = Pool(4)
            for result in tqdm(pool.imap_unordered(session, [self,]*self.n_session),
                               total=self.n_session, leave=True, desc=desc, unit="session",):
                records[index] = result
                index += 1
            pool.close()
            
            print("Saving results (%s)" % self.result_file)
            np.save(self.result_file, records)
            print("Writing report (%s)" % self.report_file)
            self.write_report()
        else:
            print("Loading previous results")
            print(' → "%s"' % (self.result_file))
            records = np.load(self.result_file)

        print("-"*30)
        return records

        
    def write_report(self):
        report = { "seed"      : self.seed,
                   "n_session" : self.n_session,
                   "n_block"   : self.n_block,
                   "n_trial"   : self.n_trial,
                   "task"      : self.task.parameters,
                   "model"     : self.model.parameters }
        with io.open(self.report_file, 'w', encoding='utf8') as fp:
            json.dump(report, fp, indent=4, ensure_ascii=False)

    def read_report(self):
        with io.open(self.report_file, 'r', encoding='utf8') as fp:
            report = json.load(fp)
        self.seed      = report["seed"]
        self.n_session = report["n_session"]
        self.n_block   = report["n_block"]
        self.n_trial   = report["n_trial"]
