# -----------------------------------------------------------------------------
# Copyright (c) 2016, Nicolas P. Rougier
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import io
import sys
import json
import time
import os
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from task import Task
from model import Model


class Experiment(object):
    def __init__(self, model, task, result, report,
                       n_session, n_block, seed=None, rootdir=None):
        """Initialize an exeperiment.

        rootdir: root directory for the `model`, `task`, `result` and `report`
                 filepath. If None, defaults to the directory of the main script.
        """
        self.rootdir = rootdir
        if rootdir is None:
            from __main__ import __file__ as __mainfile__
            self.rootdir = os.path.dirname(__mainfile__)

        self.model_file  = os.path.abspath(os.path.join(self.rootdir, model))
        self.task_file   = os.path.abspath(os.path.join(self.rootdir, task))
        self.result_file = os.path.abspath(os.path.join(self.rootdir, result))
        self.report_file = os.path.abspath(os.path.join(self.rootdir, report))
        self.n_session   = n_session
        self.n_block     = n_block
        self.seed        = seed

        if self.seed is None:
            self.seed = np.random.randint(0, 1000)
        np.random.seed(seed)

        self.model = Model(self.model_file)
        self.task  = Task(self.task_file)
        self.n_trial = len(self.task)


    def run(self, session, desc="", save=True, force=False, parse=True):

        # Command line argument parsing for the --force switch
        if parse:
            parser = argparse.ArgumentParser()
            parser.add_argument("--force", action='store_true')
            args = parser.parse_args()
            force = args.force

        if os.path.exists(self.result_file) and not force:
            print("Reading report (%s)" % self.report_file)
            self.read_report()

        print("-"*30)
        print("Seed:     {}".format(self.seed))
        print("Model:    {}".format(self.model_file))
        print("Task:     {}".format(self.task_file))
        print("Result:   {}".format(self.result_file))
        print("Report:   {}".format(self.report_file))
        n = self.n_session * self.n_block * self.n_trial
        print("Sessions: {} ({} trials)".format(self.n_session, n))
        print("-"*30)

        if not os.path.exists(self.result_file) or force:
            index = 0
            records = np.zeros((self.n_session, self.n_block, self.n_trial),
                               dtype=self.task.records.dtype)

            n_workers = multiprocessing.cpu_count() # depends on your hardware
            pool = multiprocessing.Pool(n_workers)
            # different seed for different sessions
            seeds = np.random.randint(0, 1000000000, size=self.n_session)
            session_args = [(self, session, seed) for seed in seeds]

            for result in tqdm(pool.imap(self.session_init, session_args),
                               total=self.n_session, leave=True, desc=desc, unit="session",):
                records[index] = result
                index += 1
            pool.close()

            if save:
                print("Saving results ({})".format(self.result_file))
                if not os.path.isdir(os.path.dirname(self.result_file)):
                    os.makedirs(os.path.dirname(self.result_file))
                np.save(self.result_file, records)
                if not os.path.isdir(os.path.dirname(self.report_file)):
                    os.makedirs(os.path.dirname(self.report_file))
                print("Writing report ({})".format(self.report_file))
                self.write_report()
                print("-"*30)
        else:
            print("Loading previous results")
            print(' → "{}"'.format(self.result_file))
            records = np.load(self.result_file)
            print("-"*30)

        return records

    @classmethod
    def session_init(cls, args):
        """Initialize the random seed of a process and run a session."""
        experiment, session, seed = args
        np.random.seed(seed)
        return session(experiment)

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
