import pandas as pd
import numpy as np
import time
from os import path, mkdir


class Statistics:
    def __init__(self, evaluation_columns):
        columns = ["episode_time"]
        columns.extend(evaluation_columns)
        self.columns = dict([(key, i) for i, key in enumerate(columns)])
        self.data = []
        self.episode_counter = 0
        self.episode_start = -1

        # Tmp storage for average values
        self.iteration_values = {}

    def _compute_write_mean(self):
        for key in self.iteration_values.keys():
            mean = np.mean(self.iteration_values[key])
            self.data[-1][self.columns[key]] = mean
        self.iteration_values.clear()


    def log(self, key, value):
        self.data[-1][self.columns[key]] = value

    def log_config(self, config):
        for key, val in config.items():
            if key in self.columns.keys():
                self.data[-1][self.columns[key]] = val

    def log_iteration(self, key, value):
        if value is not None:
            assert(isinstance(value, float))
            if key not in self.iteration_values.keys():
                self.iteration_values[key] = []
            self.iteration_values[key].append(value)

    def start_episode(self):
        self.episode_counter += 1

        self.finalize()

        self.episode_start = time.time()

        # create new analysis row
        self.data.append([np.nan]*len(self.columns))

    def finalize(self):
        if self.episode_counter > 1:
            self.data[-1][self.columns["episode_time"]] = time.time()-self.episode_start
            # compute means of previous episodes
            self._compute_write_mean()

    def save(self, filename):
        # Test if statistics dir exists
        if not path.isdir("statistics"):
            mkdir('statistics')

        pd.DataFrame(self.data, columns=self.columns.keys()).to_csv("statistics/" + filename)

    def __str__(self):
        return str(pd.DataFrame(self.data, columns=self.columns.keys()))
