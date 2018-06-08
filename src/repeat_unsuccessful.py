import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as scs
import sys
import json
import os
import itertools
from multiprocessing.dummy import Pool as ThreadPool
from pprint import pprint
from matplotlib.colors import ListedColormap
from agents.agent import Task
from experiments import AGENT_TYPES, VALUES, extra_parameters, run_experiment, SAVE_DIRECTORY_NAME_ALT, SAVE_DIRECTORY_NAME
from analysis import LOAD_DIRECTORY as LOAD_DIRECTORY_ALT


LOAD_DIRECTORY = "/home/simon/new_results/"


def load_unfinished_files(root_directory):
    data = []
    total_count = 0
    for directory_name, _, file_list in os.walk(root_directory):
        for file_name in file_list:
            try:
                with open(directory_name + "/" + file_name) as f:
                    d = json.load(f)
                    if d:
                        if not d["finished_successfully"] or d["got_stuck"]:
                            data.append(d)
                            if len(data) % 10 == 0:
                                print("Appended {} files to list.".format(len(data)))
                    else:
                        print("File {} is empty.".format(file_name))
                    total_count += 1
            except ValueError as e:
                print("Loading of file {} failed. Error message: '{}'".format(file_name, e))
    print("Loaded {} unfinished out of {} total files.".format(len(data), total_count))
    return data


def run_single_unfinished(single_unfinished, save_directory):
    if single_unfinished is None:
        return None

    p = single_unfinished["parameters"]
    results = run_experiment(p)
    if results["finished_successfully"]:
        try:
            directory_name = save_directory + p["target_map"] + "/" + p["experiment_name"]
            file_name = "{}_{}_{}.json".format(p["agent_type"], p["agent_count"], p["run"])
            absolute_file_name = directory_name + "/" + file_name
            with open(absolute_file_name, "w") as file:
                json.dump(results, file)
            print("Successfully saved results for run on map {} with {} agents.".format(p["target_map"], p["agent_count"]))
            with open("/home/simon/finished_experiments.txt", "a") as file:
                file.write(absolute_file_name + "\n")
            return None
        except KeyboardInterrupt:
            print("Cancelled run with the following parameters:")
            pprint(p)
        except Exception as e:
            print("Error in run with the following parameters:")
            pprint(p)
            raise e
    return single_unfinished


def run_all_unfinished(unfinished, save_directory):
    parameter_list = []
    for u in unfinished:
        parameter_list.append(u["parameters"])

    runs_completed = 0
    new_unfinished = []
    for p in parameter_list:
        results = run_experiment(p)
        if results["finished_successfully"]:
            try:
                directory_name = save_directory + p["map_name"] + "/" + p["experiment_name"]
                file_name = "{}_{}_{}.json".format(p["agent_type"], p["agent_count"], p["run"])
                absolute_file_name = directory_name + "/" + file_name
                with open(absolute_file_name, "w") as file:
                    json.dump(results, file)
                with open("/home/simon/finished_experiments.txt", "a") as file:
                    file.write(absolute_file_name + "\n")
                print("Successfully saved results for run {}/{} with {} agents.".format(runs_completed,
                                                                                        len(parameter_list),
                                                                                        p["agent_count"]))
            except KeyboardInterrupt:
                print("Cancelled run with the following parameters:")
                pprint(p)
                break
            except Exception as e:
                print("Error in run with the following parameters:")
                pprint(p)
                raise e
        else:
            new_unfinished.append(unfinished[runs_completed])
        runs_completed += 1

    return new_unfinished


def main(number_parallel, number_self, server=True):
    root_directory = LOAD_DIRECTORY if server else LOAD_DIRECTORY_ALT
    save_directory = SAVE_DIRECTORY_NAME if server else SAVE_DIRECTORY_NAME_ALT

    def split_into_chunks(l, n):
        return [l[i::n] for i in range(n)]

    unfinished = load_unfinished_files(root_directory)
    unfinished = split_into_chunks(unfinished, number_parallel)[number_self]
    run_counter = 0
    while len(unfinished) != 0:
        # pool = ThreadPool(8)
        print("STARTING RUN {} WITH {} EXPERIMENTS TO GO.".format(run_counter, len(unfinished)))
        unfinished = run_all_unfinished(unfinished, save_directory)
        print("FINISHING RUN {} WITH {} EXPERIMENTS.\n".format(run_counter, len(unfinished)))
        run_counter += 1


def save_file_names():
    root_directory = LOAD_DIRECTORY_ALT
    save_directory = SAVE_DIRECTORY_NAME_ALT

    unfinished = load_unfinished_files(root_directory)
    for f in unfinished:
        directory_name = save_directory + f["parameters"]["target_map"] + "/" + f["parameters"]["experiment_name"]
        file_name = "{}_{}_{}.json".format(
            f["parameters"]["agent_type"], f["parameters"]["agent_count"], f["parameters"]["run"])
        absolute_file_name = directory_name + "/" + file_name
        with open("/home/simon/unfinished_experiments.txt", "a") as file:
            file.write(absolute_file_name + "\n")


if __name__ == "__main__":
    # save_file_names()
    if len(sys.argv) > 3:
        main(int(sys.argv[1]), int(sys.argv[2]), False)
    else:
        main(int(sys.argv[1]), int(sys.argv[2]))


