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


def run_all_unfinished(unfinished):
    parameter_list = []
    for u in unfinished:
        parameter_list.append(u["parameters"])

    runs_completed = 0
    start_runs_at = 0
    for p in parameter_list[start_runs_at:]:
        results = run_experiment(p)
        try:
            directory_name = SAVE_DIRECTORY_NAME_ALT + p["map_name"] + "/" + p["experiment_name"]
            file_name = "{}_{}_{}.json".format(p["agent_type"], p["agent_count"], p["run"])
            absolute_file_name = directory_name + "/" + file_name
            with open(absolute_file_name, "w") as file:
                json.dump(results, file)
            runs_completed += 1
            print("Successfully saved results for run {} with {} agents.".format(p["run"], p["agent_count"]))
            print("RUNS COMPLETED: {}/{} (out of total: {}/{})\n\n".format(
                runs_completed, len(parameter_list) - start_runs_at, start_runs_at + runs_completed,
                len(parameter_list)))
        except KeyboardInterrupt:
            print("Cancelled run with the following parameters:")
            pprint(p)
            break
        except Exception as e:
            print("Error in run with the following parameters:")
            pprint(p)
            raise e


def main(server=True):
    root_directory = LOAD_DIRECTORY if server else LOAD_DIRECTORY_ALT
    save_directory = SAVE_DIRECTORY_NAME if server else SAVE_DIRECTORY_NAME_ALT

    unfinished = load_unfinished_files(root_directory)
    run_counter = 0
    while len(unfinished) != 0:
        pool = ThreadPool(8)
        print("STARTING RUN {} WITH {} EXPERIMENTS TO GO.".format(run_counter, len(unfinished)))
        new_unfinished = pool.starmap(run_single_unfinished, zip(unfinished, itertools.repeat(save_directory)))
        print("FINISHING RUN {} WITH {} EXPERIMENTS.\n".format(run_counter, len(unfinished)))
        new_unfinished = [u for u in new_unfinished if u is not None]
        finished = [u for u in unfinished if u not in new_unfinished]
        unfinished = new_unfinished

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
    if len(sys.argv) > 1:
        main(False)
    else:
        main()


