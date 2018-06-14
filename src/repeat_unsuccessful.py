import json
import os
import sys
from pprint import pprint

import numpy as np

from analysis import LOAD_DIRECTORY as LOAD_DIRECTORY_ALT
from experiments import run_experiment, SAVE_DIRECTORY_NAME_ALT, SAVE_DIRECTORY_NAME

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


def load_wait_on_perimeter_files(root_directory, map_directory):
    data = []
    total_count = 0
    for directory_name, _, file_list in os.walk(root_directory):
        if "wait_on_perimeter" in directory_name:
            for file_name in file_list:
                try:
                    with open(directory_name + "/" + file_name) as f:
                        d = json.load(f)
                        if d:
                            test = np.load(map_directory + d["parameters"]["target_map"] + ".npy").astype("int64")
                            if test.shape[0] > 1:
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


def load_ordering_files(root_directory):
    data = []
    total_count = 0
    for directory_name, _, file_list in os.walk(root_directory):
        if "sp_" in directory_name or "ordering_" in directory_name or "seed_" in directory_name:
            for file_name in file_list:
                try:
                    with open(directory_name + "/" + file_name) as f:
                        d = json.load(f)
                        if d:
                            if d["finished_successfully"]:
                                if "order_only_one_metric" not in d["parameters"]:
                                    d["parameters"]["order_only_one_metric"] = True
                                data.append(d)
                                if len(data) % 100 == 0:
                                    print("Appended {} files to list.".format(len(data)))
                        else:
                            print("File {} is empty.".format(file_name))
                        total_count += 1
                except ValueError as e:
                    print("Loading of file {} failed. Error message: '{}'".format(file_name, e))
    print("Loaded {} unfinished out of {} total files.".format(len(data), total_count))
    return data


def load_sp_files(root_directory):
    data = []
    total_count = 0
    for directory_name, _, file_list in os.walk(root_directory):
        for file_name in file_list:
            try:
                with open(directory_name + "/" + file_name) as f:
                    d = json.load(f)
                    if d:
                        if d["finished_successfully"] and d["parameters"]["agent_type"] == "LocalShortestPathAgent":
                            if "sp_" in directory_name or "ordering_" in directory_name or "seed_" in directory_name:
                                if "order_only_one_metric" not in d["parameters"]:
                                    d["parameters"]["order_only_one_metric"] = True
                            data.append(d)
                            if len(data) % 100 == 0:
                                print("Appended {} files to list.".format(len(data)))
                    else:
                        print("File {} is empty.".format(file_name))
                    total_count += 1
            except ValueError as e:
                print("Loading of file {} failed. Error message: '{}'".format(file_name, e))
    print("Loaded {} unfinished out of {} total files.".format(len(data), total_count))
    return data


def load_specified_files(file_list, add_ordering_thing=False):
    data = []
    total_count = 0
    for file_name in file_list:
        try:
            with open(file_name) as f:
                d = json.load(f)
                if d:
                    if add_ordering_thing:
                        d["parameters"]["order_only_one_metric"] = True
                    data.append(d)
                    if len(data) % 10 == 0:
                        print("Appended {} files to list.".format(len(data)))
                else:
                    print("File {} is empty.".format(file_name))
                total_count += 1
        except ValueError as e:
            print("Loading of file {} failed. Error message: '{}'".format(file_name, e))
    return data


def run_single_unfinished(single_unfinished, save_directory):
    if single_unfinished is None:
        return None

    p = single_unfinished["parameters"]
    if "order_only_one_metric" not in p:
        p["order_only_one_metric"] = True
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
                directory_name = save_directory + p["target_map"] + "/" + p["experiment_name"]
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


def main(func, number_parallel, number_self, server=True):
    root_directory = LOAD_DIRECTORY if server else LOAD_DIRECTORY_ALT
    save_directory = SAVE_DIRECTORY_NAME if server else SAVE_DIRECTORY_NAME_ALT

    def split_into_chunks(l, n):
        return [l[i::n] for i in range(n)]

    file_name = "/home/simon/unfinished_experiments.txt"
    add_ordering_thing = False
    if func == 1:
        file_name = "/home/simon/unfinished_sp_experiments.txt"
    elif func == 2:
        file_name = "/home/simon/unfinished_ordering_experiments.txt"
    elif func == 3:
        file_name = "/home/simon/final_ordering_experiments.txt"
        add_ordering_thing = True

    finished_file_list = []
    try:
        with open("/home/simon/finished_experiments.txt") as f:
            finished_file_list = f.readlines()
        finished_file_list = [x.strip() for x in finished_file_list]

        with open("/home/simon/latter_half.txt") as f:
            latter_half_list = f.readlines()
        finished_file_list.extend([x.strip() for x in latter_half_list])
    except FileNotFoundError:
        pass

    with open(file_name) as f:
        file_list = f.readlines()
    file_list = [x.strip() for x in file_list]
    file_list = [x for x in file_list if x not in finished_file_list]

    file_list = split_into_chunks(file_list, number_parallel)[number_self]
    unfinished = load_specified_files(file_list, add_ordering_thing)
    run_counter = 0
    while len(unfinished) != 0:
        # pool = ThreadPool(8)
        print("STARTING RUN {} WITH {} EXPERIMENTS TO GO.".format(run_counter, len(unfinished)))
        unfinished = run_all_unfinished(unfinished, save_directory)
        print("FINISHING RUN {} WITH {} EXPERIMENTS.\n".format(run_counter, len(unfinished)))
        run_counter += 1


def save_file_names(server=False):
    root_directory = LOAD_DIRECTORY if server else LOAD_DIRECTORY_ALT
    save_directory = SAVE_DIRECTORY_NAME if server else SAVE_DIRECTORY_NAME_ALT

    unfinished = load_unfinished_files(root_directory)
    print(len(unfinished))
    for f in unfinished:
        directory_name = save_directory + f["parameters"]["target_map"] + "/" + f["parameters"]["experiment_name"]
        file_name = "{}_{}_{}.json".format(
            f["parameters"]["agent_type"], f["parameters"]["agent_count"], f["parameters"]["run"])
        absolute_file_name = directory_name + "/" + file_name
        with open("/home/simon/unfinished_experiments.txt", "a") as file:
            file.write(absolute_file_name + "\n")


def save_ordering_file_names(server=False):
    root_directory = LOAD_DIRECTORY if server else LOAD_DIRECTORY_ALT
    save_directory = SAVE_DIRECTORY_NAME if server else SAVE_DIRECTORY_NAME_ALT

    unfinished = load_ordering_files(root_directory)
    for f in unfinished:
        directory_name = save_directory + f["parameters"]["target_map"] + "/" + f["parameters"]["experiment_name"]
        file_name = "{}_{}_{}.json".format(
            f["parameters"]["agent_type"], f["parameters"]["agent_count"], f["parameters"]["run"])
        absolute_file_name = directory_name + "/" + file_name
        with open("/home/simon/final_ordering_experiments.txt", "a") as file:
            file.write(absolute_file_name + "\n")


def save_sp_file_names(server=False):
    root_directory = LOAD_DIRECTORY if server else LOAD_DIRECTORY_ALT
    save_directory = SAVE_DIRECTORY_NAME if server else SAVE_DIRECTORY_NAME_ALT

    unfinished = load_sp_files(root_directory)
    for f in unfinished:
        directory_name = save_directory + f["parameters"]["target_map"] + "/" + f["parameters"]["experiment_name"]
        file_name = "{}_{}_{}.json".format(
            f["parameters"]["agent_type"], f["parameters"]["agent_count"], f["parameters"]["run"])
        absolute_file_name = directory_name + "/" + file_name
        with open("/home/simon/unfinished_sp_experiments.txt", "a") as file:
            file.write(absolute_file_name + "\n")


def show_wait_on_perimeter(server=False):
    root_directory = LOAD_DIRECTORY if server else LOAD_DIRECTORY_ALT
    map_directory = "/home/simon/PycharmProjects/LowFidelitySimulation/res/experiment_maps/"

    unfinished = load_wait_on_perimeter_files(root_directory, map_directory)
    print(len(unfinished))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_wait_on_perimeter()
    elif len(sys.argv) > 4:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), False)
    elif len(sys.argv) == 2:
        idx = int(sys.argv[1])
        if idx == 0:
            save_file_names(True)
        elif idx == 1:
            save_sp_file_names(True)
        elif idx == 2:
            save_ordering_file_names(True)
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))


