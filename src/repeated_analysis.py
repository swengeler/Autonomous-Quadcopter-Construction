import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import json
import os
from pprint import pprint
from agents.agent import Task
from new_experiments import AGENT_TYPES, VALUES, extra_parameters


LOAD_DIRECTORY = "/home/simon/PycharmProjects/LowFidelitySimulation/res/repeated_results/"


def mean_and_conf_size(data, confidence_level=0.95):
    avg = np.mean(data)
    std = np.std(data)
    t_bounds = scs.t.interval(confidence_level, len(data) - 1)
    size = t_bounds[1] * std / np.sqrt(len(data))
    return avg, size


def load_single_map_data(map_name):
    data = []
    directory_path = LOAD_DIRECTORY + map_name + "/"
    for filename in os.listdir(directory_path):
        try:
            with open(directory_path + filename) as f:
                d = json.load(f)
                if d:
                    data.append(d)
                else:
                    print("File {} is empty.".format(filename))
        except ValueError as e:
            print("Loading of file {} failed. Error message: '{}'".format(filename, e))
    print("Loaded {} files for map {}.\n".format(len(data), map_name))
    return data


def show_all_agent_performance(data):
    # essentially just show performance for all agent counts given the parameters
    agent_counts = {at: {} for at in AGENT_TYPES}
    step_counts = {at: {} for at in AGENT_TYPES}
    not_completed = {at: {} for at in AGENT_TYPES}
    for d in data:
        for at in AGENT_TYPES:
            if d["parameters"]["agent_type"] == at:
                agent_count = d["parameters"]["agent_count"]
                step_count = d["step_count"]
                nc = False
                if d["got_stuck"]:
                    print("Structure was not finished with {} {}s because they got stuck.".format(agent_count, at))
                    nc = True
                elif d["highest_layer"] not in d["layer_completion"].keys():
                    print("Structure was not finished with {} {}s for some other reason.".format(agent_count, at))
                    nc = True
                if agent_count not in agent_counts[at]:
                    agent_counts[at][agent_count] = []
                    step_counts[at][agent_count] = []
                    not_completed[at][agent_count] = []
                agent_counts[at][agent_count].append(agent_count)
                step_counts[at][agent_count].append(step_count)
                not_completed[at][agent_count].append(nc)

    f, ax = plt.subplots()
    for k in agent_counts:
        # if k == "GlobalShortestPathAgent":
        all_x = []
        all_y = []
        all_conf = []
        for count in agent_counts[k]:
            all_x.append(count)
            avg = np.mean(step_counts[k][count])
            all_y.append(avg)
            std = np.std(step_counts[k][count])
            t_bounds = scs.t.interval(0.95, len(step_counts[k][count]) - 1)
            size = t_bounds[1] * std / np.sqrt(len(step_counts[k][count]))
            all_conf.append(size)
        order = sorted(range(len(all_x)), key=lambda i: all_x[i])
        all_x = [all_x[i] for i in order]
        all_y = [all_y[i] for i in order]
        all_conf = [all_conf[i] for i in order]
        # ax.plot(all_x, all_y, "x", label="GlobalShortestPathAgent")
        ax.errorbar(all_x, all_y, yerr=all_conf, label=k)
        # order = sorted(range(len(agent_counts[k])), key=lambda i: agent_counts[k][i])
        # agent_counts[k] = [agent_counts[k][i] for i in order]
        # step_counts[k] = [step_counts[k][i] for i in order]
        # not_completed[k] = [not_completed[k][i] for i in order]
        # base = None
        # for nc_idx, nc in enumerate(not_completed[k]):
        #     if nc:
        #         if base is None:
        #             temp, = ax.plot(agent_counts[k][nc_idx], step_counts[k][nc_idx], "x", markersize=6
        #                             , markerfacecolor=base)
        #             base = temp.get_color()
        #         else:
        #             ax.plot(agent_counts[k][nc_idx], step_counts[k][nc_idx], "x", markersize=12, markerfacecolor=base)
        # ax.plot(agent_counts[k], step_counts[k], label=k, color=base)
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def main():
    map_name = "components_6x6x1"
    data = load_single_map_data(map_name)
    show_all_agent_performance(data)
    # show_scaling_performance(data, parameters)


if __name__ == "__main__":
    main()