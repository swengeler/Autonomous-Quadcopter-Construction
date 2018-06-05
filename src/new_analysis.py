import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pprint import pprint
from agents.agent import Task
from new_experiments import AGENT_TYPES, VALUES, extra_parameters


LOAD_DIRECTORY = "/home/simon/PycharmProjects/LowFidelitySimulation/res/results/"


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


def show_scaling_performance(data, parameters):
    # essentially just show performance for all agent counts given the parameters
    agent_counts = []
    step_counts = []
    completion_counts = []
    for d in data:
        if d["parameters"]["agent_type"] == parameters["agent_type"]:
            if all([d["parameters"][k] == v for k, v in parameters.items()]):
                agent_count = d["parameters"]["agent_count"]
                step_count = d["step_count"]
                completion_count = d["structure_completion"]
                if d["got_stuck"]:
                    print("Structure was not finished with {} agents because they got stuck.".format(agent_count))
                elif d["highest_layer"] not in d["layer_completion"].keys():
                    print("Structure was not finished with {} agents for some other reason.".format(agent_count))
                agent_counts.append(agent_count)
                step_counts.append(step_count)
                completion_counts.append(completion_count)

    order = sorted(range(len(agent_counts)), key=lambda i: agent_counts[i])
    agent_counts = [agent_counts[i] for i in order]
    step_counts = [step_counts[i] for i in order]
    completion_counts = [completion_counts[i] for i in order]

    f, ax = plt.subplots()
    ax.plot(agent_counts, step_counts, label="Total count")
    # ax.plot(agent_counts, completion_counts, label="Completion count")
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def show_all_agent_performance(data, parameter_list):
    # essentially just show performance for all agent counts given the parameters
    agent_counts = {p["agent_type"]: [] for p in parameter_list}
    step_counts = {p["agent_type"]: [] for p in parameter_list}
    not_completed = {p["agent_type"]: [] for p in parameter_list}
    for d in data:
        for p in parameter_list:
            if d["parameters"]["agent_type"] == p["agent_type"]:
                if all([d["parameters"][k] == v for k, v in p.items()]):
                    agent_count = d["parameters"]["agent_count"]
                    step_count = d["step_count"]
                    nc = False
                    if d["got_stuck"]:
                        print("Structure was not finished with {} {}s because they got stuck."
                              .format(agent_count, p["agent_type"]))
                        nc = True
                    elif d["highest_layer"] not in d["layer_completion"].keys():
                        print("Structure was not finished with {} {}s for some other reason."
                              .format(agent_count, p["agent_type"]))
                        nc = True
                    agent_counts[p["agent_type"]].append(agent_count)
                    step_counts[p["agent_type"]].append(step_count)
                    not_completed[p["agent_type"]].append(nc)

    f, ax = plt.subplots()
    for k in agent_counts:
        order = sorted(range(len(agent_counts[k])), key=lambda i: agent_counts[k][i])
        agent_counts[k] = [agent_counts[k][i] for i in order]
        step_counts[k] = [step_counts[k][i] for i in order]
        not_completed[k] = [not_completed[k][i] for i in order]
        base = None
        for nc_idx, nc in enumerate(not_completed[k]):
            if nc:
                if base is None:
                    temp, = ax.plot(agent_counts[k][nc_idx], step_counts[k][nc_idx], "x", markersize=6
                                    , markerfacecolor=base)
                    base = temp.get_color()
                else:
                    ax.plot(agent_counts[k][nc_idx], step_counts[k][nc_idx], "x", markersize=12, markerfacecolor=base)
        ax.plot(agent_counts[k], step_counts[k], label=k, color=base)
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def compare_parameters(data, parameter_list, descriptions=None):
    p_count = 0
    f, ax = plt.subplots()
    for a_type, p in parameter_list:
        agent_counts = []
        step_counts = []
        for d in data:
            if d["parameters"]["agent_type"] == a_type:
                print("Hello")
                if all([d["parameters"][k] == p[k] for k in p]):
                    agent_counts.append(d["parameters"]["agent_count"])
                    step_counts.append(d["step_count"])
        order = sorted(range(len(agent_counts)), key=lambda i: agent_counts[i])
        agent_counts = [agent_counts[i] for i in order]
        step_counts = [step_counts[i] for i in order]
        label = descriptions[p_count] if descriptions is not None else "Parameter set {}".format(p_count)
        ax.plot(agent_counts, step_counts, label=label)
        p_count += 1
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def main():
    map_name = "spacing_4"
    data = load_single_map_data(map_name)
    print(len(data))

    parameter_list = []
    for agent_type in AGENT_TYPES:
        parameters = {"agent_type": agent_type}
        parameters.update({k: VALUES[k][0] for k in extra_parameters(agent_type)})
        parameter_list.append(parameters)
    # show_all_agent_performance(data, parameter_list)
    # show_scaling_performance(data, parameters)
    a1 = a2 = "LocalPerimeterFollowingAgent"
    p1 = {
        "waiting_on_perimeter_enabled": False,
        "avoiding_crowded_stashes_enabled": True,
        "transport_avoid_others_enabled": True,
        "seed_if_possible_enabled": True,
        "seeding_strategy": "distance_center"
    }
    p2 = {
        "waiting_on_perimeter_enabled": False,
        "avoiding_crowded_stashes_enabled": True,
        "transport_avoid_others_enabled": True,
        "seed_if_possible_enabled": True,
        "seeding_strategy": "agent_count"
    }
    params = [(a1, p1), (a2, p2)]
    # descriptions = ["Seeding when possible", "Completing first"]
    descriptions = ["Seed closest to center", "Seed with fewest agents"]
    compare_parameters(data, params, descriptions=descriptions)


if __name__ == "__main__":
    main()