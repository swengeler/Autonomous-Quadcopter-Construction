import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as scs
import json
import os
from pprint import pprint
from matplotlib.colors import ListedColormap
from agents.agent import Task
from experiments import AGENT_TYPES, VALUES, extra_parameters


LOAD_DIRECTORY = "/home/simon/PycharmProjects/LowFidelitySimulation/res/new_results/"


PLATE_TO_BLOCK = {
    "plate_4x4": None,
    "plate_8x8": "block_4x4x4",
    "plate_12x12": "block_5x5x6",
    "plate_16x16": "block_6x6x7",
    "plate_20x20": "block_7x7x8",
    "plate_24x24": "block_8x8x9",
    "plate_28x28": "block_9x9x10",
    "plate_32x32": "block_10x10x10",
}


def mean_and_conf_size(data, confidence_level=0.95):
    mean = np.mean(data)
    std = np.std(data)
    t_bounds = scs.t.interval(confidence_level, len(data) - 1)
    size = t_bounds[1] * std / np.sqrt(len(data))
    return mean, size


def load_single_map_data(map_name, experiment_name=None):
    data = []
    if experiment_name is not None:
        directory_path = LOAD_DIRECTORY + map_name + "/" + experiment_name + "/"
        for file_name in os.listdir(directory_path):
            try:
                with open(directory_path + file_name) as f:
                    d = json.load(f)
                    if d:
                        data.append(d)
                    else:
                        print("File {} is empty.".format(file_name))
            except ValueError as e:
                print("Loading of file {} failed. Error message: '{}'".format(file_name, e))
    else:
        directory_path = LOAD_DIRECTORY + map_name + "/"
        for directory_name in os.listdir(directory_path):
            if os.path.isdir(directory_path + directory_name):
                for file_name in os.listdir(directory_path + directory_name):
                    try:
                        with open(directory_path + directory_name + "/" + file_name) as f:
                            d = json.load(f)
                            if d:
                                data.append(d)
                            else:
                                print("File {} is empty.".format(file_name))
                    except ValueError as e:
                        print("Loading of file {} failed. Error message: '{}'".format(file_name, e))
    print("Loaded {} files for map {}.\n".format(len(data), map_name))
    return data


def load_matching_map_data(map_generic):
    all_data = {}
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith(map_generic) and not "padded" in file_name:
            data = load_single_map_data(file_name, "defaults")
            all_data[file_name] = data
    return all_data


def get_task_stats(data, task, agent_type, agent_count, complete_only=False):
    # need to get the mean, std and count for each run
    # -> count should actually just be the number of agents
    # -> it might not actually be...
    # then need to use the following procedure to turn it into a total std:
    # http://www.burtonsys.com/climate/composite_standard_deviations.html

    # then need to use the degrees of freedom (total number of observations (?)) to compute confidence interval

    step_count = {"mean": [], "std": [], "min": [], "max": []}
    collision_avoidance_count = {"mean": [], "std": [], "min": [], "max": []}
    distance_travelled = {"mean": [], "std": [], "min": [], "max": []}
    total_number = 0
    for d in data:
        if d["parameters"]["agent_type"] == agent_type and d["parameters"]["agent_count"] == agent_count \
                and (not complete_only or d["finished_successfully"]):
            task_stats = d["task_stats"][task.name]

            step_count["mean"].append(task_stats["step_count"]["mean"])
            step_count["std"].append(task_stats["step_count"]["std"])
            step_count["min"].append(task_stats["step_count"]["min"])
            step_count["max"].append(task_stats["step_count"]["max"])

            collision_avoidance_count["mean"].append(task_stats["collision_avoidance_count"]["mean"])
            collision_avoidance_count["std"].append(task_stats["collision_avoidance_count"]["std"])
            collision_avoidance_count["min"].append(task_stats["collision_avoidance_count"]["min"])
            collision_avoidance_count["max"].append(task_stats["collision_avoidance_count"]["max"])

            distance_travelled["mean"].append(task_stats["distance_travelled"]["mean"])
            distance_travelled["std"].append(task_stats["distance_travelled"]["std"])
            distance_travelled["min"].append(task_stats["distance_travelled"]["min"])
            distance_travelled["max"].append(task_stats["distance_travelled"]["max"])

            total_number += 1

    # compute the overall stats
    step_count_total = {
        "mean": float(np.mean(step_count["mean"])),
        "min": min(step_count["min"]),
        "max": max(step_count["max"])
    }
    collision_avoidance_count_total = {
        "mean": float(np.mean(collision_avoidance_count["mean"])),
        "min": min(collision_avoidance_count["min"]),
        "max": max(collision_avoidance_count["max"])
    }
    distance_travelled_total = {
        "mean": float(np.mean(distance_travelled["mean"])),
        "min": min(distance_travelled["min"]),
        "max": max(distance_travelled["max"])
    }

    # compute the total standard deviations
    step_count_ESS = sum([std ** 2 * (agent_count - 1) for std in step_count["std"]])
    step_count_TGSS = sum([(mean - step_count_total["mean"]) ** 2 * agent_count for mean in step_count["mean"]])
    step_count_total["std"] = float(np.sqrt((step_count_ESS + step_count_TGSS) / total_number))

    collision_avoidance_count_ESS = sum([std ** 2 * (agent_count - 1) for std in collision_avoidance_count["std"]])
    collision_avoidance_count_TGSS = sum([(mean - collision_avoidance_count_total["mean"]) ** 2 * agent_count
                                          for mean in collision_avoidance_count["mean"]])
    collision_avoidance_count_total["std"] = float(
        np.sqrt((collision_avoidance_count_ESS + collision_avoidance_count_TGSS) / total_number))

    distance_travelled_ESS = sum([std ** 2 * (agent_count - 1) for std in distance_travelled["std"]])
    distance_travelled_TGSS = sum([(mean - distance_travelled_total["mean"]) ** 2 * agent_count
                                   for mean in distance_travelled["mean"]])
    distance_travelled_total["std"] = float(np.sqrt((distance_travelled_ESS + distance_travelled_TGSS) / total_number))

    return {"step_count": step_count_total,
            "collision_avoidance_count": collision_avoidance_count_total,
            "distance_travelled": distance_travelled_total}


def show_task_proportions(map_name, agent_type, metric="step_count", statistic="mean", experiment_name="defaults"):
    # f, ax = plt.subplots()
    # sns.set_palette()
    counts = [1, 2, 4, 8, 12, 16]
    # for each number of agents, compute the average over all runs and show as stacked bar chart
    # only consider tasks with a count larger than 0
    data = load_single_map_data(map_name, experiment_name)
    task_names = [t.name for t in Task]
    task_means_all = []
    for agent_count in counts:
        task_means = []
        for task in Task:
            task_stats = get_task_stats(data, task, agent_type, agent_count, True)
            task_means.append(task_stats[metric][statistic])
        task_means_all.append(task_means)
        print(task_means)
    task_means_all = np.array(task_means_all)

    # bottom = np.zeros_like(counts, dtype="float64")
    # for t_idx, t in enumerate(task_names):
    #     count_task_means = task_means_all[:, t_idx]
    #     if np.count_nonzero(count_task_means != 0) == 0:
    #         continue
    #     ax.bar(counts, count_task_means, bottom=bottom, label=t)
    #     bottom += count_task_means
    #
    # ax.legend()
    # plt.show()

    print(task_means_all)

    df = pd.DataFrame(task_means_all, index=counts, columns=task_names)
    df = df.loc[:, (df != 0).any(axis=0)]
    print(df)
    # color_map = ListedColormap(sns.color_palette("Blues_r", len(df.columns)).as_hex())
    color_map = ListedColormap(sns.color_palette("coolwarm", len(df.columns)).as_hex())
    ax = df.plot(kind="bar", stacked=True, colormap=color_map)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


def show_plate_block_comparison(map_name):
    # f, (ax_plate, ax_block, ax_block_padded) = plt.subplots(1, 3)
    f, ax_plate = plt.subplots()

    agent_counts = {at: {} for at in AGENT_TYPES}
    step_counts = {at: {} for at in AGENT_TYPES}
    not_completed = {at: {} for at in AGENT_TYPES}

    data = load_single_map_data(map_name, "defaults")
    for d in data:
        for at in AGENT_TYPES:
            if d["parameters"]["agent_type"] == at:
                if d["finished_successfully"]:
                    agent_count = d["parameters"]["agent_count"]
                    step_count = d["step_count"]
                    nc = False
                    if not d["finished_successfully"]:
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

    for k in agent_counts:
        # if k == "GlobalShortestPathAgent":
        all_x = []
        all_y = []
        all_conf = []
        for count in agent_counts[k]:
            mean, conf_size = mean_and_conf_size(step_counts[k][count])
            all_x.append(count)
            all_y.append(mean)
            all_conf.append(conf_size)
        order = sorted(range(len(all_x)), key=lambda i: all_x[i])
        all_x = [all_x[i] for i in order]
        all_y = [all_y[i] for i in order]
        all_conf = [all_conf[i] for i in order]
        ax_plate.errorbar(all_x, all_y, yerr=all_conf, label=k)

    ax_plate.set_ylim(ymin=0)
    ax_plate.legend()
    plt.show()


def show_plate_collision_proportion(agent_type="LocalPerimeterFollowingAgent"):
    data = load_matching_map_data("plate")
    for map_name in data:

        pass


def show_all_plate_block_comparisons():
    pass


def show_perimeter_comparison(agent_type):
    counts = [1, 2, 4, 8, 12, 16]
    map_names = []
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith("perim"):
            if file_name not in map_names:
                map_names.append(file_name.replace(".npy", ""))

    step_counts_total = []
    for map_name in map_names:
        data = load_single_map_data(map_name, "defaults")
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)
    print(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names)
    print(df)
    color_map = ListedColormap(sns.color_palette("coolwarm", len(df.columns)).as_hex())
    ax = df.plot(kind="line")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


def show_upward_scaling(agent_type):
    map_names = ["plate_8x8", "block_8x8x2", "block_8x8x3", "block_8x8x4", "block_8x8x6", "block_8x8x8"]
    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    for m in map_names:
        data = load_single_map_data(m, "defaults")
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                # compute average step count etc.
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names)
    ax = df.plot(kind="line")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


def show_multiple_component_scaling(agent_type):
    map_names = []
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith("component"):
            if file_name not in map_names:
                map_names.append(file_name.replace(".npy", ""))

    print(map_names)

    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    for m in map_names:
        data = load_single_map_data(m, "defaults")
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                # compute average step count etc.
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names)
    ax = df.plot(kind="line")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


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
            mean, conf_size = mean_and_conf_size(step_counts[k][count])
            all_x.append(count)
            all_y.append(mean)
            all_conf.append(conf_size)
        order = sorted(range(len(all_x)), key=lambda i: all_x[i])
        all_x = [all_x[i] for i in order]
        all_y = [all_y[i] for i in order]
        all_conf = [all_conf[i] for i in order]
        ax.errorbar(all_x, all_y, yerr=all_conf, label=k)

    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def show_attachment_site_ordering_differences(agent_type, map_name):
    if agent_type.startswith("Local"):
        parameter_name = "attachment_site_order"
    else:
        parameter_name = "attachment_site_ordering"
    options = VALUES[parameter_name]

    data = load_single_map_data(map_name)
    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    for o in options:
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"] \
                    and parameter_name in d["parameters"] and d["parameters"][parameter_name] == o:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=options)
    ax = df.plot(kind="line")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


def show_component_ordering_differences(agent_type, map_name):
    if agent_type.startswith("Local"):
        parameter_name = "seeding_strategy"
    else:
        parameter_name = "component_ordering"
    options = VALUES[parameter_name]

    data = load_single_map_data(map_name)
    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    for o in options:
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"] \
                    and parameter_name in d["parameters"] and d["parameters"][parameter_name] == o:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=options)
    ax = df.plot(kind="line")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


def main():
    # TODO: need some easy way to get the yerr
    map_name = "components_6x6x1"
    # data = load_single_map_data(map_name)
    # show_all_agent_performance(data)
    # show_scaling_performance(data, parameters)
    # show_plate_block_comparison("plate_32x32")

    # show_task_proportions("block_10x10x10", "GlobalShortestPathAgent", metric="collision_avoidance_count")
    # data = load_matching_map_data("plate")
    # show_perimeter_comparison("LocalPerimeterFollowingAgent")
    # show_upward_scaling("GlobalShortestPathAgent")
    # show_multiple_component_scaling("GlobalShortestPathAgent")
    # show_perimeter_comparison("LocalPerimeterFollowingAgent")
    # show_attachment_site_ordering_differences("GlobalShortestPathAgent", "cross_24")
    # -> probably doesn't show any real differences because even when agent count is first ordering criterion
    #    the distance thing is still second
    show_component_ordering_differences("LocalPerimeterFollowingAgent", "seed_comp_4x4_2_1")


if __name__ == "__main__":
    main()
