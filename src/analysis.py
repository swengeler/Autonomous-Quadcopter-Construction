import json
import os
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import scipy.stats as scs
import seaborn as sns
from cycler import cycler
from matplotlib.colors import ListedColormap

from agents.agent import Task
from experiments import AGENT_TYPES, VALUES
from single_run_experiments import abbreviation

style.use("ggplot")
matplotlib.rcParams["savefig.directory"] = \
    "/home/simon/GoogleDrive/University Material/Bachelor Thesis/Images/Proper plots/"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 12

LOAD_DIRECTORY = "/home/simon/PycharmProjects/LowFidelitySimulation/res/new_results/"
MAP_DIRECTORY = "/home/simon/PycharmProjects/LowFidelitySimulation/res/new_experiment_maps/"

PLATE_TO_BLOCK = {
    "plate_8x8": "block_4x4x4",
    "plate_12x12": "block_5x5x6",
    "plate_16x16": "block_6x6x7",
    "plate_20x20": "block_7x7x8",
    "plate_24x24": "block_8x8x9",
    "plate_28x28": "block_9x9x10",
    "plate_32x32": "block_10x10x10",
}


def mean_and_conf_size(data, confidence_level=0.975):
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
        if file_name.startswith(map_generic) and "padded" not in file_name:
            data = load_single_map_data(file_name, "defaults")
            all_data[file_name] = data
    return all_data


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot", H="/", split_cmap=False, **kwargs):
    # taken from https://stackoverflow.com/a/22845857
    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axis = plt.subplot(111)

    counter = 0
    for df in dfall:
        cmap = None
        if split_cmap:
            cmap = kwargs["cmap"]
            color_list = list(cmap.colors)[counter * n_col:(counter + 1) * n_col]
            new_cmap = ListedColormap(sns.color_palette(color_list).as_hex())
            kwargs["cmap"] = new_cmap
        axis = df.plot(kind="bar",
                       linewidth=1,
                       stacked=True,
                       ax=axis,
                       legend=False,
                       grid=False,
                       **kwargs)
        if split_cmap:
            kwargs["cmap"] = cmap
        counter += 1

    h, l = axis.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                if not split_cmap:
                    rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))

    axis.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axis.set_xticklabels(df.index, rotation=0)
    axis.set_title(title)

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        if split_cmap:
            n.append(axis.bar(0, 0, color=kwargs["cmap"].colors[i * n_col]))
        else:
            n.append(axis.bar(0, 0, color="gray", hatch=H * i))

    if split_cmap:
        l2 = plt.legend(n, labels)
    else:
        l1 = axis.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
        if labels is not None:
            l2 = plt.legend(n, labels, loc=[1.01, 0.1])
            # l2 = plt.legend(n, labels)
        axis.add_artist(l1)
    return axis


def get_task_stats(data, task, agent_type, agent_count, complete_only=True):
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
    print("step_count mean for {} agents for task {}: {}".format(agent_count, task, step_count_total["mean"]))
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
    # color_map = ListedColormap(sns.color_palette("Blues_r", len(df.columns)).as_hex())
    color_map = ListedColormap(sns.color_palette("coolwarm", len(df.columns)).as_hex())
    ax = df.plot(kind="bar", stacked=True, colormap=color_map)
    plt.suptitle("Task proportions for {} for {}".format(agent_type, map_name))
    plt.show()


def show_task_collision_proportion(map_name, agent_type, task, experiment_name="defaults"):
    # maybe actually do this for one task only? e.g. transport which is the most important
    # and find_attachment_site and fetch_block and so on...
    # also show task distance travelled (?)

    # maybe also try showing all agent_types next to each other
    counts = [1, 2, 4, 8, 12, 16]
    data = load_single_map_data(map_name, experiment_name)
    task_normal_means = []
    task_collision_means = []
    for agent_count in counts:
        task_stats = get_task_stats(data, task, agent_type, agent_count, True)
        step_count_mean = task_stats["step_count"]["mean"]
        collision_count_mean = task_stats["collision_avoidance_count"]["mean"]
        task_normal_means.append(step_count_mean - collision_count_mean)
        task_collision_means.append(collision_count_mean)
    task_means_both = np.array([task_normal_means, task_collision_means]).transpose()

    pprint(task_means_both)

    df = pd.DataFrame(task_means_both, index=counts, columns=["normal", "collision avoidance"])
    df = df.loc[:, (df != 0).any(axis=0)]
    color_map = ListedColormap(sns.color_palette("coolwarm", len(df.columns)).as_hex())
    ax = df.plot.bar(stacked=True, colormap=color_map)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


def show_task_collision_proportion_all_agents(map_name, task, experiment_name="defaults"):
    counts = [1, 2, 4, 8, 12, 16]
    agent_types = sorted([at for at in AGENT_TYPES])

    data = load_single_map_data(map_name, experiment_name)
    data_frames = []
    for agent_type in agent_types:
        task_normal_means = []
        task_collision_means = []
        for agent_count in counts:
            task_stats = get_task_stats(data, task, agent_type, agent_count, True)
            step_count_mean = task_stats["step_count"]["mean"]
            collision_count_mean = task_stats["collision_avoidance_count"]["mean"]
            task_normal_means.append(step_count_mean - collision_count_mean)
            task_collision_means.append(collision_count_mean)
        task_means_both = np.array([task_normal_means, task_collision_means]).transpose()
        df = pd.DataFrame(task_means_both, index=counts, columns=["normal", "collision avoidance"])
        df = df.loc[:, (df != 0).any(axis=0)]
        data_frames.append(df)

    color_map = ListedColormap(sns.color_palette("coolwarm", 2).as_hex())
    ax = plot_clustered_stacked(data_frames, agent_types, cmap=color_map, edgecolor="black")
    # ax = df.plot.bar(stacked=True, colormap=color_map)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.show()


def show_task_bar_chart(map_name, agent_type, metric="step_count", statistic="mean", experiment_name="defaults"):
    counts = [1, 2, 4, 8, 12, 16]
    data = load_single_map_data(map_name, experiment_name)
    task_names = [t.name for t in Task]
    task_means_all = []
    sum_per_task = [0 for _ in task_names]
    for agent_count in counts:
        task_means = []
        for t_idx, task in enumerate(Task):
            task_stats = get_task_stats(data, task, agent_type, agent_count, True)
            print("step_count mean for {} agents for task {}: {}".format(agent_count, task,
                                                                         task_stats["step_count"]["mean"]))
            task_means.append(task_stats[metric][statistic])
            sum_per_task[t_idx] += task_stats[metric][statistic]
        task_means_all.append(task_means)
    order = sorted(range(len(task_names)), key=lambda i: sum_per_task[i], reverse=True)
    # order = sorted(range(len(task_names)), key=lambda i: task_names[i])
    task_names = [task_names[i] for i in order]
    task_means_all = [[tm[i] for i in order] for tm in task_means_all]
    task_means_all = np.array(task_means_all)

    df = pd.DataFrame(task_means_all, index=counts, columns=task_names)
    df = df.loc[:, (df != 0).any(axis=0)]
    # color_map = ListedColormap(sns.color_palette("Blues_r", len(df.columns)).as_hex())
    # test = ['#db5f57', '#b9db57', '#57db94', '#5784db', '#c957db'][:-1]
    color_map = ListedColormap(sns.color_palette("hls", len(df.columns)).as_hex()).reversed()
    # color_map = ListedColormap(sns.color_palette(test, len(df.columns)).as_hex()).reversed()
    ax = df.plot.bar(stacked=True, colormap=color_map)
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Task distribution for {} for {}".format(agent_type, map_name))
    plt.show()


def show_steps_with_standard_deviations(map_name):
    monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '=.']) *
                  cycler('marker', ['^', ',', '.']))

    # f, (ax, ax_block, ax_block_padded) = plt.subplots(1, 3)
    f, ax = plt.subplots()
    # ax.set_prop_cycle(monochrome)

    agent_types = sorted(list(AGENT_TYPES))
    agent_types = sorted(list(AGENT_TYPES), key=lambda x: (len(x)))
    print(agent_types)

    agent_counts = {at: {} for at in agent_types}
    step_counts = {at: {} for at in agent_types}
    not_completed = {at: {} for at in agent_types}

    data = load_single_map_data(map_name, "defaults")
    for d in data:
        for at in agent_types:
            if d["parameters"]["agent_type"] == at:
                if d["finished_successfully"]:
                    agent_count = d["parameters"]["agent_count"]
                    step_count = d["step_count"]
                    nc = False
                    if not d["finished_successfully"]:
                        if d["got_stuck"]:
                            print("Structure was not finished with {} {}s because they got stuck.".format(agent_count,
                                                                                                          at))
                            nc = True
                        elif d["highest_layer"] not in d["layer_completion"].keys():
                            print(
                                "Structure was not finished with {} {}s for some other reason.".format(agent_count, at))
                            nc = True
                    if agent_count not in agent_counts[at]:
                        agent_counts[at][agent_count] = []
                        step_counts[at][agent_count] = []
                        not_completed[at][agent_count] = []
                    agent_counts[at][agent_count].append(agent_count)
                    step_counts[at][agent_count].append(step_count)
                    not_completed[at][agent_count].append(nc)

    color_map = ListedColormap(sns.color_palette("Paired", len(agent_types)).as_hex())
    ax.set_prop_cycle(cycler("color", sns.color_palette()))
    # ax.set_prop_cycle(cycler("color", sns.color_palette("Paired", len(agent_types)).as_hex()))
    for k in agent_types:
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
        ax.errorbar(all_x, all_y, yerr=all_conf, capsize=2, label=k)
        print("Averages for agent type {}: {}".format(k, all_y))

    ax.set_ylim(ymin=0)
    ax.legend()
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    # ax.grid(ls="dashed")
    plt.suptitle("Steps to construction and errors for {}".format(map_name.replace("_", "\\_")))
    plt.show()


def show_steps_with_standard_deviations_new(map_name):
    agent_types = sorted(list(AGENT_TYPES), key=lambda x: (len(x)))

    data = load_single_map_data(map_name, "defaults")
    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    standard_deviations_total = []
    for agent_type in agent_types:
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                print("AC {}, AT {}: step count: {}".format(agent_count, agent_type, step_count))
                step_counts[index].append(step_count)
        standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in step_counts]]
        standard_deviations_total.append(standard_deviations)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)
    standard_deviations_total = np.array(standard_deviations_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=agent_types)
    color_map = ListedColormap(sns.color_palette(n_colors=len(df.columns)).as_hex())
    ax = df.plot(kind="line", colormap=color_map, yerr=standard_deviations_total)
    ax.set_ylim(ymin=0)
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Steps to construction and errors for {}".format(map_name.replace("_", "\\_")))
    plt.show()


def show_plate_equivalent_block_comparison_absolute(agent_type, base=4):
    counts = [1, 2, 4, 8, 12, 16]
    map_names = []
    plate_maps = sorted(list(PLATE_TO_BLOCK), key=lambda x: (len(x), x))
    for plate_map in plate_maps:
        size = int(plate_map[(plate_map.index("_") + 1):plate_map.index("x")])
        if size % base == 0:
            map_names.append(plate_map)
            map_names.append(PLATE_TO_BLOCK[plate_map])

    step_counts_total = []
    for map_name in map_names:
        data = load_single_map_data(map_name)
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_counts[index].append(d["step_count"])
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names)
    # ax = df.plot(kind="line")
    color_map = ListedColormap(sns.color_palette("Paired", len(df.columns)).as_hex())
    df.plot(kind="line", colormap=color_map)
    plt.suptitle("Plate/block comparison for {}".format(agent_type))
    plt.show()


def show_plate_equivalent_block_comparison_relative(agent_type, base=4):
    # i.e. how much longer does block construction take compared to plate construction
    counts = [1, 2, 4, 8, 12, 16]
    map_names = []
    plate_maps = sorted(list(PLATE_TO_BLOCK), key=lambda x: (len(x), x))
    for plate_map in plate_maps:
        size = int(plate_map[(plate_map.index("_") + 1):plate_map.index("x")])
        if size % base == 0:
            map_names.append(plate_map)
            map_names.append(PLATE_TO_BLOCK[plate_map])

    step_counts_total = []
    for map_name in map_names:
        data = load_single_map_data(map_name)
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_counts[index].append(d["step_count"])
        step_counts = [float(np.mean(x)) for x in step_counts]
        if len(step_counts_total) % 2 != 0:
            step_counts = [step_counts_total[-1][i] / step_counts[i] - 1.0 for i in range(len(counts))]  # plate/block
        step_counts_total.append(step_counts)
    step_counts_total = step_counts_total[1::2]
    step_counts_total = np.array(step_counts_total)
    labels = [map_names[i] + "/" + map_names[i + 1] for i in range(0, len(map_names), 2)]

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=labels)
    color_map = ListedColormap(sns.color_palette("Greys_d", len(df.columns)).as_hex())
    fig, ax = plt.subplots()
    df.plot(kind="bar", colormap=color_map, rot=0, ax=ax)
    y_locs, y_labels = plt.yticks()
    plt.xlabel("Number of agents")
    plt.ylabel("Ratio observed/expected")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    fig.canvas.draw()
    plt.yticks(y_locs, [round(float(l.get_text().replace("−", "-")) + 1, 2) for l in y_labels])
    plt.suptitle("Plate/block relative comparison for {}".format(agent_type))
    plt.show()


def show_plate_comparison(agent_type, base=4):
    # given the "base" unit of plate_4x4 (plate_8x8?), show ratio between expected time taken (e.g. 4 times longer
    # than plate_4x4 for plate_8x8 for any given agent number) and actual time taken

    counts = [1, 2, 4, 8, 12, 16]
    map_names = []
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith("plate") and "padded" not in file_name:
            if file_name not in map_names:
                map_names.append(file_name.replace(".npy", ""))
    map_names = sorted(map_names, key=lambda x: (len(x), x))
    map_sizes = [int(m[(m.index("_") + 1):m.index("x")]) for m in map_names]
    indices = [x_idx for x_idx, x in enumerate(map_sizes) if x % base == 0]
    map_names = [map_names[i] for i in indices]
    map_sizes = [map_sizes[i] for i in indices]

    base_unit = map_sizes[0] ** 2
    base_step_counts = []
    step_counts_total = []
    for m_idx, map_name in enumerate(map_names):
        data = load_single_map_data(map_name)
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_counts[index].append(d["step_count"])
        step_counts = [float(np.mean(x)) for x in step_counts]
        if m_idx == 0:
            base_step_counts.extend(step_counts)
        step_counts_total.append(step_counts)
        step_counts_total[-1] = [(step_counts_total[-1][i] / base_step_counts[i]) / (map_sizes[m_idx] ** 2 / base_unit)
                                 - 1.0 for i in range(len(counts))]
    step_counts_total = np.array(step_counts_total[1:])

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names[1:])
    color_map = ListedColormap(sns.color_palette("Greys_d", len(df.columns)).as_hex())
    fig, ax = plt.subplots()
    df.plot(kind="bar", colormap=color_map, rot=0, ax=ax)
    y_locs, y_labels = plt.yticks()
    plt.xlabel("Number of agents")
    plt.ylabel("Ratio observed/expected")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Plate comparison (reference {}) for {}".format(map_names[0], agent_type))
    fig.canvas.draw()
    plt.yticks(y_locs, [round(float(l.get_text().replace("−", "-")) + 1, 2) for l in y_labels])
    plt.show()


def show_block_comparison(agent_type):
    # pretty much the same things as for plates...
    counts = [1, 2, 4, 8, 12, 16]
    map_names = []
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith("block") and "padded" not in file_name:
            if file_name not in map_names:
                cropped = file_name.replace(".npy", "")
                size = int(cropped[(cropped.index("_") + 1):cropped.index("x")])
                if size != 8 or str(9) in file_name:
                    map_names.append(cropped)
    map_names = sorted(map_names, key=lambda x: (len(x), x))
    map_sizes = []
    for mn in map_names:
        temp = np.load(MAP_DIRECTORY + mn + ".npy").astype("int64")
        map_sizes.append(temp.shape[0] * temp.shape[1] * temp.shape[2])

    base_unit = map_sizes[0]
    base_step_counts = []
    step_counts_total = []
    for m_idx, map_name in enumerate(map_names):
        data = load_single_map_data(map_name)
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_counts[index].append(d["step_count"])
        step_counts = [float(np.mean(x)) for x in step_counts]
        if m_idx == 0:
            base_step_counts.extend(step_counts)
        step_counts_total.append(step_counts)
        step_counts_total[-1] = [(step_counts_total[-1][i] / base_step_counts[i]) / (map_sizes[m_idx] / base_unit) - 1.0
                                 for i in range(len(counts))]
    step_counts_total = np.array(step_counts_total[1:])

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names[1:])
    color_map = ListedColormap(sns.color_palette("Greys_d", len(df.columns)).as_hex())
    fig, ax = plt.subplots()
    df.plot(kind="bar", colormap=color_map, rot=0, ax=ax)
    y_locs, y_labels = plt.yticks()
    plt.xlabel("Number of agents")
    plt.ylabel("Ratio observed/expected")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    fig.canvas.draw()
    plt.yticks(y_locs, [round(float(l.get_text().replace("−", "-")) + 1, 2) for l in y_labels])
    plt.suptitle("Block comparison (reference {}) for {}".format(map_names[0], agent_type))
    plt.show()


def show_average_distance_travelled(map_name):
    counts = [1, 2, 4, 8, 12, 16]
    distances_total = []
    data = load_single_map_data(map_name, "defaults")
    agent_types = sorted(list(AGENT_TYPES.keys()))
    for at in agent_types:
        distances = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                distance_avg = 0
                counter = 0
                for t in d["task_stats"]:
                    distance_avg += d["task_stats"][t]["distance_travelled"]["mean"]
                    counter += 1
                distance_avg /= counter
                distances[index].append(distance_avg)
        distances = [float(np.mean(x)) for x in distances]
        distances_total.append(distances)
    distances_total = np.array(distances_total)

    df = pd.DataFrame(distances_total.transpose(), index=counts, columns=agent_types)
    ax = df.plot(kind="line")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.suptitle("Average distance travelled for {}".format(map_name))
    plt.show()


def show_sp_researching(map_name, agent_type):
    counts = [1, 2, 4, 8, 12, 16]

    data = load_single_map_data(map_name, "defaults")
    research_counts = [[] for _ in counts]
    for d in data:
        if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
            agent_count = d["parameters"]["agent_count"]
            index = counts.index(agent_count)
            research_count_avg = 0
            counter = 0
            for a in d["sp_number_search_count"]:
                for count in a:
                    research_count_avg += count[0]
                    counter += 1
            research_count_avg /= counter
            research_counts[index].append(research_count_avg)
    research_counts = [float(np.mean(x)) for x in research_counts]

    f, ax = plt.subplots()
    ax.plot(counts, research_counts)
    plt.suptitle("Average re-searching for {} for {}".format(agent_type, map_name))
    plt.show()


def show_sp_researching_generic(map_generic, agent_type):
    counts = [1, 2, 4, 8, 12, 16]
    map_names = []
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith(map_generic) and not file_name.endswith("padded"):
            if file_name not in map_names:
                if "block" in map_generic and str(8) in file_name and str(9) not in file_name:
                    continue
                map_names.append(file_name.replace(".npy", ""))
    map_names = sorted(map_names, key=lambda x: (len(x), x))

    research_counts_total = []
    standard_deviations_total = []
    for map_name in map_names:
        data = load_single_map_data(map_name, "defaults")
        research_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                research_count_avg = 0
                counter = 0
                for a in d["sp_number_search_count"]:
                    for count in a:
                        research_count_avg += count[0]
                        counter += 1
                research_count_avg /= counter
                research_counts[index].append(research_count_avg)
        standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in research_counts]]
        standard_deviations_total.append(standard_deviations)
        research_counts = [float(np.mean(x)) for x in research_counts]
        research_counts_total.append(research_counts)
    research_counts_total = np.array(research_counts_total)
    standard_deviations_total = np.array(standard_deviations_total)

    df = pd.DataFrame(research_counts_total.transpose(), index=counts, columns=map_names)
    color_map = ListedColormap(sns.color_palette("hls", n_colors=len(df.columns)))
    ax = df.plot(kind="line", colormap=color_map, yerr=standard_deviations_total, capsize=2)
    plt.xlabel("Number of agents")
    plt.ylabel("Number of searches\nbefore attachment")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Average re-searching for {} for {} generic".format(agent_type, map_generic))
    plt.show()


def show_attached_block_count_distribution(map_name, agent_type):
    counts = [1, 2, 4, 8, 12, 16]
    data = load_single_map_data(map_name)
    attached_counts = [[] for _ in counts]
    for d in data:
        if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
            agent_count = d["parameters"]["agent_count"]
            index = counts.index(agent_count)
            attached_count = d["attached_block_counts"]
            attached_counts[index].extend(attached_count)

    f, ax = plt.subplots()
    average = []
    lower_range = []
    upper_range = []
    for c_idx, c in enumerate(counts):
        avg = np.mean(attached_counts[c_idx])
        average.append(avg)
        lower_range.append(abs(min(attached_counts[c_idx]) - avg))
        upper_range.append(abs(max(attached_counts[c_idx]) - avg))
    asymmetric_range = [lower_range, upper_range]
    ax.errorbar(counts, average, yerr=asymmetric_range, capsize=2)
    ax.set_ylim(ymin=0)
    plt.show()


def show_attached_block_count_distribution_all_types(map_name):
    agent_types = sorted(list(AGENT_TYPES), key=lambda x: (len(x), x))
    data = load_single_map_data(map_name)
    counts = [1, 2, 4, 8, 12, 16]
    attached_counts_total = []
    standard_deviations_total = []
    for agent_type in agent_types:
        attached_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                attached_count = d["attached_block_counts"]
                attached_counts[index].extend(attached_count)
        print(attached_counts)
        standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in attached_counts]]
        # standard_deviations = [max(y) for y in [abs(np.mean(x)) for x in attached_counts]]
        standard_deviations_total.append(standard_deviations)
        attached_counts = [float(np.mean(x)) for x in attached_counts]
        attached_counts_total.append(attached_counts)
    standard_deviations_total = np.array(standard_deviations_total)
    attached_counts_total = np.array(attached_counts_total)

    df = pd.DataFrame(attached_counts_total.transpose(), index=counts, columns=agent_types)
    ax = df.plot(kind="line", yerr=standard_deviations_total, capsize=2)
    ax.set_ylim(ymin=0)
    plt.show()


def show_collision_proportion(map_name):
    # would be good to break this down by task
    counts = [1, 2, 4, 8, 12, 16]
    data = load_single_map_data(map_name, experiment_name="defaults")
    agent_types = sorted(list(AGENT_TYPES.keys()), key=lambda x: (len(x), x))
    data_frames = []
    for at in agent_types:
        steps_without_collisions = [[] for _ in counts]
        steps_with_collisions = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_avg = 0
                collision_avg = 0
                counter = 0
                for t in d["task_stats"]:
                    step_avg += d["task_stats"][t]["step_count"]["mean"]
                    collision_avg += d["task_stats"][t]["collision_avoidance_count"]["mean"]
                    counter += 1
                print("AC {}, AT {}: Step sum from tasks: {}".format(agent_count, at, step_avg))
                print("AC {}, AT {}: Step sum from outside: {}".format(agent_count, at, d["step_count"]))
                # step_avg /= agent_count
                # collision_avg /= agent_count
                steps_without_collisions[index].append(step_avg - collision_avg)
                steps_with_collisions[index].append(collision_avg)
                if agent_count == 16:
                    print(collision_avg)
        steps_without_collisions = [float(np.mean(x)) for x in steps_without_collisions]
        steps_with_collisions = [float(np.mean(x)) for x in steps_with_collisions]
        print(steps_with_collisions)
        total = np.array([steps_without_collisions, steps_with_collisions])
        df = pd.DataFrame(total.transpose(), index=counts, columns=["Steps w/o collision", "Steps w/ collision"])
        data_frames.append(df)

    # df = pd.DataFrame(proportions_total.transpose(), index=counts, columns=agent_types)
    color_map = ListedColormap(sns.color_palette("Paired", len(agent_types) * 2).as_hex()).reversed()
    # df.plot(kind="line", colormap=color_map)
    plot_clustered_stacked(data_frames, agent_types, title="Average collision proportion for {}".format(map_name),
                           split_cmap=True, cmap=color_map, edgecolor="black")
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.show()


def show_collision_proportion_for_type(map_name, agent_type):
    # would be good to break this down by task
    counts = [1, 2, 4, 8, 12, 16]
    data = load_single_map_data(map_name)
    without_collision = [[] for _ in counts]
    with_collision = [[] for _ in counts]
    for d in data:
        if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
            agent_count = d["parameters"]["agent_count"]
            index = counts.index(agent_count)
            step_avg = 0
            collision_avg = 0
            counter = 0
            for t in d["task_stats"]:
                step_avg += d["task_stats"][t]["step_count"]["mean"]
                collision_avg += d["task_stats"][t]["collision_avoidance_count"]["mean"]
                counter += 1
            step_avg /= counter
            collision_avg /= counter
            without_collision[index].append(step_avg - collision_avg)
            with_collision[index].append(collision_avg)
    without_collision = [float(np.mean(x)) for x in without_collision]
    with_collision = [float(np.mean(x)) for x in with_collision]
    total = np.array([without_collision, with_collision])

    df = pd.DataFrame(total.transpose(), index=counts, columns=["W/o collisions", "W/ collisions"])
    color_map = ListedColormap(sns.color_palette("Greys_d", 2).as_hex())
    print(sns.color_palette("Greys_d", 2))
    df.plot(kind="bar", stacked=True, colormap=color_map, rot=0)
    plt.suptitle("Average collision proportion for {}".format(map_name))
    plt.show()


def show_all_task_collision_proportion(map_name, agent_type, agent_count):
    # for each task, show a bar of total and collision avoiding moves
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
    map_names = sorted(map_names, key=lambda x: (len(x), x))

    step_counts_total = []
    standard_deviations_total = []
    for map_name in map_names:
        data = load_single_map_data(map_name, "defaults")
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in step_counts]]
        standard_deviations_total.append(standard_deviations)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    standard_deviations_total = np.array(standard_deviations_total)
    step_counts_total = np.array(step_counts_total)
    print(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names)
    color_map = ListedColormap(sns.color_palette("Blues_d", len(df.columns)).as_hex()).reversed()
    ax = df.plot(kind="line", yerr=standard_deviations_total, colormap=color_map, capsize=2)
    ax.set_ylim(ymin=0)
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Perimeter comparison for {}".format(agent_type))
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
    map_names = sorted(map_names)

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
    color_map = ListedColormap(sns.color_palette("Paired", len(df.columns)).as_hex())
    ax = df.plot(kind="line", colormap=color_map)
    ax.set_axisbelow(True)
    plt.show()


def show_multiple_component_scaling_relative(agent_type):
    map_names = []
    map_sizes = []
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith("component"):
            if file_name not in map_names:
                map_names.append(file_name.replace(".npy", ""))
                ms = int(file_name[file_name.index("_") + 1:file_name.index("x")]) * \
                     int(file_name[file_name.index("x") + 1:file_name.index("x") + 2])
                map_sizes.append(ms)
    order = sorted(range(len(map_names)), key=lambda i: map_names[i], reverse=True)
    # map_names = sorted(map_names, reverse=True)
    map_names = [map_names[i] for i in order]
    map_sizes = [map_sizes[i] for i in order]

    base_unit = 3 ** 2

    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    base_step_counts = []
    for m_idx, m in enumerate(map_names):
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
        if m_idx == 0:
            base_step_counts.extend(step_counts)
        step_counts_total.append(step_counts)
        # if it's les than one, then
        step_counts_total[-1] = [(step_counts_total[-1][i] / base_step_counts[i]) / (map_sizes[m_idx] / base_unit)
                                 - 1.0 for i in range(len(counts))]
    step_counts_total = np.array(step_counts_total[-1:0:-1])
    # map_names = list(reversed(map_names))

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=map_names[-1:0:-1])
    color_map = ListedColormap(sns.color_palette("Paired", len(df.columns)).as_hex())
    fig, ax = plt.subplots()
    df.plot(kind="bar", colormap=color_map, rot=0, ax=ax)
    y_locs, y_labels = plt.yticks()
    plt.xlabel("Number of agents")
    plt.ylabel("Ratio observed/expected")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    fig.canvas.draw()
    plt.yticks(y_locs, [round(float(l.get_text().replace("−", "-")) + 1, 2) for l in y_labels])
    plt.suptitle("Multiple component scaling for {}".format(agent_type))
    plt.show()


def show_multiple_component_comparison(agent_type):
    pass


def show_component_finished_delay(map_name):
    counts = [1, 2, 4, 8, 12, 16]
    delay_counts_total = []
    data = load_single_map_data(map_name, "defaults")
    agent_types = sorted([at for at in AGENT_TYPES.keys() if not at.startswith("Global")])
    for at in agent_types:
        delay_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                counter = 0
                delay_average = 0
                for agent in d["complete_to_switch_delay"]:
                    for component in agent:
                        delay_average += agent[component]
                        counter += 1
                delay_average /= counter
                delay_counts[index].append(delay_average)
        delay_counts = [float(np.mean(x)) for x in delay_counts]
        delay_counts_total.append(delay_counts)
    delay_counts_total = np.array(delay_counts_total)

    df = pd.DataFrame(delay_counts_total.transpose(), index=counts, columns=agent_types)
    ax = df.plot(kind="line")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="grey", linestyle='dashed')
    plt.suptitle("Average delay to realising component is finished for {}".format(map_name))
    plt.show()


def show_all_agent_performance(map_name, experiment_name="defaults"):
    # essentially just show performance for all agent counts given the parameters
    agent_counts = {at: {} for at in AGENT_TYPES}
    step_counts = {at: {} for at in AGENT_TYPES}
    not_completed = {at: {} for at in AGENT_TYPES}
    data = load_single_map_data(map_name, experiment_name)
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
        ax.errorbar(all_x, all_y, yerr=all_conf, label=k, capsize=2)

    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def show_single_agent_type_performance(map_name, agent_type, experiment_name="defaults"):
    counts = [1, 2, 4, 8, 12, 16]
    data = load_single_map_data(map_name, experiment_name)
    step_counts = [[] for _ in counts]
    for d in data:
        if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
            agent_count = d["parameters"]["agent_count"]
            index = counts.index(agent_count)
            step_counts[index].append(d["step_count"])
    step_counts = [float(np.mean(x)) for x in step_counts]

    f, ax = plt.subplots()
    ax.plot(counts, step_counts, label=experiment_name)
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.suptitle("Average step counts for {} for {}".format(agent_type, map_name))
    plt.show()


def show_wait_on_perimeter_difference(map_name, agent_type):
    counts = [1, 2, 4, 8, 12, 16]
    normal_data = load_single_map_data(map_name, "defaults")
    normal_step_counts = [[] for _ in counts]
    for d in normal_data:
        if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
            agent_count = d["parameters"]["agent_count"]
            index = counts.index(agent_count)
            normal_step_counts[index].append(d["step_count"])
    normal_step_counts = [float(np.mean(x)) for x in normal_step_counts]

    wop_step_counts = None
    wop_exists = os.path.exists(LOAD_DIRECTORY + map_name + "/wait_on_perimeter")
    if wop_exists:
        wop_data = load_single_map_data(map_name, "wait_on_perimeter")
        wop_step_counts = [[] for _ in counts]
        for d in wop_data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                wop_step_counts[index].append(d["step_count"])
        wop_step_counts = [float(np.mean(x)) for x in wop_step_counts]

    f, ax = plt.subplots()
    ax.plot(counts, normal_step_counts, label="Normal")
    if wop_exists:
        ax.plot(counts, wop_step_counts, label="Wait on perimeter")
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def show_wait_on_perimeter_difference_all_types(map_name, wop_experiment_name="wait_on_perimeter"):
    agent_types = sorted(list(AGENT_TYPES), key=lambda x: (len(x), x))

    counts = [1, 2, 4, 8, 12, 16]
    normal_data = load_single_map_data(map_name, "defaults")

    normal_step_counts_total = []
    normal_standard_deviations_total = []
    for agent_type in agent_types:
        normal_step_counts = [[] for _ in counts]
        for d in normal_data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                normal_step_counts[index].append(d["step_count"])
        standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in normal_step_counts]]
        normal_standard_deviations_total.append(standard_deviations)
        normal_step_counts = [float(np.mean(x)) for x in normal_step_counts]
        normal_step_counts_total.append(normal_step_counts)

    wop_step_counts_total = None
    wop_standard_deviations_total = None
    wop_exists = os.path.exists(LOAD_DIRECTORY + map_name + "/" + wop_experiment_name)
    if wop_exists:
        wop_data = load_single_map_data(map_name, wop_experiment_name)
        wop_step_counts_total = []
        wop_standard_deviations_total = []
        for agent_type in agent_types:
            wop_step_counts = [[] for _ in counts]
            for d in wop_data:
                if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                    agent_count = d["parameters"]["agent_count"]
                    index = counts.index(agent_count)
                    wop_step_counts[index].append(d["step_count"])
            standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in wop_step_counts]]
            wop_standard_deviations_total.append(standard_deviations)
            wop_step_counts = [float(np.mean(x)) for x in wop_step_counts]
            wop_step_counts_total.append(wop_step_counts)
    else:
        print("No '{}' experiment exists.".format(wop_experiment_name))

    f, ax = plt.subplots()
    colors = sns.color_palette("Paired", 8)
    for at_idx, at in enumerate(agent_types):
        # ax.plot(counts, normal_step_counts_total[at_idx],
        #         label="{} Normal".format(abbreviation(at)),
        #         color=colors[at_idx * 2 + 1])
        ax.errorbar(counts, normal_step_counts_total[at_idx],
                    yerr=normal_standard_deviations_total[at_idx],
                    label="{} Normal".format(abbreviation(at)),
                    color=colors[at_idx * 2 + 1])
        if wop_exists:
            # ax.plot(counts, wop_step_counts_total[at_idx],
            #         label="{} WOP".format(abbreviation(at)),
            #         color=colors[at_idx * 2])
            ax.errorbar(counts, wop_step_counts_total[at_idx],
                        yerr=wop_standard_deviations_total[at_idx],
                        label="{} WOP".format(abbreviation(at)),
                        color=colors[at_idx * 2])
    ax.set_ylim(ymin=0)
    ax.legend(loc="upper left")
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Waiting on perimeter for {}".format(map_name))
    plt.show()


def show_wait_on_perimeter_collision_proportion_difference(map_name, agent_type):
    counts = [1, 2, 4, 8, 12, 16]
    normal_data = load_single_map_data(map_name, "defaults")
    normal_step_counts = [[] for _ in counts]
    for d in normal_data:
        if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
            agent_count = d["parameters"]["agent_count"]
            index = counts.index(agent_count)
            step_avg = 0
            collision_avg = 0
            counter = 0
            for t in d["task_stats"]:
                step_avg += d["task_stats"][t]["step_count"]["mean"]
                collision_avg += d["task_stats"][t]["collision_avoidance_count"]["mean"]
                counter += 1
            step_avg /= counter
            collision_avg /= counter
            normal_step_counts[index].append(collision_avg / step_avg)
    normal_step_counts = [float(np.mean(x)) for x in normal_step_counts]

    wop_step_counts = None
    wop_exists = os.path.exists(LOAD_DIRECTORY + map_name + "/wait_on_perimeter")
    if wop_exists:
        wop_data = load_single_map_data(map_name, "wait_on_perimeter")
        wop_step_counts = [[] for _ in counts]
        for d in wop_data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_avg = 0
                collision_avg = 0
                counter = 0
                for t in d["task_stats"]:
                    step_avg += d["task_stats"][t]["step_count"]["mean"]
                    collision_avg += d["task_stats"][t]["collision_avoidance_count"]["mean"]
                    counter += 1
                step_avg /= counter
                collision_avg /= counter
                wop_step_counts[index].append(collision_avg / step_avg)
        wop_step_counts = [float(np.mean(x)) for x in wop_step_counts]

    f, ax = plt.subplots()
    ax.plot(counts, normal_step_counts, label="Normal")
    if wop_exists:
        ax.plot(counts, wop_step_counts, label="Wait on perimeter")
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.show()


def show_component_local_global_proportions():
    map_names = []
    for file_name in os.listdir(LOAD_DIRECTORY):
        if file_name.startswith("component"):
            if file_name not in map_names:
                map_names.append(file_name.replace(".npy", ""))
    map_names = sorted(map_names)

    f, ax = plt.subplots()
    for m in map_names:
        show_local_global_proportion(m, ax)
    plt.suptitle("All global/local proportions for component maps")
    plt.show()


def show_local_global_proportion(map_name, ax=None):
    # proportion being global/local
    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = {}
    data = load_single_map_data(map_name, "defaults")
    agent_types = sorted([at for at in AGENT_TYPES.keys()])
    for at in agent_types:
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_counts[index].append(d["step_count"])
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total[at] = step_counts

    perimeter_following_proportion = [step_counts_total["GlobalPerimeterFollowingAgent"][i] /
                                      step_counts_total["LocalPerimeterFollowingAgent"][i] for i in range(len(counts))]
    shortest_path_proportion = [step_counts_total["GlobalShortestPathAgent"][i] /
                                step_counts_total["LocalShortestPathAgent"][i] for i in range(len(counts))]

    plot_self = False
    if ax is None:
        f, ax = plt.subplots()
        plot_self = True
    plt.set_cmap(ListedColormap(sns.color_palette().as_hex()))
    ax.plot(counts, perimeter_following_proportion, label="PF {}".format(map_name))
    ax.plot(counts, shortest_path_proportion, label="SP {}".format(map_name))
    ax.set_ylim(ymin=0)
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    ax.legend()
    if plot_self:
        plt.suptitle("Ratio global/local for {}".format(map_name))
        plt.show()


def show_all_local_global_proportions(map_names):
    counts = [1, 2, 4, 8, 12, 16]
    total = []
    labels = []
    for map_name in map_names:
        step_counts_total = {}
        data = load_single_map_data(map_name, "defaults")
        agent_types = sorted([at for at in AGENT_TYPES.keys()])
        for at in agent_types:
            step_counts = [[] for _ in counts]
            for d in data:
                if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                    agent_count = d["parameters"]["agent_count"]
                    index = counts.index(agent_count)
                    step_counts[index].append(d["step_count"])
            step_counts = [float(np.mean(x)) for x in step_counts]
            step_counts_total[at] = step_counts

        perimeter_following_proportion = [step_counts_total["GlobalPerimeterFollowingAgent"][i] /
                                          step_counts_total["LocalPerimeterFollowingAgent"][i]
                                          for i in range(len(counts))]
        shortest_path_proportion = [step_counts_total["GlobalShortestPathAgent"][i] /
                                    step_counts_total["LocalShortestPathAgent"][i] for i in range(len(counts))]

        total.append(perimeter_following_proportion)
        total.append(shortest_path_proportion)

        labels.append("{} (PF)".format(map_name))
        labels.append("{} (SP)".format(map_name))
    total = np.array(total)

    df = pd.DataFrame(total.transpose(), index=counts, columns=labels)
    color_map = ListedColormap(sns.color_palette("Paired", len(df.columns)).as_hex())
    ax = df.plot(kind="line", colormap=color_map)
    ax.set_ylim(ymin=0)
    plt.xlabel("Number of agents")
    plt.ylabel("Fraction steps global/local")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    # plt.suptitle("Ratio global/local for\n{}".format(map_names))
    plt.suptitle("Ratio global/local for")
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
    standard_deviations_total = []
    for o in options:
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"] \
                    and "order_only_one_metric" in d["parameters"] and d["parameters"]["order_only_one_metric"] \
                    and parameter_name in d["parameters"] and d["parameters"][parameter_name] == o:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in step_counts]]
        standard_deviations_total.append(standard_deviations)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    standard_deviations_total = np.array(standard_deviations_total)
    step_counts_total = np.array(step_counts_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=options)
    ax = df.plot(kind="line", yerr=standard_deviations_total, capsize=2)
    ax.set_ylim(ymin=0)
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("AS ordering for {} for {}".format(agent_type, map_name))
    plt.show()


def show_component_ordering_differences(agent_type, map_name, seed_if_possible=False):
    if agent_type.startswith("Local"):
        parameter_name = "seeding_strategy"
    else:
        parameter_name = "component_ordering"
    options = VALUES[parameter_name]

    data = load_single_map_data(map_name)
    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    standard_deviations_total = []
    for o in options:
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"] \
                    and parameter_name in d["parameters"] and d["parameters"][parameter_name] == o \
                    and (not agent_type.startswith("Local")
                         or (agent_type.startswith("Local")
                             and d["parameters"]["seed_if_possible_enabled"] == seed_if_possible)):
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_count = d["step_count"]
                step_counts[index].append(step_count)
        standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in step_counts]]
        standard_deviations_total.append(standard_deviations)
        step_counts = [float(np.mean(x)) for x in step_counts]
        step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)
    standard_deviations_total = np.array(standard_deviations_total)

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=options)
    ax = df.plot(kind="line", yerr=standard_deviations_total, capsize=2)
    ax.set_ylim(ymin=0)
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Component ordering for {} for \n{} (seed_if_possible: {})"
                 .format(agent_type, map_name, seed_if_possible))
    plt.show()


def show_component_ordering_differences_both_seeding_types(agent_type, map_name):
    if not agent_type.startswith("Local"):
        return
    options = VALUES["seeding_strategy"]

    data = load_single_map_data(map_name)
    counts = [1, 2, 4, 8, 12, 16]
    step_counts_total = []
    standard_deviations_total = []
    for o in options:
        for seed in (False, True):
            step_counts = [[] for _ in counts]
            for d in data:
                if d["parameters"]["agent_type"] == agent_type and d["finished_successfully"] \
                        and "seeding_strategy" in d["parameters"] and d["parameters"]["seeding_strategy"] == o \
                        and (not agent_type.startswith("Local")
                             or (agent_type.startswith("Local")
                                 and d["parameters"]["seed_if_possible_enabled"] == seed)):
                    agent_count = d["parameters"]["agent_count"]
                    index = counts.index(agent_count)
                    step_count = d["step_count"]
                    step_counts[index].append(step_count)
            standard_deviations = [conf for _, conf in [mean_and_conf_size(x) for x in step_counts]]
            standard_deviations_total.append(standard_deviations)
            step_counts = [float(np.mean(x)) for x in step_counts]
            step_counts_total.append(step_counts)
    step_counts_total = np.array(step_counts_total)
    standard_deviations_total = np.array(standard_deviations_total)

    combined_options = []
    for o in options:
        combined_options.append("{} (no seed)".format(o))
        combined_options.append("{} (seed)".format(o))

    df = pd.DataFrame(step_counts_total.transpose(), index=counts, columns=combined_options)
    color_map = ListedColormap(sns.color_palette("Paired", len(df.columns)).as_hex())
    ax = df.plot(kind="line", yerr=standard_deviations_total, capsize=2, colormap=color_map)
    ax.set_ylim(ymin=0)
    plt.xlabel("Number of agents")
    plt.ylabel("Steps")
    plt.gcf().subplots_adjust(left=0.17, right=0.95)
    plt.suptitle("Component ordering for {} for {}".format(agent_type, map_name))
    plt.show()


def show_sp_per_search_attachment_site_count(map_name, agent_count=1):
    # just use agent count 1 for now
    data = load_single_map_data(map_name, "defaults")
    agent_types = sorted([at for at in AGENT_TYPES if "Perimeter" not in at])
    longest_step_number = 0
    per_search_attachment_site_count = []
    for d in data:
        for at in agent_types:
            if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                ac = d["parameters"]["agent_count"]
                if ac == agent_count:
                    psatc = d["per_search_attachment_site_count"][0]["possible"]
                    per_search_attachment_site_count.append(psatc)
                    if len(psatc) > longest_step_number:
                        longest_step_number = len(psatc)
                    break
                # index = counts.index(agent_count)

    f, ax = plt.subplots()
    x_vals = np.arange(longest_step_number)
    for at_idx, at in enumerate(agent_types):
        if len(per_search_attachment_site_count[at_idx]) < len(x_vals):
            rest = [per_search_attachment_site_count[at_idx][-1]] * \
                   (len(x_vals) - len(per_search_attachment_site_count[at_idx]))
            rest = [0] * (len(x_vals) - len(per_search_attachment_site_count[at_idx]))
            per_search_attachment_site_count[at_idx].extend(rest)
        ax.plot(x_vals, per_search_attachment_site_count[at_idx], label=at)
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.suptitle("Attachment sites at each moment for {} agent(s) for {}".format(agent_count, map_name))
    plt.show()


def show_agents_over_construction_area(map_name, experiment_name="defaults", agent_count=1):
    data = load_single_map_data(map_name, experiment_name)
    agent_types = sorted([at for at in AGENT_TYPES])
    longest_step_number = 0
    agents_over_construction_area = []
    for d in data:
        for at in agent_types:
            if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                ac = d["parameters"]["agent_count"]
                if ac == agent_count:
                    aoca = d["agents_over_construction_area"]
                    agents_over_construction_area.append(aoca)
                    if len(aoca) > longest_step_number:
                        longest_step_number = len(aoca)
                    break

    f, ax = plt.subplots()
    x_vals = np.arange(longest_step_number)
    for at_idx, at in enumerate(agent_types):
        if len(agents_over_construction_area[at_idx]) < len(x_vals):
            rest = [0] * (len(x_vals) - len(agents_over_construction_area[at_idx]))
            agents_over_construction_area[at_idx].extend(rest)
        ax.plot(x_vals, agents_over_construction_area[at_idx], label=at)
    ax.set_ylim(ymin=0)
    ax.legend()
    plt.suptitle("Agents over the structure over time for {} agent(s) for {}".format(agent_count, map_name))
    plt.show()


def show_comparison(map_names, agent_types, experiment_names):
    f, ax = plt.subplots()
    counts = [1, 2, 4, 8, 12, 16]
    for m_idx, map_name in enumerate(map_names):
        data = load_single_map_data(map_name, experiment_names[m_idx])
        step_counts = [[] for _ in counts]
        for d in data:
            if d["parameters"]["agent_type"] == agent_types[m_idx] and d["finished_successfully"]:
                agent_count = d["parameters"]["agent_count"]
                index = counts.index(agent_count)
                step_counts[index].append(d["step_count"])
        step_counts = [float(np.mean(x)) for x in step_counts]
        ax.plot(counts, step_counts, label="{}-{}-{}".format(map_name, agent_types[m_idx], experiment_names[m_idx]))

    ax.set_ylim(ymin=0)
    ax.legend()
    plt.suptitle("Comparison")
    plt.show()


def show_distribution_global_local_all():
    counts = [1, 2, 4, 8, 12, 16]
    agent_types = sorted([at for at in AGENT_TYPES.keys()])
    pf_proportions = [[] for _ in counts]
    sp_proportions = [[] for _ in counts]
    for map_name in os.listdir(LOAD_DIRECTORY):
        if "defaults" in os.listdir(LOAD_DIRECTORY + map_name):
            data = load_single_map_data(map_name, "defaults")
            step_counts_total = {}
            for at in agent_types:
                step_counts = [[] for _ in counts]
                for d in data:
                    if d["parameters"]["agent_type"] == at and d["finished_successfully"]:
                        agent_count = d["parameters"]["agent_count"]
                        index = counts.index(agent_count)
                        step_counts[index].append(d["step_count"])
                step_counts = [float(np.mean(x)) for x in step_counts]
                step_counts_total[at] = step_counts

            perimeter_following_proportion = [step_counts_total["GlobalPerimeterFollowingAgent"][i] /
                                              step_counts_total["LocalPerimeterFollowingAgent"][i]
                                              for i in range(len(counts))]
            shortest_path_proportion = [step_counts_total["GlobalShortestPathAgent"][i] /
                                        step_counts_total["LocalShortestPathAgent"][i] for i in range(len(counts))]

            for i in range(len(counts)):
                if not np.isnan(perimeter_following_proportion[i]):
                    pf_proportions[i].append(perimeter_following_proportion[i])
                if not np.isnan(shortest_path_proportion[i] != np.nan):
                    sp_proportions[i].append(shortest_path_proportion[i])

            # pf_proportions.append(perimeter_following_proportion)
            # sp_proportions.append(shortest_path_proportion)
    # plt.boxplot(pf_proportions, counts)
    # plt.show()
    test = {"PF": pf_proportions, "SP": sp_proportions}
    df = pd.DataFrame(test)

    sns.boxplot(data=pf_proportions)
    plt.ylim(ymin=0)
    plt.xticks(range(len(counts)), counts)
    plt.show()


def main():
    # TODO: need some easy way to get the yerr
    map_name = "components_6x6x1"
    # data = load_single_map_data(map_name)
    # show_all_agent_performance("block_10x10x10")
    # show_single_agent_type_performance("block_6x6x7", "GlobalShortestPathAgent",
    #                                    "wait_on_perimeter_higher_density_test")
    # show_comparison(["block_6x6x7"] * 3, ["GlobalShortestPathAgent"] * 3,
    #                 ["wait_on_perimeter_higher_density_test", "wait_on_perimeter", "defaults"])
    # show_scaling_performance(data, parameters)
    # show_steps_with_standard_deviations("block_4x4x4")
    # show_steps_with_standard_deviations_new("block_4x4x4")
    # show_plate_block_comparison("block_4x4x4")
    # show_local_sp_researching_generic("plate")
    # show_average_distance_travelled("plate_8x8")
    # show_collision_proportion("block_6x6x7")
    # show_collision_proportion_for_type("plate_32x32", "GlobalShortestPathAgent")
    # show_attached_block_count_distribution("plate_16x16", "LocalShortestPathAgent")
    # show_attached_block_count_distribution_all_types("block_10x10x10")
    # show_component_finished_delay("component_3x3")
    # show_wait_on_perimeter_difference("block_6x6x7", "GlobalShortestPathAgent")
    # show_wait_on_perimeter_difference_all_types("hole_half_width_8", "wait_on_perimeter_4_blocks_only")
    show_wait_on_perimeter_difference_all_types("plate_8x8", "wait_on_perimeter_4")
    # show_wait_on_perimeter_collision_proportion_difference("block_4x4x4", "LocalPerimeterFollowingAgent")
    # show_perimeter_comparison("GlobalPerimeterFollowingAgent")
    # show_local_global_proportion("plate_32x32")
    # show_all_local_global_proportions(["plate_8x8", "plate_32x32", "block_4x4x4", "block_10x10x10"])
    # show_all_local_global_proportions(["perim_4", "perim_8", "perim_12", "perim_16"])
    # show_all_local_global_proportions(["component_1x1", "component_2x2", "component_3x3"])
    # show_multiple_component_scaling("LocalPerimeterFollowingAgent")
    # show_multiple_component_scaling_relative("LocalPerimeterFollowingAgent")
    # show_component_local_global_proportions()
    # show_sp_per_search_attachment_site_count("hole_same_32", 2)
    # show_agents_over_construction_area("block_8x8x8", agent_count=8)
    # show_task_bar_chart("block_6x6x7", "GlobalShortestPathAgent", experiment_name="defaults")
    # show_task_area_chart("block_6x6x7", "GlobalShortestPathAgent", experiment_name="wait_on_perimeter")
    # show_task_collision_proportion("block_6x6x7", "GlobalShortestPathAgent", Task.TRANSPORT_BLOCK)
    # show_task_collision_proportion_all_agents("block_6x6x7", Task.TRANSPORT_BLOCK)
    # show_task_proportions("block_6x6x7", "GlobalShortestPathAgent")
    # show_task_proportions("plate_16x16", "GlobalShortestPathAgent")
    # show_plate_comparison("LocalPerimeterFollowingAgent", base=4)
    # show_block_comparison("LocalPerimeterFollowingAgent")
    # show_plate_equivalent_block_comparison_absolute("LocalShortestPathAgent", 4)
    # show_plate_equivalent_block_comparison_relative("LocalShortestPathAgent", 4)
    # show_sp_researching_generic("plate", "LocalShortestPathAgent")
    # show_steps_with_standard_deviations("block_6x6x7")

    # show_distribution_global_local_all()

    # show_task_proportions("block_10x10x10", "GlobalShortestPathAgent", metric="collision_avoidance_count")
    # data = load_matching_map_data("plate")
    # show_perimeter_comparison("LocalPerimeterFollowingAgent")
    # show_upward_scaling("GlobalShortestPathAgent")
    # show_multiple_component_scaling("GlobalShortestPathAgent")
    # show_perimeter_comparison("LocalPerimeterFollowingAgent")
    # show_attachment_site_ordering_differences("LocalShortestPathAgent", "hole_half_width_24")
    # -> probably doesn't show any real differences because even when agent count is first ordering criterion
    #    the distance thing is still second
    # show_component_ordering_differences("GlobalShortestPathAgent", "seed_comp_4x4_2_1")
    # show_component_ordering_differences_both_seeding_types("LocalPerimeterFollowingAgent", "seed_comp_4x4_3_1")


if __name__ == "__main__":
    main()
