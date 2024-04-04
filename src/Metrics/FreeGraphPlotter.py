import os
import shutil

import numpy as np
from matplotlib import pyplot as plt


class FreeGraphPlotter:

    @staticmethod
    def plot_filled_graph(plot_axis, steps, values, label, colour='blue'):
        mean_rewards = np.mean(values, axis=0)
        min_rewards = np.min(values, axis=0)
        max_rewards = np.max(values, axis=0)

        plot_axis.plot(steps * 15, mean_rewards, label=label, linewidth=1.0, color=colour)
        plot_axis.fill_between(steps * 15, min_rewards, max_rewards, color=colour, alpha=0.05)
        plot_axis.grid(True)

    # Given a directory containing n number of runs (rewards.txt) of the same type, plot the average and min/max range
    @staticmethod
    def plot_average(directory, r_avg_window_size=100, plot_graphs=True):
        parent_dir_name = directory.rstrip("/").split("/")[-1]

        stacked_rewards = np.empty(0)
        stacked_speeds = np.empty(0)

        steps = np.empty(0)

        file_count = 0
        for item in os.listdir(directory):
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):
                if "rewards" not in item and "returns" not in item:
                    continue
                steps, optimal_policy_rewards, optimal_policy_speeds = np.loadtxt(file_path, delimiter=',')

                steps = steps[r_avg_window_size - 1:]

                rolling_avg_rewards = np.convolve(optimal_policy_rewards,
                                                  np.ones(r_avg_window_size) / r_avg_window_size,
                                                  mode='valid')
                rolling_avg_speeds = np.convolve(optimal_policy_speeds,
                                                 np.ones(r_avg_window_size) / r_avg_window_size,
                                                 mode='valid')

                stacked_rewards = rolling_avg_rewards if stacked_rewards.size == 0 else np.vstack(
                    (stacked_rewards, rolling_avg_rewards))
                stacked_speeds = rolling_avg_speeds if stacked_speeds.size == 0 else np.vstack(
                    (stacked_speeds, rolling_avg_speeds))

                file_count += 1

        if not plot_graphs:
            return steps, stacked_rewards, stacked_speeds

        # plots[key][0] = [plt_fig, plt_axis] --> eg: plots[key][0][0] = plt_fig
        # plots[key][1] = the values (rewards, speeds, etc)
        plots = {
            "Rolling Average Reward": [plt.subplots(), stacked_rewards],
            "Rolling Average Speed": [plt.subplots(), stacked_speeds]
        }

        save_dir = directory + "/Results"
        os.makedirs(save_dir, exist_ok=True)

        for key in plots:
            FreeGraphPlotter.plot_filled_graph(plots[key][0][1], steps, plots[key][1], parent_dir_name)
            plots[key][0][1].set_xlabel('Frames')
            plots[key][0][1].set_ylabel(key)
            plots[key][0][1].set_title("\n(Rolling Average with window size = " + str(r_avg_window_size) + ")")

            plots[key][0][1].legend(fontsize='8')
            figure_save_dir = save_dir + "/" + key.replace(" ", "_") + "_window=" + str(
                r_avg_window_size) + "_runs=" + str(file_count)
            plots[key][0][0].savefig(figure_save_dir)

    # Takes in:
    # - parent_directory
    #   - run_type_directory_1
    #       - rewards_1.txt
    #       - rewards_2.txt
    #   - run_type_directory_2
    #       - rewards_1.txt
    #       - rewards_2.txt
    @staticmethod
    def plot_multiple_average_graphs(parent_directory, r_avg_window_size=100):
        line_colours = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'cyan', 'gray', 'yellow', 'magenta']
        colour_index = 0
        rewards_fig, rewards_axis = plt.subplots()
        speeds_fig, speeds_axis = plt.subplots()
        for run_directory in os.listdir(parent_directory):
            current_directory = os.path.join(parent_directory, run_directory)
            steps, stacked_rewards, stacked_speeds = FreeGraphPlotter.plot_average(
                current_directory,
                r_avg_window_size=r_avg_window_size,
                plot_graphs=False
            )
            if steps.size == 0:
                print(current_directory, " does not contain any rewards.txt files")
                continue
            FreeGraphPlotter.plot_filled_graph(rewards_axis, steps, stacked_rewards, run_directory,
                                               line_colours[colour_index])
            FreeGraphPlotter.plot_filled_graph(speeds_axis, steps, stacked_speeds, run_directory,
                                               line_colours[colour_index])
            colour_index += 1

        save_dir = parent_directory + "/AveragesResults/"
        os.makedirs(save_dir, exist_ok=True)

        rewards_axis.legend(fontsize='8')
        rewards_axis.set_title(parent_directory.rstrip("/").split("/")[-1])
        rewards_fig.savefig(save_dir + "Returns Rolling Average (window_size=" + str(r_avg_window_size) + ")")

        speeds_axis.legend(fontsize='8')
        speeds_axis.set_title(parent_directory.rstrip("/").split("/")[-1])
        speeds_fig.savefig(save_dir + "Speeds Rolling Average (window_size=" + str(r_avg_window_size) + ")")

    @staticmethod
    def plot_multiple_individual_graphs(directory, r_avg_window_size=100):
        rolling_avg_rewards_fig, rolling_avg_rewards_axis = plt.subplots()
        rolling_avg_speed_fig, rolling_avg_speed_axis = plt.subplots()

        for item in os.listdir(directory):
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):
                if "rewards" not in item and "returns" not in item:
                    continue
                steps, optimal_policy_rewards, optimal_policy_speeds = np.loadtxt(file_path, delimiter=',')

                rolling_avg_steps = steps[r_avg_window_size - 1:]

                rolling_avg_rewards = np.convolve(optimal_policy_rewards,
                                                  np.ones(r_avg_window_size) / r_avg_window_size,
                                                  mode='valid')
                rolling_avg_speeds = np.convolve(optimal_policy_rewards,
                                                 np.ones(r_avg_window_size) / r_avg_window_size,
                                                 mode='valid')

                rolling_avg_rewards_axis.plot(rolling_avg_steps * 15, rolling_avg_rewards, label=item, linewidth=1.0)

        save_dir = directory + "/Results"
        os.makedirs(save_dir, exist_ok=True)

        rolling_avg_rewards_axis.set_xlabel('Frames')
        rolling_avg_rewards_axis.set_ylabel('Rolling Average Reward')
        rolling_avg_rewards_axis.set_title(
            "\n(Individual Rolling Averages with window size = " + str(r_avg_window_size) + ")")

        rolling_avg_rewards_axis.legend(fontsize='8')

        rolling_avg_rewards_fig.savefig(save_dir + "/individual_rolling_averages_" + str(r_avg_window_size))

    @staticmethod
    def download_txt_files(directory_to_search, save_directory, file_to_search_for="returns.txt"):
        download_count = 0

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        for root, _, files in os.walk(directory_to_search):
            for file in files:
                if file == file_to_search_for or file == "rewards.txt":
                    file_path = os.path.join(root, file)
                    FreeGraphPlotter.download_file(file_path, "returns_" + str(download_count) + ".txt", save_directory)
                    download_count += 1
                elif file == "config.txt":
                    file_path = os.path.join(root, file)
                    FreeGraphPlotter.download_file(file_path, "config.txt", save_directory)

        for parent_dir in os.listdir(directory_to_search):
            parent_dir_path = os.path.join(directory_to_search, parent_dir)
            if os.path.isdir(parent_dir_path):
                shutil.rmtree(parent_dir_path)

    @staticmethod
    def download_file(file_path, save_file_as, save_directory):
        try:
            destination_path = os.path.join(save_directory, save_file_as)
            shutil.copyfile(file_path, destination_path)
            print(file_path, " downloaded successfully")
        except Exception as e:
            print("Error downloading ", file_path, "\n", e)

# TODO: Label the axis on the plots
if __name__ == '__main__':
    plots_dir = "C:/Users/shiva/OneDrive/Desktop/University/Fourth Year/Dissertation/Code/Dissertation/Plots/IPPO/"

    download_from = plots_dir + "4 Agents 9 Obstructions/Standard"
    download_to = plots_dir + "4 Agents 9 Obstructions/Standard"

    # FreeGraphPlotter.download_txt_files(download_from, download_to)
    #
    average_dir = plots_dir + "4 Agents 9 Obstructions"
    FreeGraphPlotter.plot_multiple_average_graphs(average_dir)

    # FreeGraphPlotter.plot_multiple_individual_graphs("C:/Users/shiva/OneDrive/Desktop/University/Fourth Year/Dissertation/Code/Dissertation/Random", r_avg_window_size=1500)
    # FreeGraphPlotter.plot_average("TestDownloads")

