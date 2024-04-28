import os
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import distinctipy

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 14}

plt.rc('font', **font)

dpi = 300
smooth_standard_errors = True

class FreeGraphPlotter:

    @staticmethod
    def save_stats_df_to_csv(directory, df, r_avg_window_size=100):
        max_returns = np.empty(0)
        max_return_steps = np.empty(0)
        max_speeds = np.empty(0)
        max_speed_steps = np.empty(0)
        max_successes = np.empty(0)
        max_success_rates = np.empty(0)
        for item in os.listdir(directory):
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):
                if "rewards" not in item and "returns" not in item:
                    continue
                all_steps, optimal_policy_returns, optimal_policy_speeds, optimal_policy_ends_reached = np.loadtxt(
                    file_path, delimiter=',')

                steps = all_steps[r_avg_window_size - 1:]

                rolling_avg_returns = np.convolve(optimal_policy_returns,
                                                  np.ones(r_avg_window_size) / r_avg_window_size,
                                                  mode='valid')
                rolling_avg_speeds = np.convolve(optimal_policy_speeds,
                                                 np.ones(r_avg_window_size) / r_avg_window_size,
                                                 mode='valid')

                max_returns = np.append(max_returns, np.max(rolling_avg_returns))
                max_return_steps = np.append(max_return_steps, steps[np.argmax(rolling_avg_returns)])
                max_speeds = np.append(max_speeds, np.max(rolling_avg_speeds))
                max_speed_steps = np.append(max_speed_steps, steps[np.argmax(rolling_avg_speeds)])
                max_successes = np.append(max_successes, optimal_policy_ends_reached[-1])
                max_success_rates = np.append(max_success_rates, optimal_policy_ends_reached[-1] / len(all_steps))

        row_data = np.round(
            np.array(
                [np.mean(max_returns), np.floor(np.mean(max_return_steps)), np.mean(max_speeds),
                 np.mean(max_speed_steps),
                 np.floor(np.mean(max_successes)), np.mean(max_success_rates)]
            ),
            decimals=2
        )
        run_name = directory.split('\\')[-1]
        df.loc[run_name] = row_data

        return df

    @staticmethod
    def plot_filled_graph(plot_axis, steps, values, label, colour='blue', allow_negatives=True):
        mean_rewards = np.mean(values, axis=0)

        standard_error = 1.0 * (np.std(values) / np.sqrt(2))
        min_rewards = mean_rewards - standard_error
        max_rewards = mean_rewards + standard_error
        if not allow_negatives:
            min_rewards = np.maximum(min_rewards, 0)

        if smooth_standard_errors:
            min_rewards = savgol_filter(min_rewards, 7, 2)
            max_rewards = savgol_filter(max_rewards, 7, 2)
        plot_axis.plot(steps * 15, mean_rewards, marker='o', markersize=1, label=label, linewidth=1.2, color=colour)
        plot_axis.fill_between(steps * 15, min_rewards, max_rewards, color=colour, alpha=0.15)
        plot_axis.grid(True)

    # Given a directory containing n number of runs (rewards.txt) of the same type, plot the average and min/max range
    @staticmethod
    def plot_average(directory, r_avg_window_size=100, plot_graphs=True):
        parent_dir_name = directory.rstrip("/").split("/")[-1]

        stacked_rewards = np.empty(0)
        stacked_speeds = np.empty(0)
        stacked_ends_reached = np.empty(0)

        all_steps = np.empty(0)
        steps = np.empty(0)

        file_count = 0
        for item in os.listdir(directory):
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):
                if "rewards" not in item and "returns" not in item:
                    continue
                all_steps, optimal_policy_rewards, optimal_policy_speeds, optimal_policy_ends_reached = np.loadtxt(
                    file_path, delimiter=',')

                steps = all_steps[r_avg_window_size - 1:]

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
                stacked_ends_reached = optimal_policy_ends_reached if stacked_ends_reached.size == 0 else np.vstack(
                    (stacked_ends_reached, optimal_policy_ends_reached))

                file_count += 1

        if not plot_graphs:
            return all_steps, steps, stacked_rewards, stacked_speeds, stacked_ends_reached

        # plots[key][0] = [plt_fig, plt_axis] --> eg: plots[key][0][0] = plt_fig
        # plots[key][1] = the values (rewards, speeds, etc)
        plots = {
            "Rolling Average Return": [plt.subplots(), stacked_rewards],
            "Rolling Average Speed": [plt.subplots(), stacked_speeds],
            "Ends Reached": [plt.subplots(), stacked_ends_reached]
        }

        save_dir = directory + "/Results"
        os.makedirs(save_dir, exist_ok=True)

        # TODO: Graph names only correct for rolling avg graphs!
        for key in plots:
            FreeGraphPlotter.plot_filled_graph(plots[key][0][1], steps, plots[key][1], parent_dir_name)
            plots[key][0][1].set_xlabel('Frames')
            plots[key][0][1].set_ylabel(key)
            plots[key][0][1].set_title("\n(Rolling Average with window size = " + str(r_avg_window_size) + ")")

            plots[key][0][1].legend(fontsize='8')
            figure_save_dir = save_dir + "/" + key.replace(" ", "_") + "_window=" + str(
                r_avg_window_size) + "_runs=" + str(file_count)
            plots[key][0][0].savefig(figure_save_dir, dpi=dpi)

    # Takes in:
    # - parent_directory
    #   - run_type_directory_1
    #       - rewards_1.txt
    #       - rewards_2.txt
    #   - run_type_directory_2
    #       - rewards_1.txt
    #       - rewards_2.txt
    @staticmethod
    def plot_multiple_average_graphs(parent_directory, r_avg_window_size=100, title=None):
        plt.rc('font', **font)
        num_folders_in_parent_dir = len(os.listdir(parent_directory))
        # line_colours = ['blue', 'red', 'green', 'magenta', 'cyan', 'black', 'purple', 'gray', 'yellow', 'orange']
        line_colours = distinctipy.get_colors(num_folders_in_parent_dir, rng=22, colorblind_type="Deuteranomaly")
        colour_index = 0
        columns = ['Return', 'Step', 'Speed (mph)', 'Step',
                   'Count', 'Rate (0 - 1)']
        df = pd.DataFrame(columns=columns)
        rewards_fig, rewards_axis = plt.subplots()
        speeds_fig, speeds_axis = plt.subplots()
        end_reached_fig, ends_reached_axis = plt.subplots()
        rewards_axis.margins(0, 0)
        speeds_axis.margins(0, 0)
        ends_reached_axis.margins(0, 0)

        combined_fig, combined_axis = plt.subplots(1, 3, figsize=(17, 5))
        combined_axis[0].margins(0, 0)
        combined_axis[1].margins(0, 0)
        combined_axis[2].margins(0, 0)
        for run_directory in os.listdir(parent_directory):
            current_directory = os.path.join(parent_directory, run_directory)
            if not os.path.isdir(current_directory):
                continue
            all_steps, steps, stacked_rewards, stacked_speeds, stacked_ends_reached = FreeGraphPlotter.plot_average(
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
            FreeGraphPlotter.plot_filled_graph(ends_reached_axis, all_steps, stacked_ends_reached, run_directory,
                                               line_colours[colour_index], allow_negatives=False)

            FreeGraphPlotter.plot_filled_graph(combined_axis[0], steps, stacked_rewards, run_directory,
                                               line_colours[colour_index])
            FreeGraphPlotter.plot_filled_graph(combined_axis[1], steps, stacked_speeds, run_directory,
                                               line_colours[colour_index])
            FreeGraphPlotter.plot_filled_graph(combined_axis[2], all_steps, stacked_ends_reached, run_directory,
                                               line_colours[colour_index], allow_negatives=False)
            colour_index += 1

            df = FreeGraphPlotter.save_stats_df_to_csv(current_directory, df, r_avg_window_size=r_avg_window_size)

        save_dir = parent_directory + "/AveragesResults/"
        os.makedirs(save_dir, exist_ok=True)
        legend_font_size = 10

        rewards_axis.set_xlabel('Frames')
        rewards_axis.set_ylabel('Rolling Average Return')
        rewards_axis.legend(fontsize=legend_font_size)
        rewards_axis.set_title(title)
        rewards_fig.savefig(save_dir + "Returns Rolling Average (window_size=" + str(r_avg_window_size) + ")", dpi=dpi)

        speeds_axis.set_xlabel('Frames')
        speeds_axis.set_ylabel('Rolling Average Speed')
        speeds_axis.legend(fontsize=legend_font_size)
        speeds_axis.set_title(title)
        speeds_fig.savefig(save_dir + "Speeds Rolling Average (window_size=" + str(r_avg_window_size) + ")", dpi=dpi)

        ends_reached_axis.set_xlabel('Frames')
        ends_reached_axis.set_ylabel('Ends Reached')
        ends_reached_axis.legend(fontsize=legend_font_size)
        ends_reached_axis.set_title(title)
        end_reached_fig.savefig(save_dir + "Ends Reached", dpi=dpi)

        table_save_dir = save_dir + "/Table"
        os.makedirs(table_save_dir, exist_ok=True)


        df.to_csv(table_save_dir + "/dataframe.csv")
        latex_table_format = {
            'column_format': 'ccc|cc|cc',
            'bold_rows': True,
            'float_format': '%.2f',
            'escape': False
        }
        latex_table = df.to_latex(**latex_table_format)

        custom_latex_code = ('\\toprule\n'
                             '& \multicolumn{2}{c:}{\\textbf{Maximum Return}}\n'
                             '& \multicolumn{2}{c:}{\\textbf{Maximum Speed}}\n'
                             '& \multicolumn{2}{c}{\\textbf{Successfully Clear}}\n'
                             '\\\\ \\midrule\n')
        latex_table = latex_table.replace('\\toprule\n', custom_latex_code, 1)
        latex_table = latex_table.replace('ccc|cc|cc', 'ccc:cc:cc', 1)
        latex_table = "\\begin{table}[H]\n\\centering\n" + latex_table + "\\caption{caption}\n\\end{table}"

        with open(table_save_dir + "/" + parent_directory.split('/')[-1] + "_table.tex", 'w') as f:
            f.write(latex_table)

        combined_axis[0].set_xlabel('Frames')
        combined_axis[0].set_ylabel('Rolling Average Return')
        combined_axis[0].set_title('Team Return')

        combined_axis[1].set_xlabel('Frames')
        combined_axis[1].set_ylabel('Rolling Average Speed')
        combined_axis[1].set_title('Team Average Speed')

        combined_axis[2].set_xlabel('Frames')
        combined_axis[2].set_ylabel('Ends Reached')
        combined_axis[2].set_title('Obstructions Cleared Frequency')

        handles, labels = [], []
        for h, l in zip(*combined_axis[0].get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)

        n_cols = 8
        # Create a single legend
        combined_fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), ncol=n_cols, fontsize=16)

        combined_fig.tight_layout(w_pad=1.5)

        combined_fig.subplots_adjust(left=0.04, bottom=0.22, top=0.9)
        combined_fig.savefig(save_dir + '/all_plots_in_one.png', dpi=dpi)

    @staticmethod
    def plot_multiple_individual_graphs(directory, r_avg_window_size=100):
        rolling_avg_rewards_fig, rolling_avg_rewards_axis = plt.subplots()
        rolling_avg_speed_fig, rolling_avg_speed_axis = plt.subplots()

        for item in os.listdir(directory):
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):
                if "rewards" not in item and "returns" not in item:
                    continue
                steps, optimal_policy_rewards, optimal_policy_speeds, optimal_policy_ends_reached = (
                    np.loadtxt(file_path, delimiter=','))

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
        rolling_avg_rewards_axis.set_ylabel('Rolling Average Return')
        rolling_avg_rewards_axis.set_title(
            "\n(Individual Rolling Averages with window size = " + str(r_avg_window_size) + ")")

        rolling_avg_rewards_axis.legend(fontsize='8')

        rolling_avg_rewards_fig.savefig(save_dir + "/individual_rolling_averages_" + str(r_avg_window_size), dpi=dpi)

    @staticmethod
    def download_txt_files(directory_to_search, save_directory, file_to_search_for="returns.txt", multiple=False):
        if not multiple:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            FreeGraphPlotter.iterate_and_download(directory_to_search, save_directory, file_to_search_for)

        else:
            for main_dir in os.listdir(directory_to_search):
                main_dir = os.path.join(directory_to_search, main_dir)
                FreeGraphPlotter.iterate_and_download(main_dir, main_dir, file_to_search_for)

    @staticmethod
    def iterate_and_download(directory_to_search, save_directory, file_to_search_for):
        download_count = 0
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

        return download_count

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
    local_path = "../../"
    plots_dir = local_path + "ForTheReport/Experiments/Experiment_4/2,9,3/MAPPO"

    # download_from = plots_dir + "4 Agents 9 Obstructions/Standard"
    # download_to = plots_dir + "4 Agents 9 Obstructions/Standard"

    # average_dir = plots_dir + "gae_lambda"

    # FreeGraphPlotter.plot_multiple_individual_graphs(plots_dir)
    # FreeGraphPlotter.plot_average(plots_dir)

    FreeGraphPlotter.download_txt_files(plots_dir, plots_dir, multiple=True)
    FreeGraphPlotter.plot_multiple_average_graphs(plots_dir)
