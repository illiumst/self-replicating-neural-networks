""" ----------------------------------------- Methods for summarizing the experiments ------------------------------------------ """
import os
from pathlib import Path

from visualization import line_chart_fixpoints, bar_chart_fixpoints


def summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, directory,
                                summary_pre_title):
    avg_fixpoint_counters = {
        "avg_identity_func": 0,
        "avg_divergent": 0,
        "avg_fix_zero": 0,
        "avg_fix_weak": 0,
        "avg_fix_sec": 0,
        "avg_other_func": 0
    }

    for i in range(len(experiments)):
        fixpoint_counters = experiments[i].fixpoint_counters

        avg_fixpoint_counters["avg_identity_func"] += fixpoint_counters["identity_func"]
        avg_fixpoint_counters["avg_divergent"] += fixpoint_counters["divergent"]
        avg_fixpoint_counters["avg_fix_zero"] += fixpoint_counters["fix_zero"]
        avg_fixpoint_counters["avg_fix_weak"] += fixpoint_counters["fix_weak"]
        avg_fixpoint_counters["avg_fix_sec"] += fixpoint_counters["fix_sec"]
        avg_fixpoint_counters["avg_other_func"] += fixpoint_counters["other_func"]

    # Calculating the average for each fixpoint
    avg_fixpoint_counters.update((x, y / len(experiments)) for x, y in avg_fixpoint_counters.items())

    # Checking where the data is coming from to have a relevant title in the plot.
    if summary_pre_title not in ["ST", "SA", "soup", "mixed", "robustness"]:
        summary_pre_title = ""

    # Plotting the summary
    source_checker = "summary"
    exp_details = f"{summary_pre_title}: {runs} runs & {epochs} epochs each."
    bar_chart_fixpoints(avg_fixpoint_counters, population_size, directory, net_learning_rate, exp_details,
                        source_checker)


def summary_fixpoint_percentage(runs, epochs, fixpoints_percentages, ST_steps, SA_steps, directory_name,
                                population_size):
    fixpoints_percentages = [round(fixpoints_percentages[i] / runs, 1) for i in range(len(fixpoints_percentages))]

    # Plotting summary
    if "soup" in directory_name:
        line_chart_fixpoints(fixpoints_percentages, epochs / ST_steps, ST_steps, SA_steps, directory_name,
                             population_size)
    else:
        line_chart_fixpoints(fixpoints_percentages, epochs, ST_steps, SA_steps, directory_name, population_size)


""" --------------------------------------------------- Miscellaneous ---------------------------------------------------------- """
def check_folder(experiment_folder: str):
    exp_path = Path('experiments') / experiment_folder
    exp_path.mkdir(parents=True, exist_ok=True)
