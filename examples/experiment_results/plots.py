import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

def make_performance_plots_for_dset(dset_name: str) -> None:
    curr_dirname = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(curr_dirname, f"{dset_name}.yaml")
    results = None
    with open(fname, "r", encoding="utf8") as f:
        results = yaml.load(f, Loader=yaml.FullLoader)

    ticks_acc = np.array(["std", "r3", "ibp_ex", "ibp_ex+r3", "r4"])
    ticks_rob = ticks_acc[1:]
    test_accs = np.array([results[tick]["test_acc"] for tick in ticks_acc])
    robust_delta = np.array([results[tick]["min_robust_delta"] for tick in ticks_rob])
    lower_bounds_avg = np.array([results[tick]["min_lower_bound"] for tick in ticks_rob])
    upper_bounds_avg = np.array([results[tick]["max_upper_bound"] for tick in ticks_rob])

    _, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].bar(ticks_acc, test_accs, color="#0000FF")
    ax[0, 0].set_title("Test Accuracy")

    ax[0, 1].bar(ticks_rob, robust_delta, color="#00FF00")
    ax[0, 1].set_title("Delta for which test set is certifiably 1-delta-input-robust")

    ax[1, 0].bar(ticks_rob, lower_bounds_avg, color="#FF0000")
    ax[1, 0].set_title("Minimum lower bound (averaged over the number of runs)")

    ax[1, 1].bar(ticks_rob, upper_bounds_avg, color="#0000FF")
    ax[1, 1].set_title("Maximum upper bound (averaged over the number of runs)")

    plt.show()

def make_sample_complexity_plots_for_dset(dset_name: str) -> None:
    curr_dirname = os.path.dirname(os.path.realpath(__file__))
    perf_fname = os.path.join(curr_dirname, f"{dset_name}.yaml")
    ablation_fname = os.path.join(curr_dirname, f"{dset_name}_sample_complexity.yaml")
    x_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    results_for_methods, results_perf, results_abl = {}, None, None
    all_methods = ["r3", "ibp_ex", "ibp_ex+r3", "r4"]
    with open(perf_fname, "r", encoding="utf8") as f:
        results_perf = yaml.load(f, Loader=yaml.FullLoader)
    with open(ablation_fname, "r", encoding="utf8") as f:
        results_abl = yaml.load(f, Loader=yaml.FullLoader)
    for method_ratio in results_abl.keys():
        method, ratio = method_ratio.rsplit("_", 1)
        ratio = int(ratio) / 100
        if method not in results_for_methods:
            results_for_methods[method] = {}
        results_for_methods[method][ratio] = results_abl[method_ratio]

    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    for method in all_methods:
        test_acc_abl, delta_abl, lower_bound_abl, upper_bound_abl = [], [], [], []
        test_acc_abl.append(results_perf["std"]["test_acc"]) # i.e. for 0% mask ratio
        delta_abl.append(results_perf["std"]["min_robust_delta"])
        lower_bound_abl.append(results_perf["std"]["min_lower_bound"])
        upper_bound_abl.append(results_perf["std"]["max_upper_bound"])
        for ratio in x_ticks[1:-1]:
            test_acc_abl.append(results_for_methods[method][ratio]["test_acc"])
            delta_abl.append(results_for_methods[method][ratio]["min_robust_delta"])
            lower_bound_abl.append(results_for_methods[method][ratio]["min_lower_bound"])
            upper_bound_abl.append(results_for_methods[method][ratio]["max_upper_bound"])
        test_acc_abl.append(results_perf[method]["test_acc"])
        delta_abl.append(results_perf[method]["min_robust_delta"])
        lower_bound_abl.append(results_perf[method]["min_lower_bound"])
        upper_bound_abl.append(results_perf[method]["max_upper_bound"])
        test_acc_abl, delta_abl = np.array(test_acc_abl), np.array(delta_abl)
        lower_bound_abl, upper_bound_abl = np.array(lower_bound_abl), np.array(upper_bound_abl)

        ax[0, 0].plot(x_ticks, test_acc_abl, label=method)
        ax[0, 0].xaxis.set_inverted(True)
        ax[0, 0].set_title("Test Accuracy for different mask ratios")
        ax[0, 0].legend()


        ax[0, 1].plot(x_ticks[1:], delta_abl[1:], label=method)
        ax[0, 1].xaxis.set_inverted(True)
        ax[0, 1].set_title("Delta for which test set is certifiably 1-delta-input-robust for different mask ratios")
        ax[0, 1].legend()

        ax[1, 0].plot(x_ticks[1:], lower_bound_abl[1:], label=method)
        ax[1, 0].xaxis.set_inverted(True)
        ax[1, 0].set_title("Minimum lower bound (averaged over the number of runs) for different mask ratios")
        ax[1, 0].legend()

        ax[1, 1].plot(x_ticks[1:], upper_bound_abl[1:], label=method)
        ax[1, 1].xaxis.set_inverted(True)
        ax[1, 1].set_title("Maximum upper bound (averaged over the number of runs) for different mask ratios")
        ax[1, 1].legend()

    fig.tight_layout()

    plt.show()

def make_size_ablation_plots_for_medmnist() -> None:
    curr_dirname = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(curr_dirname, "derma_size.yaml")
    results = None
    with open(fname, "r", encoding="utf8") as f:
        results = yaml.load(f, Loader=yaml.FullLoader)

    ticks = np.array(list(results.keys()))
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    test_accs = np.array([results[tick]["test_acc"] for tick in ticks])
    robust_delta = np.array([results[tick]["min_robust_delta"] for tick in ticks])
    lower_bounds_avg = np.array([results[tick]["min_lower_bound"] for tick in ticks])
    upper_bounds_avg = np.array([results[tick]["max_upper_bound"] for tick in ticks])
    ticks = np.array([f"img size {str(tick)}" for tick in ticks])

    ax[0, 0].bar(ticks, test_accs, color="#0000FF", width=0.5)
    ax[0, 0].set_title("Test Accuracy")

    ax[0, 1].bar(ticks, robust_delta, color="#00FF00", width=0.5)
    ax[0, 1].set_title("Delta for which test set is certifiably 1-delta-input-robust")

    ax[1, 0].bar(ticks, lower_bounds_avg, color="#FF0000", width=0.5)
    ax[1, 0].set_title("Minimum lower bound (averaged over the number of runs)")

    ax[1, 1].bar(ticks, upper_bounds_avg, color="#0000FF", width=0.5)
    ax[1, 1].set_title("Maximum upper bound (averaged over the number of runs)")


    plt.show()
