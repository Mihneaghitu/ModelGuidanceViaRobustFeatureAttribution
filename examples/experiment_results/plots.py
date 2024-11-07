import sys
import os
sys.path.append("../")
sys.path.append("../../")
import matplotlib.pyplot as plt
import yaml
import numpy as np
import seaborn as sns
from examples.datasets import derma_mnist, decoy_mnist
from examples.models.R4_models import DermaNet
from examples.metrics import get_avg_acc_with_stddev, get_avg_rob_metrics
from examples.models.fully_connected import FCNAugmented
import pandas as pd


def make_mask_ablation_paper_plots(dset_name: str, with_data: bool = False) -> None:
    assert dset_name in ["decoy_mnist", "derma_mnist"]
    methods = ["r3", "r4", "ibp_ex", "ibp_ex+r3"]
    suffix = "data_and_mask" if with_data else "mask"
    dl_test, model_dir, model, eps, has_conv, num_classes, loss_fn = None, None, None, None, None, None, None
    if dset_name == "decoy_mnist":
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        loss_fn = "cross_entropy"
        eps = 0.1
        model_dir = f"saved_experiment_models/ablations/{suffix}/decoy_mnist"
        has_conv = False
        num_classes = 10
    if dset_name == "derma_mnist":
        test_derma = derma_mnist.DecoyDermaMNIST(False, size=64)
        dl_test = derma_mnist.get_dataloader(test_derma, 256)
        model = DermaNet(3, 64, 1)
        loss_fn = "binary_cross_entropy"
        eps = 0.05
        model_dir = f"saved_experiment_models/ablations/{suffix}/derma_mnist"
        has_conv = True
        num_classes = 2

    sns.set_theme(context="poster", font_scale=2)
    sns.color_palette("bright")
    ratios = np.array([1, 0.8, 0.6, 0.4, 0.2, 0])
    fig, ax = plt.subplots(2, 2, figsize=(70, 60))
    for method in methods:
        mean_accs, std_dev_accs = [], [],
        mean_delta, mean_lb, mean_ub, std_dev_delta, std_dev_lb, std_dev_ub = [], [], [], [], [], []
        for ratio in ratios:
            dir_for_method = model_dir + f"/{method}" + f"/ratio_{int(ratio * 100)}"
            avg_acc_ratio, std_dev_acc_ratio = get_avg_acc_with_stddev(model, dl_test, "cuda:0", dir_for_method, num_classes)
            mean_accs.append(avg_acc_ratio)
            std_dev_accs.append(std_dev_acc_ratio)
            avg_delta_ratio, avg_lb_ratio, avg_ub_ratio, std_delta_ratio, std_lb_ratio, std_ub_ratio = get_avg_rob_metrics(
                model, dl_test, "cuda:0", dir_for_method, eps, loss_fn, has_conv)
            mean_delta.append(avg_delta_ratio)
            mean_lb.append(avg_lb_ratio)
            mean_ub.append(avg_ub_ratio)
            std_dev_delta.append(std_delta_ratio)
            std_dev_lb.append(std_lb_ratio)
            std_dev_ub.append(std_ub_ratio)

        sns.lineplot(x=ratios, y=mean_accs, label=f"{method.upper()}", marker="o", legend="full", ax=ax[0][0], linewidth=10, estimator=None)
        sns.lineplot(x=ratios, y=mean_delta, label=f"{method.upper()}", marker="o", legend="full", ax=ax[0][1], linewidth=10, estimator=None)
        sns.lineplot(x=ratios, y=mean_lb, label=f"{method.upper()}", marker="o", legend="full", ax=ax[1][0], linewidth=10, estimator=None)
        sns.lineplot(x=ratios, y=mean_ub, label=f"{method.upper()}", marker="o", legend="full", ax=ax[1][1], linewidth=10, estimator=None)
        ax[0][0].fill_between(ratios, np.array(mean_accs) - np.array(std_dev_accs), np.array(mean_accs) + np.array(std_dev_accs), alpha=0.25)
        ax[0][0].xaxis.set_inverted(True)
        ax[0][0].set(xlabel="Mask Ratio\n", ylabel="Average Test Accuracy")
        ax[0][1].fill_between(ratios, np.array(mean_delta) - np.array(std_dev_delta), np.array(mean_delta) + np.array(std_dev_delta), alpha=0.25)
        ax[0][1].xaxis.set_inverted(True)
        ax[0][1].set(xlabel="Mask Ratio\n", ylabel=r'Average $\delta$')
        ax[0][1].set_yscale("symlog")
        ax[1][0].fill_between(ratios, np.array(mean_lb) - np.array(std_dev_lb), np.array(mean_lb) + np.array(std_dev_lb), alpha=0.25)
        ax[1][0].xaxis.set_inverted(True)
        ax[1][0].set_yscale("symlog")
        ax[1][0].set(xlabel="Mask Ratio\n", ylabel="Average Lower Bound")
        ax[1][1].fill_between(ratios, np.array(mean_ub) - np.array(std_dev_ub), np.array(mean_ub) + np.array(std_dev_ub), alpha=0.25)
        ax[1][1].xaxis.set_inverted(True)
        ax[1][1].set_yscale("symlog")
        ax[1][1].set(xlabel="Mask Ratio\n", ylabel="Average Upper Bound")

    title_suffix = "and data" if with_data else ""
    ax[0][0].set_title(f"Average test accuracy for {dset_name.upper()} upon varying \n the ratio of mask {title_suffix}", weight="bold")
    ax[0][1].set_title(r'Average $\delta$ (1-$\delta$-input-robustness) for ' + f"{dset_name.upper()} \n \
                       upon varying the ratio of mask {title_suffix}", weight="bold")
    ax[1][0].set_title(f"Average lower bound for {dset_name.upper()} upon \n varying the ratio of mask {title_suffix}", weight="bold")
    ax[1][1].set_title(f"Average upper bound for {dset_name.upper()} upon \n varying the ratio of mask {title_suffix}", weight="bold")

    plt.show()


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

    ticks_acc = np.array(["STD", "R3", "IBP_EX", "IBP_EX+R3", "R4"])
    ticks_rob = ticks_acc[1:]
    _, ax = plt.subplots(2, 2, figsize=(17, 13))
    color_gradient_accs = ["#ff7ca0", "#ffa7b4", "#ffcbd2", "#ffb979", "#ffa463"]
    color_gradient_delta = ["#fae442", "#ddf969", "#a9f36a", "#57e86b"]
    color_gradient_lb = ["#d92122", "#b8203d", "#971e58",  "#761d72"]
    color_gradient_ub = ["#0000ff", "#4d4dff", "#7a7aff", "#bcbcff"]
    # make the color a gradient with specified hex color
    ax[0, 0].bar(ticks_acc, test_accs, color=color_gradient_accs)
    ax[0, 0].set_ylim([min(test_accs) - 0.1, max(test_accs) + 0.1])
    ax[0, 0].set_title("Test Accuracy")

    ax[0, 1].bar(ticks_rob, robust_delta, color=color_gradient_delta)
    ax[0, 1].set_yscale("symlog")
    ax[0, 1].set_title("Delta for which test set is certifiably 1-delta-input-robust")

    ax[1, 0].bar(ticks_rob, lower_bounds_avg, color=color_gradient_lb)
    ax[1, 0].set_yscale("symlog")
    ax[1, 0].set_title("Minimum lower bound (averaged over the number of runs)")

    ax[1, 1].bar(ticks_rob, upper_bounds_avg, color=color_gradient_ub)
    ax[1, 1].set_yscale("symlog")
    ax[1, 1].set_title("Maximum upper bound (averaged over the number of runs)")

    plt.show()


def make_model_ablation_paper_plots(dset_name: str) -> None:
    assert dset_name in ["decoy_mnist", "derma_mnist"]
    methods = ["r3", "r4", "ibp_ex", "ibp_ex+r3"]
    dl_test, model_dir, size_names, eps, has_conv, num_classes, loss_fn, model_archs = None, None, None, None, None, None, None, None
    if dset_name == "decoy_mnist":
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model_archs = [(784, 10, 512, 1), (784, 10, 512, 2), (784, 10, 512, 3), (784, 10, 512, 4)]
        loss_fn = "cross_entropy"
        eps = 0.1
        model_dir = "saved_experiment_models/ablations/size/decoy_mnist"
        has_conv = False
        num_classes = 10
        size_names = ["1_layer", "2_layer", "3_layer", "4_layer"]
    if dset_name == "derma_mnist":
        test_derma = derma_mnist.DecoyDermaMNIST(False, size=64)
        dl_test = derma_mnist.get_dataloader(test_derma, 256)
        model_archs = [(3, 64, 1, "small"), (3, 64, 1, "small_medium"), (3, 64, 1), (3, 64, 1, "large")]
        loss_fn = "binary_cross_entropy"
        eps = 0.05
        model_dir = "saved_experiment_models/ablations/size/derma_mnist"
        has_conv = True
        num_classes = 2
        size_names = ["small", "small_medium", "medium_large", "large"]

    sns.set_theme(context="poster", font_scale=2)
    sns.color_palette("bright")
    fig, ax = plt.subplots(2, 2, figsize=(60, 40))
    for method in methods:
        mean_accs, std_dev_accs = [], [],
        mean_delta, mean_lb, mean_ub, std_dev_delta, std_dev_lb, std_dev_ub = [], [], [], [], [], []
        for sz_nm, arch in zip(size_names, model_archs):
            model = DermaNet(*arch) if dset_name == "derma_mnist" else FCNAugmented(*arch)
            dir_for_method = model_dir + f"/{method}" + f"/{sz_nm}"
            avg_acc_ratio, std_dev_acc_ratio = get_avg_acc_with_stddev(model, dl_test, "cuda:0", dir_for_method, num_classes)
            mean_accs.append(avg_acc_ratio)
            std_dev_accs.append(std_dev_acc_ratio)
            avg_delta_ratio, avg_lb_ratio, avg_ub_ratio, std_delta_ratio, std_lb_ratio, std_ub_ratio = get_avg_rob_metrics(
                model, dl_test, "cuda:0", dir_for_method, eps, loss_fn, has_conv)
            mean_delta.append(avg_delta_ratio)
            mean_lb.append(avg_lb_ratio)
            mean_ub.append(avg_ub_ratio)
            std_dev_delta.append(std_delta_ratio)
            std_dev_lb.append(std_lb_ratio)
            std_dev_ub.append(std_ub_ratio)

        sns.lineplot(x=size_names, y=mean_accs, label=f"{method.upper()}", marker="o", legend="full", ax=ax[0][0], linewidth=10, estimator=None)
        sns.lineplot(x=size_names, y=mean_delta, label=f"{method.upper()}", marker="o", legend="full", ax=ax[0][1], linewidth=10, estimator=None)
        sns.lineplot(x=size_names, y=mean_lb, label=f"{method.upper()}", marker="o", legend="full", ax=ax[1][0], linewidth=10, estimator=None)
        sns.lineplot(x=size_names, y=mean_ub, label=f"{method.upper()}", marker="o", legend="full", ax=ax[1][1], linewidth=10, estimator=None)
        ax[0][0].fill_between(size_names, np.array(mean_accs) - np.array(std_dev_accs), np.array(mean_accs) + np.array(std_dev_accs), alpha=0.35)
        ax[0][0].set(xlabel="Model Size\n", ylabel="Average Test Accuracy")
        ax[0][1].fill_between(size_names, np.array(mean_delta) - np.array(std_dev_delta), np.array(mean_delta) + np.array(std_dev_delta), alpha=0.35)
        ax[0][0].set(xlabel="Model Size\n", ylabel=r'Average $\delta$')
        ax[0][1].set_yscale("symlog")
        ax[1][0].fill_between(size_names, np.array(mean_lb) - np.array(std_dev_lb), np.array(mean_lb) + np.array(std_dev_lb), alpha=0.35)
        ax[1][0].set(xlabel="Model Size\n", ylabel="Average Lower Bound")
        ax[1][0].set_yscale("symlog")
        ax[1][1].fill_between(size_names, np.array(mean_ub) - np.array(std_dev_ub), np.array(mean_ub) + np.array(std_dev_ub), alpha=0.35)
        ax[1][1].set(xlabel="Model Size\n", ylabel="Average Upper Bound")
        ax[1][1].set_yscale("symlog")

    ax[0][0].set_title(f"Test Accuracy {dset_name.upper()} upon varying \n the model size", weight="bold")
    ax[0][1].set_title(r'Average $\delta$ (1-$\delta$-input-robustness) for ' + \
                          f"{dset_name.upper()} \n upon varying the model size", weight="bold")
    ax[1][0].set_title(f"Average Lower bound for {dset_name.upper()} upon \n varying the model size", weight="bold")
    ax[1][1].set_title(f"Average upper bound for {dset_name.upper()} upon \n varying the model size", weight="bold")

    plt.show()

def make_sample_complexity_plots_for_dset(dset_name: str, with_data_removal: bool = False) -> None:
    curr_dirname = os.path.dirname(os.path.realpath(__file__))
    perf_fname = os.path.join(curr_dirname, f"{dset_name}.yaml")
    suffix =  "" if not with_data_removal else "_data_removal"
    ablation_fname = os.path.join(curr_dirname, f"{dset_name}_sample_complexity{suffix}.yaml")
    x_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    results_for_methods, results_perf, results_abl = {}, None, None
    all_methods = ["r3", "ibp_ex", "ibp_ex+r3", "r4"]
    with open(perf_fname, "r", encoding="utf8") as f:
        results_perf = yaml.load(f, Loader=yaml.FullLoader)
    with open(ablation_fname, "r", encoding="utf8") as f:
        results_abl = yaml.load(f, Loader=yaml.FullLoader)
    t = set()
    for k in results_abl.keys():
        if k.startswith("r3"):
            t.add(k.rsplit("_", 1)[1])
    x_ticks = np.array(sorted([int(i) / 100 for i in t]))
    x_ticks = np.insert(x_ticks, 0, 0)
    x_ticks = np.append(x_ticks, 1)
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

    ax[0, 0].bar(ticks, test_accs, color="#0000FF", width=0.75)
    ax[0, 0].set_title("Test Accuracy")

    ax[0, 1].bar(ticks, robust_delta, color="#00FF00", width=0.75)
    ax[0, 1].set_title("Delta for which test set is certifiably 1-delta-input-robust")

    ax[1, 0].bar(ticks, lower_bounds_avg, color="#FF0000", width=0.75)
    ax[1, 0].set_title("Minimum lower bound (averaged over the number of runs)")

    ax[1, 1].bar(ticks, upper_bounds_avg, color="#0000FF", width=0.75)
    ax[1, 1].set_title("Maximum upper bound (averaged over the number of runs)")


    plt.show()
