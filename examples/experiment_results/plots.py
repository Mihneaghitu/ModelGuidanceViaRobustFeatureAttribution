import sys
from copy import deepcopy
sys.path.append("../")
sys.path.append("../../")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from examples.datasets import derma_mnist, decoy_mnist
from examples.models.R4_models import DermaNet
from examples.metrics import get_restart_avg_and_worst_group_accuracy_with_stddev, get_avg_rob_metrics
from examples.models.pipeline import load_params_or_results_from_file
from examples.models.fully_connected import FCNAugmented

def make_mask_and_data_sample_complexity_plots(dset_name: str, device: str, with_data: bool = False, with_l2_prop: bool = False,  methods: list[str] = None) -> None:
    assert dset_name in ["decoy_mnist", "derma_mnist"]
    if methods is None:
        methods = ["r3", "r4", "ibp_ex", "rand_r4", "pgd_r4"]
    dl_test, model_dir, model, eps, has_conv, loss_fn, num_groups = None, None, None, None, None, None, None
    result_file_suffix = "_data_removal" if with_data else ""
    result_file_suffix = result_file_suffix + "_propl2" if with_l2_prop else result_file_suffix
    result_yaml_file = f"experiment_results/{dset_name}_sample_complexity{result_file_suffix}.yaml"
    if dset_name == "decoy_mnist":
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        loss_fn = "cross_entropy"
        eps = 0.1
        model_dir = "saved_experiment_models/performance/decoy_mnist"
        has_conv = False
        num_groups = 10
    if dset_name == "derma_mnist":
        test_derma = derma_mnist.DecoyDermaMNIST(False, size=64)
        dl_test = derma_mnist.get_dataloader(test_derma, 256)
        model = DermaNet(3, 64, 1)
        loss_fn = "binary_cross_entropy"
        eps = 0.01
        model_dir = "saved_experiment_models/performance/derma_mnist"
        has_conv = True
        num_groups = 2
    dset_name = dset_name.replace("_", " ")

    sns.set_theme(context="poster", font_scale=3)
    sns.color_palette("bright")
    ratios = np.array([1, 0.8, 0.6, 0.4, 0.2, 0])
    if with_data:
        ratios = ratios[:-1] # the 0 data does not make any sense, it is basically a randomly initialized model
    fig, ax = plt.subplots(1, 2, figsize=(70, 28))
    min_wg, max_wg = 1, 0
    for method in methods:
        mean_wg_accs, stddev_wg_accs, mean_delta, std_dev_delta = [], [], [], []
        #* Measure acc and rob metrics for ratio 1
        _, wg_acc, _, _, stddev_wg_acc = get_restart_avg_and_worst_group_accuracy_with_stddev(dl_test, model_dir + f"/{method}", model, device, num_groups, multi_class=(num_groups > 2), suppress_log=True)
        delta_mean, _, _, delta_std, *_ = get_avg_rob_metrics(model, dl_test, device, model_dir + f"/{method}", eps, loss_fn, has_conv=has_conv)
        delta_mean, delta_std = round(float(delta_mean.item()), 5), round(float(delta_std.item()), 5)
        mean_wg_accs.append(wg_acc)
        stddev_wg_accs.append(stddev_wg_acc)
        mean_delta.append(delta_mean)
        std_dev_delta.append(delta_std)
        for ratio in ratios[1:]:
            abl_results_for_method_and_ratio = load_params_or_results_from_file(result_yaml_file, method + f"_{int(ratio * 100)}")
            mean_wg_accs.append(abl_results_for_method_and_ratio["worst_group_acc"])
            stddev_wg_accs.append(abl_results_for_method_and_ratio["stddev_worst_group_acc"])
            mean_delta.append(abl_results_for_method_and_ratio["delta_mean"])
            std_dev_delta.append(abl_results_for_method_and_ratio["delta_stddev"])

        min_wg, max_wg = min(min_wg, *mean_wg_accs), max(max_wg, *mean_wg_accs)
        xlabel_suffix = "and data" if with_data else ""
        sns.lineplot(x=ratios, y=mean_wg_accs, label=f"{method.upper()}", marker="o", legend="full", ax=ax[0], linewidth=10, estimator=None)
        sns.lineplot(x=ratios, y=mean_delta, label=f"{method.upper()}", marker="o", legend="full", ax=ax[1], linewidth=10, estimator=None)
        ax[0].fill_between(ratios, np.array(mean_wg_accs) - np.array(stddev_wg_accs), np.array(mean_wg_accs) + np.array(stddev_wg_accs), alpha=0.15)
        ax[0].xaxis.set_inverted(True)
        ax[0].set(xlabel=f"% of masks {xlabel_suffix}\n", ylabel="Average Worst Group Test Accuracy")
        ax[1].fill_between(ratios, np.array(mean_delta) - np.array(std_dev_delta), np.array(mean_delta) + np.array(std_dev_delta), alpha=0.15)
        ax[1].xaxis.set_inverted(True)
        ax[1].set(xlabel=f"% of masks {xlabel_suffix}\n", ylabel=r'Average $\delta$')
        ax[1].set_yscale("symlog")

    ax[0].set_ylim([min_wg * 0.95, max_wg * 1.05])
    title_suffix = "and data" if with_data else ""
    ax[0].set_title(f"Worst group test accuracy for {dset_name.upper()} upon \n varying the ratio of mask {title_suffix}", weight="bold")
    ax[1].set_title(r'Average $\delta$ (1-$\delta$-input-robustness) for ' + f"{dset_name.upper()} \n upon varying the ratio of mask {title_suffix}",
                       weight="bold")

    plt.tight_layout()
    dset_name = dset_name.replace(" ", "_")
    plt.savefig(f"paper_plots_r4/{dset_name}_sample_complexity{result_file_suffix}.png")
    plt.show()


def make_mask_corruption_sample_complexity_plots(dset_name: str, device: str, corruption_type: int, with_l2_prop: bool = False, methods: list[str] = None) -> None:
    assert dset_name in ["decoy_mnist", "derma_mnist"]
    if methods is None:
        methods = ["r3", "r4", "ibp_ex", "rand_r4", "pgd_r4"]
    dl_test, model_dir, model, eps, has_conv, loss_fn, num_groups = None, None, None, None, None, None, None
    result_file_suffix = None
    match corruption_type:
        case 0:
            result_file_suffix = "_misposition"
        case 1:
            result_file_suffix = "_shift"
        case 2:
            result_file_suffix = "_shrink"
        case 3:
            result_file_suffix = "_dilation"
    crfs = deepcopy(result_file_suffix)
    result_file_suffix = result_file_suffix + "_propl2" if with_l2_prop else result_file_suffix
    result_yaml_file = f"experiment_results/{dset_name}_sample_complexity{result_file_suffix}.yaml"
    if dset_name == "decoy_mnist":
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        loss_fn = "cross_entropy"
        eps = 0.1
        model_dir = "saved_experiment_models/performance/decoy_mnist"
        has_conv = False
        num_groups = 10
    if dset_name == "derma_mnist":
        test_derma = derma_mnist.DecoyDermaMNIST(False, size=64)
        dl_test = derma_mnist.get_dataloader(test_derma, 256)
        model = DermaNet(3, 64, 1)
        loss_fn = "binary_cross_entropy"
        eps = 0.01
        model_dir = "saved_experiment_models/performance/derma_mnist"
        has_conv = True
        num_groups = 2
    dset_name = dset_name.replace("_", " ")

    sns.set_theme(context="poster", font_scale=3)
    sns.color_palette("bright")
    ratios = np.array([1, 0.8, 0.6, 0.4, 0.2, 0])
    fig, ax = plt.subplots(1, 2, figsize=(70, 28))
    min_wg, max_wg = 1, 0
    for method in methods:
        mean_wg_accs, stddev_wg_accs, mean_delta, std_dev_delta = [], [], [], []
        #* Measure acc and rob metrics for ratio 1
        _, wg_acc, _, _, stddev_wg_acc = get_restart_avg_and_worst_group_accuracy_with_stddev(dl_test, model_dir + f"/{method}", model, device, num_groups, multi_class=(num_groups > 2), suppress_log=True)
        delta_mean, _, _, delta_std, *_ = get_avg_rob_metrics(model, dl_test, device, model_dir + f"/{method}", eps, loss_fn, has_conv=has_conv)
        delta_mean, delta_std = round(float(delta_mean.item()), 5), round(float(delta_std.item()), 5)
        mean_wg_accs.append(wg_acc)
        stddev_wg_accs.append(stddev_wg_acc)
        mean_delta.append(delta_mean)
        std_dev_delta.append(delta_std)
        for ratio in ratios[1:]:
            abl_results_for_method_and_ratio = load_params_or_results_from_file(result_yaml_file, method + f"_{int(ratio * 100)}")
            mean_wg_accs.append(abl_results_for_method_and_ratio["worst_group_acc"])
            stddev_wg_accs.append(abl_results_for_method_and_ratio["stddev_worst_group_acc"])
            mean_delta.append(abl_results_for_method_and_ratio["delta_mean"])
            std_dev_delta.append(abl_results_for_method_and_ratio["delta_stddev"])

        min_wg, max_wg = min(min_wg, *mean_wg_accs), max(max_wg, *mean_wg_accs)
        sns.lineplot(x=ratios, y=mean_wg_accs, label=f"{method.upper()}", marker="o", legend="full", ax=ax[0], linewidth=10, estimator=None)
        sns.lineplot(x=ratios, y=mean_delta, label=f"{method.upper()}", marker="o", legend="full", ax=ax[1], linewidth=10, estimator=None)
        ax[0].fill_between(ratios, np.array(mean_wg_accs) - np.array(stddev_wg_accs), np.array(mean_wg_accs) + np.array(stddev_wg_accs), alpha=0.15)
        ax[0].xaxis.set_inverted(True)
        ax[0].set(xlabel=f"% of healthy masks ({crfs[1:]})\n", ylabel="Average Worst Group Test Accuracy")
        ax[1].fill_between(ratios, np.array(mean_delta) - np.array(std_dev_delta), np.array(mean_delta) + np.array(std_dev_delta), alpha=0.15)
        ax[1].xaxis.set_inverted(True)
        ax[1].set(xlabel=f"% of healthy masks ({crfs[1:]})\n", ylabel=r'Average $\delta$')
        ax[1].set_yscale("symlog")

    ax[0].set_ylim([min_wg * 0.95, max_wg * 1.05])
    ax[0].set_title(f"Worst group test accuracy for {dset_name.upper()} upon \n varying the ratio of corrupted masks", weight="bold")
    ax[1].set_title(r'Average $\delta$ (1-$\delta$-input-robustness) for ' + f"{dset_name.upper()} \n upon varying the ratio of corrupted masks",
                       weight="bold")

    plt.tight_layout()
    dset_name = dset_name.replace(" ", "_")
    plt.savefig(f"paper_plots_r4/{dset_name}_sample_complexity{result_file_suffix}.png")
    plt.show()


def make_mask_abl_hmap(dset_name: str,  methods: list[str] = None) -> None:
    assert dset_name in ["decoy_mnist", "derma_mnist"]
    if methods is None:
        methods = ["r3", "r4", "ibp_ex", "rand_r4", "pgd_r4", "ibp_ex+r3"]
    result_yaml_file = f"experiment_results/{dset_name}_hmap.yaml"
    dset_as_title = dset_name.replace("_", " ").upper()

    sns.set_theme(context="poster", font_scale=1.6)
    sns.color_palette("bright")
    ratios = np.array([1, 0.8, 0.6, 0.4, 0.2, 0])
    fig, ax = plt.subplots(3, 2, figsize=(35, 46))
    for method_idx, method in enumerate(methods):
        # Pull up the weight decay or weight reg coeff from the saved parameters
        perf_param_file = f"experiment_results/{dset_name}_params.yaml"
        method_params = load_params_or_results_from_file(perf_param_file, method)
        wreg_init = None
        if "weight_decay" in method_params and method_params["weight_decay"] > 0:
            wreg_init = method_params["weight_decay"]
        elif "weight_coeff" in method_params and method_params["weight_coeff"] > 0:
            wreg_init = method_params["weight_coeff"]
        else:
            wreg_init = 0
        wregs_for_method = [wreg_init / 1000, wreg_init / 100, wreg_init / 10, wreg_init, wreg_init * 10, wreg_init * 100]
        #* Special case
        if method == "r4" and dset_name == "decoy_mnist":
            wregs_for_method = [0.001, 0.01, 0.1, 0, 10, 100]
        min_wg, max_wg = 100, 0
        row, col = method_idx // 2, method_idx % 2
        mean_wg_accs = np.zeros((len(wregs_for_method), len(ratios)))
        #* Measure acc and rob metrics for ratio 1
        for wreg_id, _ in enumerate(wregs_for_method):
            for ridx, ratio in enumerate(ratios):
                key_name = f"{method}wd_{wreg_id}_{int(ratio * 100)}"
                abl_results_for_method_and_ratio = load_params_or_results_from_file(result_yaml_file, key_name)
                mean_wg_accs[int(wreg_id)][ridx] = abl_results_for_method_and_ratio["worst_group_acc"] * 100
        min_wg, max_wg = min(min_wg, *mean_wg_accs.flatten()), max(max_wg, *mean_wg_accs.flatten())

        # Round every element in wregs_for_method to 1 decimal and scientific notation
        wregs_for_method = [f"{wreg:.0e}" for wreg in wregs_for_method]
        sns.heatmap(mean_wg_accs, ax=ax[row][col], xticklabels=ratios, yticklabels=wregs_for_method, annot=True, fmt=".1f",
                    vmin=min_wg * 0.95, vmax=max_wg * 1.05, linewidth=.3)
        ax[row][col].set(xlabel="% of masks\n\n", ylabel="Weight decay")
        method_as_title = method.replace("_", "-").upper()
        ax[row][col].set_title(f"Worst group test accuracy for dataset \n {dset_as_title} and method {method_as_title}", weight="bold")

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f"paper_plots_r4/{dset_name}_mask_hmap.png")
    plt.show()

def make_r4_worst_mask_corr_plot(dset_name: str, device: str, with_l2_prop: bool = False) -> None:
    assert dset_name in ["decoy_mnist", "derma_mnist"]
    dl_test, model_dir, model, eps, has_conv, loss_fn, num_groups = None, None, None, None, None, None, None
    corruption_types = ["MISPOS", "SHIFT", "SHRINK", "DILATE"]
    suffixes = ["_misposition", "_shift", "_shrink", "_dilation"]
    suffixes = [suffix + "_propl2" if with_l2_prop else suffix for suffix in suffixes]
    result_yaml_files = [f"experiment_results/{dset_name}_sample_complexity{result_file_suffix}.yaml" for result_file_suffix in suffixes]
    if dset_name == "decoy_mnist":
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        loss_fn = "cross_entropy"
        eps = 0.1
        model_dir = "saved_experiment_models/performance/decoy_mnist"
        has_conv = False
        num_groups = 10
    if dset_name == "derma_mnist":
        test_derma = derma_mnist.DecoyDermaMNIST(False, size=64)
        dl_test = derma_mnist.get_dataloader(test_derma, 256)
        model = DermaNet(3, 64, 1)
        loss_fn = "binary_cross_entropy"
        eps = 0.01
        model_dir = "saved_experiment_models/performance/derma_mnist"
        has_conv = True
        num_groups = 2
    dset_name = dset_name.replace("_", " ")

    sns.set_theme(context="poster", font_scale=3)
    sns.color_palette("bright")
    ratios = np.array([1, 0.8, 0.6, 0.4, 0.2, 0])
    fig, ax = plt.subplots(1, 2, figsize=(70, 28))
    min_wg, max_wg = 1, 0
    _, r4_100_wg_acc, _, _, r4_100_stddev_wg_acc = get_restart_avg_and_worst_group_accuracy_with_stddev(dl_test, model_dir + "/r4", model, device, num_groups, multi_class=(num_groups > 2), suppress_log=True)
    r4_100_delta_mean, _, _, r4_100_delta_std, *_ = get_avg_rob_metrics(model, dl_test, device, model_dir + "/r4", eps, loss_fn, has_conv=has_conv)
    r4_100_delta_mean, r4_100_delta_std = round(float(r4_100_delta_mean.item()), 5), round(float(r4_100_delta_std.item()), 5)
    for result_yaml_file, corr_type in zip(result_yaml_files, corruption_types):
        mean_wg_accs, stddev_wg_accs, mean_delta, std_dev_delta = [], [], [], []
        #* Measure acc and rob metrics for ratio 1
        mean_wg_accs.append(r4_100_wg_acc)
        stddev_wg_accs.append(r4_100_stddev_wg_acc)
        mean_delta.append(r4_100_delta_mean)
        std_dev_delta.append(r4_100_delta_std)
        for ratio in ratios[1:]:
            abl_results_for_method_and_ratio = load_params_or_results_from_file(result_yaml_file, f"r4_{int(ratio * 100)}")
            mean_wg_accs.append(abl_results_for_method_and_ratio["worst_group_acc"])
            stddev_wg_accs.append(abl_results_for_method_and_ratio["stddev_worst_group_acc"])
            mean_delta.append(abl_results_for_method_and_ratio["delta_mean"])
            std_dev_delta.append(abl_results_for_method_and_ratio["delta_stddev"])

        min_wg, max_wg = min(min_wg, *mean_wg_accs), max(max_wg, *mean_wg_accs)
        lsty = "dotted" if corr_type == "DILATE" and dset_name == "derma mnist" else "solid"
        sns.lineplot(x=np.flip(ratios), y=mean_wg_accs, label=corr_type, marker="o", legend="full", ax=ax[0], linewidth=10, estimator=None, linestyle=lsty)
        sns.lineplot(x=np.flip(ratios), y=mean_delta, label=corr_type, marker="o", legend="full", ax=ax[1], linewidth=10, estimator=None)
        ax[0].fill_between(np.flip(ratios), np.array(mean_wg_accs) - np.array(stddev_wg_accs), np.array(mean_wg_accs) + np.array(stddev_wg_accs), alpha=0.15)
        ax[0].set(xlabel="% of corrupted masks \n", ylabel="Average Worst Group Test Accuracy")
        ax[1].fill_between(np.flip(ratios), np.array(mean_delta) - np.array(std_dev_delta), np.array(mean_delta) + np.array(std_dev_delta), alpha=0.15)
        ax[1].set(xlabel="% of corrupted masks \n", ylabel=r'Average $\delta$')
        ax[1].set_yscale("symlog")

    ax[0].set_ylim([min_wg * 0.95 if min_wg > 0 else -0.1, max_wg * 1.05])
    ax[0].set_title(f"Worst group test accuracy for R4 on {dset_name.upper()} upon \n varying the ratio of corrupted masks", weight="bold")
    ax[1].set_title(r'Average $\delta$ (1-$\delta$-input-robustness) for ' + f"{dset_name.upper()} \n upon varying the ratio of corrupted masks",
                       weight="bold")

    plt.tight_layout()
    dset_name = dset_name.replace(" ", "_")
    plt.savefig(f"paper_plots_r4/{dset_name}_sample_complexity_r4_corr.png")
    plt.show()


def make_model_size_ablation_plot(dset_name: str, methods: list[str] = None) -> None:
    assert dset_name in ["decoy_mnist", "derma_mnist"]
    if methods is None:
        methods = ["r3", "r4", "ibp_ex", "rand_r4", "pgd_r4"]
    size_names, size_name_xlabels  = None, None
    result_yaml_file = f"experiment_results/{dset_name}_size.yaml"
    if dset_name == "decoy_mnist":
        size_names = ["1_layer", "2_layer", "3_layer", "4_layer"]
        size_name_xlabels = ["1 layer", "2 layers", "3 layers", "4 layers"]
    if dset_name == "derma_mnist":
        size_names = ["small", "small_medium", "medium_large", "large"]
        size_name_xlabels = ["Small", "Small-Medium", "Medium-Large", "Large"]
    dset_name = dset_name.replace("_", " ")

    sns.set_theme(context="poster", font_scale=3)
    sns.color_palette("bright")
    fig, ax = plt.subplots(1, 2, figsize=(70, 28))
    min_wg, max_wg = 1, 0
    for method in methods:
        mean_wg_accs, stddev_wg_accs, mean_delta, std_dev_delta = [], [], [], []
        for size_name in size_names:
            abl_results_for_method_and_ratio = load_params_or_results_from_file(result_yaml_file, method + f"_{size_name}")
            mean_wg_accs.append(abl_results_for_method_and_ratio["worst_group_acc"])
            stddev_wg_accs.append(abl_results_for_method_and_ratio["stddev_worst_group_acc"])
            mean_delta.append(abl_results_for_method_and_ratio["delta_mean"])
            std_dev_delta.append(abl_results_for_method_and_ratio["delta_stddev"])

        min_wg, max_wg = min(min_wg, *mean_wg_accs), max(max_wg, *mean_wg_accs)
        sns.lineplot(x=size_name_xlabels, y=mean_wg_accs, label=f"{method.upper()}", marker="o", legend="full", ax=ax[0], linewidth=10, estimator=None)
        sns.lineplot(x=size_name_xlabels, y=mean_delta, label=f"{method.upper()}", marker="o", legend="full", ax=ax[1], linewidth=10, estimator=None)
        ax[0].fill_between(size_name_xlabels, np.array(mean_wg_accs) - np.array(stddev_wg_accs), np.array(mean_wg_accs) + np.array(stddev_wg_accs), alpha=0.15)
        ax[0].set(xlabel="Model Architecture\n", ylabel="Average Worst Group Test Accuracy")
        ax[1].fill_between(size_name_xlabels, np.array(mean_delta) - np.array(std_dev_delta), np.array(mean_delta) + np.array(std_dev_delta), alpha=0.15)
        ax[1].set(xlabel="Model Architecture\n", ylabel=r'Average $\delta$')
        ax[1].set_yscale("symlog")

    ax[0].set_ylim([min_wg * 0.95, max_wg * 1.05])
    ax[0].set_title(f"Worst group test accuracy for {dset_name.upper()} upon \n varying the model architecture", weight="bold")
    ax[1].set_title(r'Average $\delta$ (1-$\delta$-input-robustness) for ' + f"{dset_name.upper()} \n upon varying model architecture",
                       weight="bold")

    plt.tight_layout()
    dset_name = dset_name.replace(" ", "_")
    plt.savefig(f"paper_plots_r4/{dset_name}_size.png")
    plt.show()
