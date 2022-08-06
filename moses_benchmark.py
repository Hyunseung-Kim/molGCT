import moses
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

list_of_generated_smiles = pd.read_csv("moses_bench2_lat=128_epo=1111111111111_k=4_20210118.csv")
mols = list_of_generated_smiles['mol']
# metrics = moses.get_all_metrics(mols)
# print(metrics)


e1c = sns.color_palette('muted')[0]
e2c = sns.color_palette('pastel')[6]
e3c = sns.color_palette('colorblind')[0]
e4c = sns.color_palette('bright')[0]


target = ["trg(logP)", "trg(tPSA)", "trg(QED)", "toklen"]
generated = ["rdkit(logP)", "rdkit(tPSA)", "rdkit(QED)", "toklen_gen"]
title = ["logP", "tPSA", "QED", "Number of tokens"]
x_label = ["Target (logP)", "Target (tPSA)", "Target (QED)", "Length of latent code"]
y_label = ["Generated molecule", "Generated molecule", "Generated molecule", "Num. of generated tokens"]

for i in range(4):
    with sns.axes_style("white"):
        sns.set(font_scale=2.0)
        sns.axes_style({'font.serif': ['Arial']})
        xlim_lb = min(np.min(list_of_generated_smiles[generated[i]]), np.min(list_of_generated_smiles[target[i]]))
        xlim_ub = max(np.max(list_of_generated_smiles[generated[i]]), np.max(list_of_generated_smiles[target[i]]))
        gap = xlim_ub - xlim_lb
        xlim_lb, xlim_ub = xlim_lb + (-0.05 * gap), xlim_ub + (0.05 *gap)
        ylim_lb, ylim_ub = xlim_lb, xlim_ub

        print(xlim_lb, xlim_ub, ylim_lb, ylim_ub)
        plt.xlim(xlim_lb, xlim_ub)
        plt.ylim(ylim_lb, ylim_ub)
        plt.gca().set_aspect('equal', adjustable='box')

        g = sns.JointGrid(list_of_generated_smiles[target[i]], list_of_generated_smiles[generated[i]])
        g.plot_marginals(sns.distplot, kde=False, color=e3c)
        g.plot_joint(plt.scatter, color=e3c, alpha=.05, label=title[i])
        g.set_axis_labels(x_label[i], y_label[i])
        if i <3:
            if i == 1:
                plt.xticks(np.arange(50, 200, step=50), np.arange(50, 200, step=50))
                plt.yticks(np.arange(50, 200, step=50), np.arange(50, 200, step=50))
            if i == 2:
                plt.xticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8])
                plt.yticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8])
        else:
            plt.xticks(np.arange(20, 60, step=10), np.arange(20, 60, step=10))
            plt.yticks(np.arange(20, 60, step=10), np.arange(20, 60, step=10))

        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        g.savefig("{}2.svg".format(generated[i]), figsize=(20,20))
        plt.show()