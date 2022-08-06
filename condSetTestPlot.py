import moses
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

list_of_generated_smiles = pd.read_csv("moses_bench2_10conds_lat=128_epo=1111111111111_k=4_20210118.csv")
mols = list_of_generated_smiles['mol']

e1c = sns.color_palette('bright')[0]
e1c1 = sns.color_palette('bright')[3]

target = ["trg(logP)", "trg(tPSA)", "trg(QED)", "toklen"]
generated = ["rdkit(logP)", "rdkit(tPSA)", "rdkit(QED)", "toklen_gen"]
title = ["logP", "tPSA", "QED", "Num. of tokens"]
x_label = ["Target (logP)", "Target (tPSA)", "Target (QED)", "Length of latent code"]
y_label = ["Generated molecule", "Generated molecule", "Generated molecule", "Number of generated tokens"]
xticker = ["#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10"]

with sns.axes_style("darkgrid"):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12,5))
    fig.subplots_adjust(hspace=0.5)

    sns.set(font_scale=2.0)
    sns.axes_style({'font.serif': ['Arial']})

idx = -1
for ax, name, trg, gen in zip(axes.flatten(), title, list_of_generated_smiles[target], list_of_generated_smiles[generated]):
    idx = 0
    if name == "logP":
        ax.set_yticks(np.arange(0.5, 5, step=1))
    if name == "tPSA":
        ax.set_yticks(np.arange(50, 100, step=15))
    if name == "QED":
        ax.set_yticks(np.arange(0.4, 1.0, step=0.2))
    if name == "Num. of tokens":
        ax.set_yticks(np.arange(20, 60, step=10))

    ax.set_xticks(np.arange(0, 11, step=1))
    ax.set_xticklabels('')
    ax.set_xticks(np.arange(0.5, 10.5, step=1), minor=True)
    ax.set_xticklabels(xticker, minor=True)

    for i in range(10):
        x = np.arange((idx*200)/200, ((idx+1)*200)/200, step=1/200)
        if trg != "toklen":
            sns.lineplot(x, list_of_generated_smiles[trg][idx*200:(idx+1)*200], color='red', ax=ax)
        if name == "logP" and i == 9:
            sns.lineplot(x, list_of_generated_smiles[trg][idx * 200:(idx + 1) * 200], color='red', ax=ax, label='Precondition set')
        if name == "Num. of tokens" and i == 9:
            sns.scatterplot(x, list_of_generated_smiles[trg][idx * 200:(idx + 1) * 200], color='red', ax=ax, s=3, marker='x', label='Length of latent code')
        else:
            sns.scatterplot(x, list_of_generated_smiles[trg][idx * 200:(idx + 1) * 200], color='red', ax=ax, s=3, marker='x')
        if name == "logP" and i == 9:
            sns.scatterplot(x, list_of_generated_smiles[gen][idx*200:(idx+1)*200], color='blue', linewidth=0, ax=ax, s=3, label='Generated')
        if name == "Num. of tokens" and i == 9:
            sns.scatterplot(x, list_of_generated_smiles[gen][idx*200:(idx+1)*200], color='blue', linewidth=0, ax=ax, s=3, label='Generated')
        else:
            sns.scatterplot(x, list_of_generated_smiles[gen][idx * 200:(idx + 1) * 200], color='blue', linewidth=0, ax=ax, s=3)
        idx += 1
    if name == 'logP':
        ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1), fontsize=9.5)
    if name == 'Num. of tokens':
        ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1), fontsize=9.5)
    ax.set_ylabel(name)

plt.xlabel("Precondition set")
# plt.savefig("conds10.svg")
plt.show()

