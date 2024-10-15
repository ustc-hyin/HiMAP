import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_props4img(data, path):

    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)

    x = np.arange(data.shape[1])
    colors = ['#007c9a', '#965e9b']
    custom_legend_labels = ['Intra-Visual Flow', 'Visual-Textual Flow']

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.bar(x, data[0], label=custom_legend_labels[0], alpha=0.3, color=colors[0], width=0.7)
    ax.bar(x, data[1], label=custom_legend_labels[1], alpha=0.3, color=colors[1], width=0.7)

    ax.set_xlabel('Transformer Layer', fontsize=24, fontfamily='Times New Roman', labelpad=12)
    ax.set_ylabel('Importance Metric', fontsize=24, fontfamily='Times New Roman', labelpad=12)
    plt.xticks(fontsize=24, fontfamily='Times New Roman')
    plt.yticks(fontsize=24, fontfamily='Times New Roman')
    ax.legend(fontsize=24, fancybox=True, loc='upper right',
                   prop={'size':24, 'family': 'Times New Roman','style': 'italic'})

    plt.tight_layout()
    plt.savefig(path) 

def plot_props4all(data, path):

    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)

    x = np.arange(data.shape[1])
    colors = ['#C6210D',  '#007fbc', '#3f7430']
    custom_legend_labels = ['System Prompts', 'Image Tokens', 'User Instructions']

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.bar(x, data[2], label=custom_legend_labels[2], alpha=0.5, color=colors[2], width=0.8)
    ax.bar(x, data[0], label=custom_legend_labels[0], alpha=0.3, color=colors[0], width=0.8)
    ax.bar(x, data[1], label=custom_legend_labels[1], alpha=0.3, color=colors[1], width=0.8)

    ax.set_xlabel('Transformer Layer', fontsize=24, fontfamily='Times New Roman', labelpad=12)
    ax.set_ylabel('Imporatance Metric', fontsize=24, fontfamily='Times New Roman', labelpad=12)
    plt.xticks(fontsize=24, fontfamily='Times New Roman')  
    plt.yticks(fontsize=24, fontfamily='Times New Roman')  
    ax.legend(fontsize=4, fancybox=True, loc='upper right',
                   prop={'size':24, 'family': 'Times New Roman','style': 'italic'})
    
    plt.tight_layout()
    plt.savefig(path, dpi=600) 

if __name__ == "__main__":
    
    proportions = torch.load('scivqa_props_7b.pt')

    props_all, props_img = [], []

    for i, props in enumerate(proportions):

        props4all, props4img = props
        props_all.append(props4all)
        props_img.append(props4img)

    props_all = np.array(props_all)
    props_img = np.array(props_img)

    props_all = props_all.mean(axis = 0).transpose(1, 0)
    props_img = props_img.mean(axis = 0).transpose(1, 0)

    plot_props4all(props_all, 'scivqa-props4all-7b.png')
    plot_props4img(props_img, 'scivqa-props4img-7b.png')