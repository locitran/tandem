import os
import matplotlib.pyplot as plt
import numpy as np
from .features import TANDEM_FEATS, all_feat, dynamics_feat, structure_feat, sequence_feat
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

featSet = TANDEM_FEATS['v1.1']
error_params = {
    'capsize': 2,
    'capthick': 1.5,
    'elinewidth': 1
}
colors = np.array([
    'lightcoral' if f in dynamics_feat else
    'lightgreen'   if f in structure_feat else
    'skyblue' if f in sequence_feat else
    'gray'  # default/fallback color if not in any group
    for f in featSet
])
labels = np.array([
    'Dynamics' if f in dynamics_feat else
    'Structure'   if f in structure_feat else
    'Sequence&Chemical' if f in sequence_feat else
    'gray'  # default/fallback color if not in any group
    for f in featSet
])
hatches = np.array([
    '//' if f in dynamics_feat else
    '..'   if f in structure_feat else
    '' if f in sequence_feat else
    ''  # default/fallback hatch if not in any group
    for f in featSet
])

featnames = np.array([all_feat[f] for f in featSet])
n_features = len(featnames)

def _plotSHAP_bar(
        phi,
        phi_sem,
        feature_order,
        title,
        axis_fontsize=12, 
        legend_fontsize=10,
    ):

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(
        x=np.arange(n_features) + 0.5,
        height=phi[feature_order][::-1],
        yerr=phi_sem[feature_order][::-1],
        color=colors[feature_order][::-1],
        label=labels[feature_order][::-1],
        hatch=hatches[feature_order][::-1],
        edgecolor='grey', capsize=2, width=0.7,
        error_kw=error_params
    )
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.set_ylabel('Absolute SHAP value', fontsize=axis_fontsize)
    ax.set_xticks(np.arange(n_features)+0.5)
    ax.set_xticklabels(featnames[feature_order][::-1], rotation=90, fontsize=legend_fontsize)
    ax.set_xlabel('Protein features', fontsize=axis_fontsize)
    ax.set_title(title, fontsize=axis_fontsize)

    handles, labels_ = ax.get_legend_handles_labels()
    unique_labels = set(labels_)
    unique_labels = ['Sequence&Chemical', 'Structure', 'Dynamics']
    handles = [handles[labels_.index(label)] for label in unique_labels]
    ax.legend(
        handles, unique_labels, fontsize=legend_fontsize, 
        title='Feature category', title_fontsize=str(legend_fontsize), loc='upper right')
    return fig, ax

def plotSHAP_bar(struct_featImp, title, folder='.', filename=None,
        axis_fontsize=12, legend_fontsize=10,
    ):  

    # globalSHAP: (nSAVs * n_models, n_features) 
    # individualSHAP: (n_models, n_features)
    featImp_arr    = np.vstack(struct_featImp) 
    abs_phi        = np.abs(featImp_arr)
    global_phi     = abs_phi.mean(axis=0) # 
    global_phi_sem = abs_phi.std(axis=0, ddof=1) / np.sqrt(abs_phi.shape[0])
    feature_order  = np.argsort(global_phi)

    fig, ax = _plotSHAP_bar(
        global_phi, global_phi_sem, feature_order, title,
        axis_fontsize=axis_fontsize, legend_fontsize=legend_fontsize
    )
    if filename:
        filepath = os.path.join(folder, filename)
        plt.savefig(
            filepath,
            dpi=300,               # higher resolution (300–600 for papers)
            bbox_inches='tight',   # avoid cutting off labels
        )
    plt.close(fig)