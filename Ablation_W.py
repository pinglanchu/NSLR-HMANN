import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


path = './Ablation experiments/'
file = 'W.xlsx'

auc = pd.read_excel(path+file, sheet_name='AUC')
f1 = pd.read_excel(path+file, sheet_name='F1')
algorithms = auc['Networks'].values
fig, ax = plt.subplots(1, 2, figsize=(12, 3))
index = np.arange(8)
bar_width = 0.3

ax[0].grid(axis='y', color='gray', linestyle='--', alpha=0.5, zorder=0)
ax[0].bar(index, auc['W(ones)'], width=bar_width, label='W=I', color='#49c0b6', hatch='///', zorder=10)
ax[0].bar(index+bar_width, auc['ours'], width=bar_width, label='W by eq.(10)', color='#0c3866', hatch='xxx', zorder=10)
ax[0].set_xticks(index + bar_width / 2, algorithms, rotation=-15)
ax[0].set_ylabel('AUC')

ax[1].grid(axis='y', color='gray', linestyle='--', alpha=0.5, zorder=0)
ax[1].bar(index, f1['W(ones)'], width=bar_width, label='W=I', color='#49c0b6', hatch='///', zorder=10)
ax[1].bar(index+bar_width, f1['ours'], width=bar_width, label='W by eq.(10)', color='#0c3866', hatch='xxx', zorder=10)
ax[1].set_xticks(index + bar_width / 2, algorithms, rotation=-15)
ax[1].set_ylabel('F1-Macro')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', ncol=2, handlelength=2, borderpad=0.1)
plt.savefig('Ablation_W.png', dpi=600, bbox_inches='tight')
plt.show()
