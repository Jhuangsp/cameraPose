import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import numpy.random as rd

n = 5
num = [20,20,30,10,20]
p_terrs = [rd.rand(n)+10 for n in num]
p_rerrs_i = [rd.rand(n)+3 for n in num]
p_rerrs_j = [rd.rand(n)+4 for n in num]
p_rerrs_k = [rd.rand(n)+5 for n in num]

n_sec = [len(i) for i in p_rerrs_i]

p_terrs = np.concatenate(p_terrs)
p_rerrs_i = np.concatenate(p_rerrs_i)
p_rerrs_j = np.concatenate(p_rerrs_j)
p_rerrs_k = np.concatenate(p_rerrs_k)

pit_label = []
for i,n in enumerate(n_sec):
    for nn in range(n):
        pit_label.append("r{}".format(i))
type_label = ["Translation Error (m)", "Rotation Error (rad)"]
tra_label = ["camera position"]
rot_label = ["axis_i", "axis_j", "axis_k"]

rot_labels = pd.MultiIndex.from_product(
    [[type_label[1]], rot_label, pit_label], #5*1*3
    names=['type', 'station', 'measurement'])
rot_data = pd.DataFrame({'value': np.concatenate((p_rerrs_i, p_rerrs_j, p_rerrs_k))}, index=rot_labels)
rot_data = rot_data.reset_index()
# print(rot_data)

tra_labels = pd.MultiIndex.from_product(
    [[type_label[0]], tra_label, pit_label], #5*1*3
    names=['type', 'station', 'measurement'])
tra_data = pd.DataFrame({'value': p_terrs}, index=tra_labels)
tra_data = tra_data.reset_index()
# print(tra_data)

g1 = sns.FacetGrid(rot_data, col="measurement", row="type", height=3, aspect=1, sharey=True, sharex=True)
for ax in g1.axes:
    for a in ax:
        a.grid(True)
g1.map(sns.boxplot, "station", "value", order=rot_label, palette=['lightcoral', 'lightgreen', 'lightblue'])
g1.set_axis_labels("", "val").set_titles("{col_name}").despine(bottom=True, left=True)
g1.axes[0,0].set_ylabel(type_label[1])
g1.fig.suptitle('Pitch Matter Rotation error', y=0.99)
g1.fig.set_size_inches(15, 4)
g1.savefig("rotErr.png")
plt.show()

# g2 = sns.FacetGrid(tra_data, col="measurement", row="type", sharey=True, sharex=True)
# for ax in g2.axes:
#     for a in ax:
#         a.grid(True)
# g2.map(sns.boxplot, "station", "value", order=tra_label, palette=['orange'])
# g2.set_axis_labels("", "val").set_titles("{col_name}").despine(bottom=True, left=True)
# g2.axes[0,0].set_ylabel(type_label[0])
# g2.fig.suptitle('Pitch Matter Translation error', y=0.99)
# g2.savefig("traErr.png")
# plt.show()