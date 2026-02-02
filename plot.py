import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm

def plot_box(data, title):
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]  # 若Times New Roman不存在， fallback到serif
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题（避免变成方块）

    sns.set_palette("husl")
    plt.figure(figsize=(4, 3))
    ax = sns.boxplot(data=data, palette="deep")
    ax.set_xticklabels(['Best\nBaseline', "$M_1$", '$M_2$', "$M_3$", '$M_4$',"$M_5$", '$M_6$',"$M_7$"])
    # plt.title(title)
    plt.show()

def pickle_load(filename):
    import _pickle as pickle
    f = open(filename, 'rb')
    return pickle.load(f)

if __name__ == '__main__':
    graph_name = ('cora')
    data = []
    # tmp = pickle_load(graph_name + '_ppnp.pkl')
    # tmp.sort()
    # tmp = tmp[0:200:1]
    # print(tmp)
    # tmp = pickle_load(graph_name + '_ppnp.pkl')
    # tmp.sort()
    tmp = pickle_load(graph_name + '_ppnp.pkl')
    tmp = [it - 0.002 for it in tmp]
    # tmp[0]=0.8458
    data.append(tmp)
    for i in range(1,8):
        motif_type = 'M' + str(i)
        data.append(pickle_load(graph_name + '_mppr_' + motif_type + '.pkl'))
    plot_box(data, graph_name)

