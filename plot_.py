import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_box(data, title):
    sns.set_palette("husl")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, palette="husl")
    plt.title(title)
    plt.show()

def pickle_load(filename):
    import _pickle as pickle
    f = open(filename, 'rb')
    return pickle.load(f)

if __name__ == '__main__':
    graph_name = ('pubmed')
    data = []
    # tmp = pickle_load(graph_name + '_ppnp.pkl')
    # tmp.sort()
    # tmp = tmp[0:200:1]
    # print(tmp)
    data.append(pickle_load(graph_name + '_ppnp.pkl'))
    for i in range(1,8):
        motif_type = 'M' + str(i)
        data.append(pickle_load(graph_name + '_0_2_mppr_' + motif_type + '.pkl'))
    plot_box(data, graph_name)
