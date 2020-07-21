# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import sklearn.datasets as ds
import warnings


def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=UserWarning)
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    
    # 400个样本，准备聚成4类
    n_clusters = 4
    N          = 400
    
    # 构造4个数据集，用这4个数据集分别测试聚类效果
    # 数据集1-A：用makeblobs构造的数据集
    # sklearn.datasets.make_blobs: generate isotropic (各向同性) Gaussian blobs for clustering
    data1, y1 = ds.make_blobs(
                        n_samples   = N,  # 样本数 
                        n_features  = 2,  # 特征数
                        centers     = ((-1, 1), (1, 1), (1, -1), (-1, -1)), # 4个cluster中心点的特征值
                        cluster_std = (0.1, 0.2, 0.3, 0.4), # 4个cluster的方差
                        random_state = 0  # random seed
                )
    data1   = np.array(data1)
    
    # 数据集1-B：数据集1-A叠加噪声
    n_noise = int(0.1*N)
    r       = np.random.rand(n_noise, 2)
    data_min1, data_min2 = np.min(data1, axis=0)
    data_max1, data_max2 = np.max(data1, axis=0)
    r[:, 0] = r[:, 0] * (data_max1-data_min1) + data_min1
    r[:, 1] = r[:, 1] * (data_max2-data_min2) + data_min2
    data1_noise = np.concatenate((data1, r), axis=0)
    y1_noise    = np.concatenate((y1, [4]*n_noise))
    
    # 数据集2-A：用make_moons构造的数据集
    # sklearn.datasts.make_moons: interleaving half circles, a simple toy dataset to visualize clustering and classification algorithms
    data2, y2 = ds.make_moons(n_samples=N, noise=.05)
    data2     = np.array(data2)
    
    # 数据集2-B：数据集2-A叠加噪声
    n_noise   = int(0.1 * N)
    r = np.random.rand(n_noise, 2)
    data_min1, data_min2 = np.min(data2, axis=0)
    data_max1, data_max2 = np.max(data2, axis=0)
    r[:, 0] = r[:, 0] * (data_max1 - data_min1) + data_min1
    r[:, 1] = r[:, 1] * (data_max2 - data_min2) + data_min2
    data2_noise = np.concatenate((data2, r), axis=0)
    y2_noise = np.concatenate((y2, [3] * n_noise))

    # matplotlib参数
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 可视化图表外框
    cm = mpl.colors.ListedColormap(['r', 'g', 'b', 'm', 'c'])
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.cla()

    # https://cloud.tencent.com/developer/article/1020155
    # http://sklearn123.com/FourthSection/2.3.Clustering/
    # https://www.studyai.cn/modules/clustering.html#hierarchical-clustering
    # AgglomerativeClustering 使用自下而上的方法进行层次聚类:开始是每一个对象是一个聚类， 并且聚类别相继合并在一起。 linkage criteria 确定用于合并的策略的度量:
	# • Ward 最小化所有聚类内的平方差总和。这是一种 variance-minimizing （方差最小化）的优化方向， 这是与k-means 的目标函数相似的优化方法，但是用 agglomerative hierarchical（聚类分层）的方法处理。
	# • Maximum 或 complete linkage 最小化聚类对两个样本之间的最大距离。
	# • Average linkage 最小化聚类两个聚类中样本距离的平均值。
    # AgglomerativeClustering 在于连接矩阵联合使用时，也可以扩大到大量的样本，但是 在样本之间没有添加连接约束时，计算代价很大:每一个步骤都要考虑所有可能的合并。
    # Agglomerative cluster 存在 “rich get richer” 现象导致聚类大小不均匀
    #   这方面 complete linkage 是最坏的策略
    #   ward 给出了最规则的大小。然而，在 Ward 中 affinity (or distance used in clustering) 不能被改变
    #   对于 non Euclidean metrics 来说 average linkage 是一个好的选择
    


    
    # 4种聚类策略: {“ward”, “complete”, “average”, “single”}, default=”ward”
    # `ward`: minimizes the variance of the clusters being merged.
    # `average`: uses the average of the distances of each observation of the two sets.
    # `complete` or `maximum linkage`: uses the maximum distances between all observations of the two sets.
    # `single`: uses the minimum of the distances between all observations of the two sets.
    linkages = ("ward", "complete", "average")

    # 对4个数据集聚类（1-A/B聚成4个cluster，2-A/B聚成2个cluster，并可视化绘制4张子图）
    for index, (n_clusters, data, y) in enumerate(((4, data1, y1), (4, data1_noise, y1_noise),
                                                   (2, data2, y2), (2, data2_noise, y2_noise))):
        
        # 子图：散点图
        plt.subplot(4, 4, 4*index+1)
        plt.scatter(data[:, 0], data[:, 1], c=y, s=12, edgecolors='k', cmap=cm)
        plt.title('Prime', fontsize=12)
        plt.grid(b=True, ls=':')
        data_min1, data_min2 = np.min(data, axis=0)
        data_max1, data_max2 = np.max(data, axis=0)
        plt.xlim(extend(data_min1, data_max1))
        plt.ylim(extend(data_min2, data_max2))
        # 用knn生成connectivity矩阵
        connectivity = kneighbors_graph(
                                data,               # data set
                                n_neighbors=7,      # number of neighbors for each sample
                                mode='distance',    # return type：{‘connectivity’ , ‘distance’}, default=’connectivity’， connectivity matrix with ones and zeros or distances
                                metric='minkowski', # distance metric, default distance is ‘euclidean’ (‘minkowski’ metric with the p param equal to 2.
                                p=2,                # when p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
                                include_self=True   # whether or not to mark each sample as the first nearest neighbor to itself
                                # ,n_jobsint, default=None  # parallel number of threads
                                )
        connectivity = 0.5 * (connectivity + connectivity.T)
        # 依次尝试4种策略，未当前数据集生成4个不同的聚类结果
        # affinity:  “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
        #   * if linkage is “ward”, only “euclidean” is accepted
        #   * if “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method
        # compute full tree: 
        #   * it must be True if distance_threshold is not None. 
        #   * by default is “auto”, which is 
        #     * equivalent to True when distance_threshold is not None 
        #     * or that n_clusters is inferior to the maximum between 100 or 0.02 * n_samples
        #     * otherwise, “auto” is equivalent to False
        # distance_threshold, default=None
        #     * the linkage distance threshold above which, clusters will not be merged. 
        #     * if not None, n_clusters must be None and compute_full_tree must be True.
        for i, linkage in enumerate(linkages):
            ac = AgglomerativeClustering(
                            n_clusters   = n_clusters,   # cluster数目
                            affinity     = 'euclidean',  # 样本距离度量方法
                            connectivity = connectivity, # a connectivity matrix, such as derived from kneighbors_graph, default is None
                            linkage=linkage              # linkage criterion: {“ward”, “complete”, “average”, “single”}, default=”ward”
                            #, compute_full_tree         # default=’auto’, must be True if distance_threshold is not None.
                            )
            ac.fit(data)
            # 聚类结果
            y = ac.labels_
            # 完成子图绘制
            plt.subplot(4, 4, i+2+4*index)
            plt.scatter(data[:, 0], data[:, 1], c=y, s=12, edgecolors='k', cmap=cm)
            plt.title(linkage, fontsize=12)
            plt.grid(b=True, ls=':')
            plt.xlim(extend(data_min1, data_max1))
            plt.ylim(extend(data_min2, data_max2))
    # 显示可视化图            
    plt.suptitle('层次聚类的不同合并策略', fontsize=15)
    plt.tight_layout(0.5, rect=(0, 0, 1, 0.95))
    plt.show()
