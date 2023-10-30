def MiniBatchKMEANS(x, max_iter, error, batch_size, a, b):
    import matplotlib.pyplot as plt
    from sklearn.metrics import pairwise_distances

    x1, x2 = x.iloc[:,a], x.iloc[:,b]
    feature_A, feature_B = a, b
    fig1,axes1 = plt.subplots(3,3,figsize = (8,8))
    colors = ["b", "orange", "g", "indigo", "c", "m", "y", "k", "Brown", "ForestGreen"]
    Model_Cluster, Labels, Centers_Matrix = [], [], []
    Silhouette, CH, DB, DU = [], [], [], []

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        Model_Kmeans = MiniBatchKMeans(n_clusters = ncenters, max_iter = max_iter, tol = error, 
                                 batch_size = batch_size, random_state = 17, n_init = 'auto')
        Model_Kmeans.fit(x)
        Classes = Model_Kmeans.labels_
        Centers = Model_Kmeans.cluster_centers_
        Model_Cluster.append(Model_Kmeans)
        Labels.append(Classes)
        Centers_Matrix.append(Centers)  
        Silhouette.append(silhouette_score(x, Classes))
        CH.append(calinski_harabasz_score(x, Classes))
        DB.append(davies_bouldin_score(x, Classes))
        dist = pairwise_distances(x)
        DU.append(dunn(dist,Classes))
  
        for j in range(ncenters):
            ax.plot(x1[Classes == j], x2[Classes == j], '.', color = colors[j])  
        for pt in Centers:
            ax.plot(pt[feature_A],pt[feature_B],'rs')

        ax.set_title('Cluster = {0}'.format(ncenters))
        ax.axis('off')

    Metrics = np.vstack([Silhouette, CH, DB, DU])
    fig1.tight_layout()
    plt.show()

    return Model_Cluster, Labels, Centers_Matrix, Metrics