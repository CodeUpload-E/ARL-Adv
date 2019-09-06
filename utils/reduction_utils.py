from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE


# def embed_topic_file2point_topic_file(infile, outfile, *args, **kwargs):
#     embed_list, topic_list = np.load(infile)
#     # print(infile, embed_list.shape, topic_list.shape)
#     embed_list = np.array([np.array(v) for v in embed_list])
#     point_list = fit_tsne(embed_list, *args, **kwargs)
#     points_to_file(point_list, topic_list, outfile)
#
#
# def points_to_file(points, labels, outfile):
#     import utils.array_utils as au
#     import utils.io_utils as fu
#     labels = au.reindex(labels)
#     lines = ['{:.4f} {:.4f} {}'.format(p[0], p[1], i) for p, i in zip(points, labels)]
#     fu.write_lines(outfile, lines)


def fit_tsne(x, n_components=2, init='pca', *args, **kwargs):
    return TSNE(n_components=n_components, init=init, *args, **kwargs).fit_transform(x)


def fit_kernel_pca(x, n_components, kernel='rbf', *args, **kwargs):
    x_new = KernelPCA(n_components=n_components, kernel=kernel, *args, **kwargs).fit_transform(x)
    return x_new


def fit_pca(x, n_components, *args, **kwargs):
    pca = PCA(n_components=n_components, *args, **kwargs)
    pca.fit(x)
    print('component portion:', pca.explained_variance_ratio_[:10])
    return pca.transform(x)


def fit_svd(x, n_components, n_iter, *args, **kwargs):
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, *args, **kwargs)
    svd.fit(x)
    return svd.transform(x)


def fit_multi(func, xs, kwargs_list=None):
    import utils.multiprocess_utils as mu
    res_list = mu.multi_process(func, [(x,) for x in xs], kwargs_list)
    return res_list

# if __name__ == '__main__':
#     a = np.random.normal(size=[33, 40])
#     b = np.random.normal(size=[44, 40])
#     a_, b_ = fit_multi(fit_tsne, [a, b], kwargs_list=[{'n_components': 1}] * 2)
#     print(a_.shape, b_.shape)
