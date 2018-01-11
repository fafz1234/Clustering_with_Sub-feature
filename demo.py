import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from KMeans_Sub import KMeans_Sub

# load dataset
dataset = load_wine()
# select features
data_features = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# label of datset
data_label = dataset.target

# number of features in data
n_features = data_features.shape[1]
# number of clusters (given by data)
n_clusters = len(Counter(data_label))

# normalize features
normed_data_features = pd.DataFrame(
    StandardScaler().fit_transform(data_features),
    columns=dataset.feature_names
)
sns.set(font_scale=1.5)

# apply PCA to data
full_pca = PCA(n_components=n_features, random_state=14).fit(normed_data_features)
# pick the optimum number of component from Accumulative Variance Ration plot
plt.plot(range(1, n_features + 1), np.cumsum(full_pca.explained_variance_ratio_))
plt.xlabel("PCA Components")
plt.ylabel("Accumulative Variance Ratio")

# select components
threshold_accum_var_ratio = 0.8
pca_n_features = int(np.nanmin(np.where(np.cumsum(full_pca.explained_variance_ratio_) > threshold_accum_var_ratio, range(1, n_features + 1), np.nan)))

# PCA decomposition
partial_pca = PCA(n_components=pca_n_features, random_state=14)
decomposed_pca_data_features = pd.DataFrame(partial_pca.fit_transform(normed_data_features),
                                            columns=['x%02d' % x for x in range(1, pca_n_features + 1)])

# visualize PCA decomposition
pca_tsne = TSNE(n_components=2, random_state=14)
visualized_pca_data_features = pd.DataFrame(pca_tsne.fit_transform(decomposed_pca_data_features), columns=['x%02d' % x for x in range(1, 3)])
ax = None
for c in range(n_clusters):
    ax = visualized_pca_data_features.iloc[
        list(np.where(np.array(data_label) == c)[0]), :].plot(kind='scatter', x='x01', y='x02',
                                                               color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax)
plt.legend(loc=4, bbox_to_anchor=(1, 0.6))
plt.title('Ground truth of PCA decomposed features')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
savefig('Ground truth of PCA.png')

# apply K-Means to decomposed data
pca_km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=14)
pca_clusters = pca_km.fit_predict(decomposed_pca_data_features)
ax = None
for c in range(n_clusters):
    ax = visualized_pca_data_features.iloc[list(np.where(np.array(pca_clusters) == c)[0]), :].plot(kind='scatter', x='x01', y='x02',
                color=sns.color_palette('husl', 4)[c], label='cluster %d' % c, ax=ax)
plt.legend(loc=4, bbox_to_anchor=(1, 0.6))
plt.title('K-Means clustering on PCA decomposed features')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
savefig('K-Means with PCA.png')

# using the NMI and AMI scores to evaluate the cluster
pca_nmi_score = normalized_mutual_info_score(data_label, pca_clusters)
pca_ami_score = adjusted_mutual_info_score(data_label, pca_clusters)

# applyting the Sub-KMeans method to the data
skm = SubspaceKMeans(n_clusters=n_clusters, random_state=14)
skm_clusters = skm.fit_predict(normed_data_features)

# visualizing the cluster space using only two features
transformed_data_features = pd.DataFrame(skm.transform(normed_data_features), columns=['x%02d' % x for x in range(1, n_features + 1)])
ax = None
for c in range(n_clusters):
    ax = transformed_data_features.iloc[list(np.where(np.array(data_label) == c)[0]), :].plot(kind='scatter', x='x01', y='x02',
                    color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax)
plt.legend(loc=4, bbox_to_anchor=(0.28, 0))
# plt.legend(loc=4, bbox_to_anchor=(0.52, 0.6))
plt.title('Ground truth of cluster space features')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
savefig('Ground truth Sub-Kmeans cluster space.png')

# visualizing the Sub-KMeans predicted clusters
ax = None
for c in range(n_clusters):
    ax = transformed_data_features.iloc[list(np.where(np.array(skm_clusters) == c)[0]), :].plot(kind='scatter', x='x01', y='x02',
            color=sns.color_palette('husl', 4)[c], label='cluster %d' % c, ax=ax)
plt.legend(loc=4, bbox_to_anchor=(0.52, 0.6))
plt.title('Sub-KMeans cluters of cluster space')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
savefig('Sub-KMeans cluters space .png')

# visualizing the noise
noise_tsne = TSNE(n_components=2, random_state=14)
visualized_noise_data_features = pd.DataFrame(noise_tsne.fit_transform(transformed_data_features.iloc[:, skm.m_:]),
                                              columns=['x%02d' % x for x in range(1, 3)])
ax = None
for c in range(n_clusters):
    ax = visualized_noise_data_features.iloc[list(np.where(np.array(data_label) == c)[0]), :].plot(kind='scatter', x='x01', y='x02',
                                    color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax)
plt.legend(loc=4, bbox_to_anchor=(0.28, 0.5))
plt.title('Ground truth of noise space features')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.tight_layout()
savefig('iris_noise_space_GT.png')

ax = None
for c in range(n_clusters):
    ax = visualized_noise_data_features.iloc[
        list(np.where(np.array(skm_clusters) == c)[0]), :].plot(kind='scatter', x='x01', y='x02',
                        color=sns.color_palette('husl', 4)[c], label='class %d' % c, ax=ax)
plt.legend(loc=4, bbox_to_anchor=(0.28, 0.5))
plt.title('Sub-KMeans cluters of noise space')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.tight_layout()
savefig('iris_noise_space_SKM.png')

# using the NMI and AMI scores to evaluate the cluster
skm_nmi_score = normalized_mutual_info_score(data_label, skm_clusters)
skm_ami_score = adjusted_mutual_info_score(data_label, skm_clusters)

print 