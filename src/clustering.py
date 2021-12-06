import numpy as np
import hdbscan
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
import math


class Clustering:
    def __init__(self):
        pass

    def make_partial_sets(self, embeddings, partial_set_size):
        '''
        Takes all the embeddings and returns a list of partial sets
        :param embeddings: all embeddings
        :param partial_set_size: partial set embedding size
        :return: partial sets each of len(partial_set_size) from embeddings | list of np.array
        '''
        partial_sets = []
        embeds_shape = embeddings.shape[0]
        num_partial_sets = math.ceil(embeds_shape / partial_set_size)
        for i in range(num_partial_sets):
            start = i * partial_set_size
            end = (i + 1) * partial_set_size
            if i == num_partial_sets - 1 or embeds_shape < partial_set_size:
                end = embeds_shape + 1
            partial_set = embeddings[start:end][:]
            partial_sets.append(partial_set)

        return partial_sets

    def get_cluster_embeddings(self, embeddings, labels):
        '''
        Takes embeddings (np.array) and their corresponding labels (list),
        Returns a dictionary with cluster label as key and respective cluster embeddings as value
        '''
        cluster_vs_embeds = dict({})
        number_of_clusters = max(labels)
        start = 0
        if -1 in labels:
            start = -1
        for cluster in range(start, number_of_clusters + 1):
            cluster_indices = [index for index, _ in enumerate(labels) if _ == cluster]
            cluster_vs_embeds[cluster] = embeddings[cluster_indices]
        return cluster_vs_embeds

    def run_hdbscan(self, embeddings, metric, min_cluster_size, min_samples, cluster_selection_method):
        # because HDBSCAN expects double dtype
        embeddings = embeddings.astype('double')
        distance_matrix = cosine_distances(embeddings)

        clusterer = hdbscan.HDBSCAN(metric=metric,
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_method=cluster_selection_method)
        clusterer.fit(distance_matrix)

        return clusterer

    def run_partial_set_clusterings(self, embeddings, min_cluster_size=15, partial_set_size=11122, min_samples=15,
                                    cluster_selection_method='eom'):
        '''
        Runs HDBSCAN on partial sets of orginial data,
        Returns:
          - mean embeddings: np.ndarray -> mean embeds of each cluster found in each partial set
          - flat_noise_embeds: np.ndarray -> an array containing all the noise points found over all partial sets.
          - all_cluster_embeds: list of np.arrays ->

        '''

        noise = []
        mean_embeddings = []
        all_cluster_embeds = []
        if min_samples is None:
            min_samples = min_cluster_size

        partial_sets = self.make_partial_sets(embeddings, partial_set_size=partial_set_size)
        for ind, partial_set in enumerate(partial_sets):

            clusterer = self.run_hdbscan(partial_set, metric='precomputed', min_cluster_size=min_cluster_size,
                                         min_samples=min_samples, cluster_selection_method=cluster_selection_method)

            partial_set_labels = clusterer.labels_

            noise_point_embeds = [partial_set[index] for index, label in enumerate(partial_set_labels) if label == -1]
            noise.append(noise_point_embeds)

            print('Points classified as noise in partial set {} :{}'.format(ind + 1, len(noise_point_embeds)))

            # mapping contains cluster label as key and cluster embeddings as values
            mapping = self.get_cluster_embeddings(partial_set, partial_set_labels)

            # logic for calculating mean embedding of the cluster if cluster-label != -1 (noise)
            for i in mapping.items():
                if i[0] != -1:
                    raw_embed = np.mean(i[1], axis=0)
                    mean_embeddings.append(raw_embed / np.linalg.norm(raw_embed, 2))
                    all_cluster_embeds.append(list(i[1]))

        # getting flat noise embeds -> noise contains a list of numpy arrays : len(noise) = num_partial_sets
        flat_noise_embeds = [item for sublist in noise for item in sublist]

        return np.array(mean_embeddings), np.array(flat_noise_embeds), all_cluster_embeds
