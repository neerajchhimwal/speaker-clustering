from src.create_file_mappings import Map
from src.clustering import Clustering
import numpy as np

from src.merging import Merge
from src.splitting import get_big_cluster_embeds
from src.final_data_prep import get_final_clusters_and_noise
from src.save_file_cluster_mapping import save_json


def create_speaker_clusters(
        embed_filename_map_path,
        source_name,
        min_cluster_size=4,
        partial_set_size=10000,
        min_samples=1,
        fit_noise_on_similarity=0.80,
        out_json='sp_clusters.json'
):
    # step:1 -> ENCODING AND SAVING : done by create_embeddings.py

    # step:2 -> CLUSTERING AND MAPPING FILES TO CLUSTERS
    embed_speaker_map = np.load(embed_filename_map_path)
    embeddings = embed_speaker_map['embeds']
    file_paths = embed_speaker_map['file_paths']

    # step:2.1 -> CLUSTERING ON PARTIAL SETS
    clustering_obj = Clustering()
    mean_embeds, noise_embeds, all_cluster_embeds = clustering_obj.run_partial_set_clusterings(embeddings=embeddings,
                                                                                               min_cluster_size=min_cluster_size,
                                                                                               partial_set_size=partial_set_size,
                                                                                               min_samples=min_samples)
    num_clusters = len(mean_embeds)
    print('Num clusters before merging/splitting= {}'.format(num_clusters))
    all_cluster_embeds_after_noise_fit = []
    unallocated_noise_embeds = []
    if num_clusters >= 1:
        # step:2.2 -> APPLYING MERGING OVER SIMILAR CLUSTERS FROM PARTIAL SETS CLUSTERS

        merger = Merge()
        all_cluster_embeds_merged_initial, mean_embeds_merged_initial = merger.run_repetitive_merging(
            all_cluster_embeds,
            mean_embeds,
            start_similarity_allowed=0.96,
            end_similarity_allowed=0.94,
            merge_closest_only=True)

        num_clusters = len(mean_embeds_merged_initial)
        print('Num clusters after initial merging= {}'.format(num_clusters))

        # step:2.3 -> SPLITTING "BIG" CLUSTERS AND MERGING AGAIN
        flat_embeddings_big_clusters, big_clusters_indices = get_big_cluster_embeds(all_cluster_embeds_merged_initial)

        if big_clusters_indices:
            mean_embeds_big, noise_embeds_big, all_cluster_embeds_big = clustering_obj.run_partial_set_clusterings(
                embeddings=flat_embeddings_big_clusters,
                min_cluster_size=min_cluster_size,
                partial_set_size=partial_set_size,
                min_samples=min_samples,
                cluster_selection_method='leaf')
            if len(mean_embeds_big) == 1:
                if len(all_cluster_embeds_big) != 1:
                    all_cluster_embeds_big = [all_cluster_embeds_big]
            big_cluster_embeds_merged = []
            big_mean_embeds_merged = []
            if len(mean_embeds_big) != 0:
                merger_big_cl = Merge()
                big_cluster_embeds_merged, big_mean_embeds_merged = merger_big_cl.run_repetitive_merging(
                    all_cluster_embeds_big,
                    mean_embeds_big,
                    start_similarity_allowed=0.96,
                    end_similarity_allowed=0.94,
                    merge_closest_only=True)

                print('Num clusters after merging big clusters = {}'.format(len(big_mean_embeds_merged)))

            # preparing new list of clusters (after adding split+merged clusters together) and updated final noise
            all_cluster_embeds_to_merge, mean_embeddings_to_merge, noise_embeds_final = get_final_clusters_and_noise(
                big_clusters_indices, all_cluster_embeds_merged_initial, mean_embeds_merged_initial,
                noise_embeds, big_cluster_embeds_merged, big_mean_embeds_merged, noise_embeds_big)
            print('Num clusters before final merging  = {}'.format(len(mean_embeddings_to_merge)))
            print('Num final noise points = {}'.format(len(noise_embeds_final)))

            # step:2.4 -> repetitive merging on the final clusters from step 2.3
            merger_final = Merge()
            all_cluster_embeds_merged, mean_embeds_merged = merger_final.run_repetitive_merging(
                all_cluster_embeds_to_merge,
                mean_embeddings_to_merge,
                start_similarity_allowed=0.96,
                end_similarity_allowed=0.94,
                merge_closest_only=True)
            print('Num clusters after final merging = {}'.format(len(mean_embeds_merged)))

            # step:3 -> FIT NOISE
            all_cluster_embeds_after_noise_fit, mean_embeds_new, unallocated_noise_embeds = merger.fit_noise_points(
                mean_embeds_merged,
                noise_embeds_final,
                all_cluster_embeds_merged,
                max_sim_allowed=fit_noise_on_similarity)

        else:
            # step:3 -> FIT NOISE
            if noise_embeds:
                all_cluster_embeds_after_noise_fit, mean_embeds_new, unallocated_noise_embeds = merger.fit_noise_points(
                    mean_embeds_merged_initial,
                    noise_embeds,
                    all_cluster_embeds_merged_initial,
                    max_sim_allowed=fit_noise_on_similarity)

    # step:4 -> SAVE FILE_NAMES TO CLUSTER MAPPINGS
    print('Creating mappings for files')
    map_obj = Map(embeddings, file_paths)
    indices = [map_obj.find_index(cluster) for cluster in all_cluster_embeds_after_noise_fit]
    # indices = [map_obj.find_index(cluster) for cluster in all_cluster_embeds_merged_initial]
    files_in_clusters = [map_obj.find_file(row) for row in indices]
    file_map_dict = {source_name + '_sp_' + str(ind): j for ind, j in enumerate(files_in_clusters)}

    noise_file_map_dict = dict({})
    if unallocated_noise_embeds:
        print('Creating mapping for noise points')
        noise_indices = [map_obj.find_index(cluster) for cluster in [unallocated_noise_embeds]]
        noise_files = [map_obj.find_file(row) for row in noise_indices]
        noise_file_map_dict = {source_name + '_noise': j for ind, j in enumerate(noise_files)}

    print('len of clusters:')
    for i in file_map_dict.values():
        print(len(i))
    save_json(out_json, file_map_dict)
    save_json('noise_'+out_json, noise_file_map_dict)
    return file_map_dict, noise_file_map_dict


if __name__ == "__main__":
    file_map_dict, noise_file_map_dict = create_speaker_clusters(
        embed_filename_map_path='file.npz',
        source_name='source')

# '/Users/neerajchhimwal/ekstep-speech-recognition/ekstep-speaker-clustering/Namma_Adige/Namma_Adige_embed_file.npz'
