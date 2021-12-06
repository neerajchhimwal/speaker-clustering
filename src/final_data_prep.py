def get_final_clusters_and_noise(big_clusters_indices, all_cluster_embeds, mean_embeds, noise_embeds, big_cluster_embeds, big_mean_embeds, noise_embeds_big):
    # sanity check and final data prep

    all_cluster_embeds_minus_big = [em_list for i, em_list in enumerate(all_cluster_embeds) if i not in big_clusters_indices]
    mean_embeddings_minus_big = [mean_em for i, mean_em in enumerate(mean_embeds) if i not in big_clusters_indices]

    all_cluster_embeds_to_merge = []
    all_cluster_embeds_to_merge.extend(all_cluster_embeds_minus_big)
    all_cluster_embeds_to_merge.extend(big_cluster_embeds)

    mean_embeddings_to_merge = []
    mean_embeddings_to_merge.extend(mean_embeddings_minus_big)
    mean_embeddings_to_merge.extend(big_mean_embeds)

    # noise
    noise_embeds_final = []
    noise_embeds_final.extend(noise_embeds)
    noise_embeds_final.extend(noise_embeds_big)

    return all_cluster_embeds_to_merge, mean_embeddings_to_merge, noise_embeds_final