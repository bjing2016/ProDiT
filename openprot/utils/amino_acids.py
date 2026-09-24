import torch
import numpy as np

NUM_AMINO_ACIDS = 20


# compare generated aatypes to distribution from scope128 dataset (per Campbell et al. 2024)
def calc_aatype_distr(generated_aatypes):
    unique_aatypes, counts = np.unique(generated_aatypes, return_counts=True)

    counts_list = []
    for i in range(NUM_AMINO_ACIDS):
        if i in unique_aatypes:
            counts_list.append(counts[np.where(unique_aatypes == i)[0][0]])
        else:
            counts_list.append(0)  # did not get generated

    # from the scope128 dataset
    reference_normalized_counts = [
        0.0739,
        0.05378621,
        0.0410424,
        0.05732177,
        0.01418736,
        0.03995128,
        0.07562267,
        0.06695857,
        0.02163064,
        0.0580802,
        0.09333149,
        0.06777057,
        0.02034217,
        0.03673995,
        0.04428474,
        0.05987899,
        0.05502958,
        0.01228988,
        0.03233601,
        0.07551553,
    ]

    reference_normalized_counts = np.array(reference_normalized_counts)

    normalized_counts = counts_list / np.sum(counts_list)

    # compute the hellinger distance between the normalized counts
    # and the reference normalized counts
    hellinger_distance = np.sqrt(
        np.sum(
            np.square(np.sqrt(normalized_counts) - np.sqrt(reference_normalized_counts))
        )
    )

    return {"aatype_histogram_dist": hellinger_distance}
