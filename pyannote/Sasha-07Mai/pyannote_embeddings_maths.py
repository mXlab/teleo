from pyannote_embeddings_familiarity import familiarity  , known_data
from pyannote_embeddings_familiarity import assess_familiarity, familiarity_scores


def compute_voice(x, embeddings, labels, k=20):
    # Compute distances
    similarities = 1 - cdist(x, embeddings, metric='cosine').flatten()

    # Sort distances ascendingly along each row
    nearest_similarity = np.sort(similarities, axis=1)[:, -k_nearest_neighbors:]  # take k smallest distances

    # Get the indices of the top-k most similar embeddings
    top_k_indices = np.argsort(similarities)[-k:][::-1]  # Sort in descending order

    # Get the labels of the top-k neighbors
    top_k_labels = labels[top_k_indices]

    sum_f = 0
    sum_trust = 0
    sum_f_trust = 0
    for i in top_k_indices:
        xi = embeddings[i]
        label = labels[i]
        f = np.max(similarities[i], 0)  # Use the similarity score as familiarity
        t = trust(label)  # Use the trust function
        f_t = f * t
        sum_f += f
        sum_trust += t
        sum_f_trust += f_t

        w_trust = sum_f_trust / sum_f 
    
    return w_trust
    # Calculate familiarity and trust


#  # Update the loop to correctly calculate f for each xi
#  for i in range(len(xi_list)):
#      xi = xi_list[i]  # Access the i-th nearest neighbor

#      f = familiarity_scores

#      t = trust(x, xi)  # Use the trust function
#      f_t = f * t
#      sum_f += f
#      sum_trust += t
#      sum_f_trust += f_t

#  # Calculate familiarity and trust
#  familiarity = sum_f / len(xi_list)
#  trust = sum_trust / len(xi_list)

#  # Handle division by zero for w_trust
#  w_trust = sum_f_trust / sum_f if sum_f != 0 else 0