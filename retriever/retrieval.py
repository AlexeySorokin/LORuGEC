import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from kmedoids import KMedoids
from faiss import normalize_L2
from tqdm.auto import tqdm

def extract_encoder_states_to_save(hidden_states, probs, keep_index, n_non_keep=3, min_prob=0.1, 
                                   n_keep=0, random_state=189, bucket_size=5000):
    vectors, indexes, positions = [], [], []
    keep_vectors, keep_indexes, keep_positions = [], [], []
    for i, (curr_states, curr_probs) in enumerate(zip(hidden_states, probs)):
        keep_probs = curr_probs[:,keep_index]
        order = np.argsort(keep_probs)
        for r, pos in enumerate(order[:n_non_keep]):
            if r > 0 and curr_probs[pos, keep_index] > 1.0-min_prob:
                break
            vectors.append(curr_states[pos])
            indexes.append(i)
            positions.append(pos)
        curr_keep_positions = np.where(keep_probs > 1.0-min_prob)[0]
        keep_vectors.extend(curr_states[curr_keep_positions])
        keep_indexes.extend([i]*len(curr_keep_positions))
        keep_positions.extend(curr_keep_positions)
    keep_vectors, keep_indexes, keep_positions = np.array(keep_vectors), np.array(keep_indexes), np.array(keep_positions)
    are_vectors_non_keep = [1] * len(keep_vectors)
    if n_keep > 0:
        kmedoids_learner = KMedoids(n_clusters=int(len(hidden_states)*n_keep), random_state=random_state)
        for r, bucket_start in enumerate(range(0, len(keep_vectors), bucket_size)):
            print(f"Bucket {r} started")
            distances = euclidean_distances(keep_vectors[bucket_start:bucket_start+bucket_size])
            kmedoids_result = kmedoids_learner.fit(distances)
            indexes.extend(list(keep_indexes[bucket_start+kmedoids_result.medoid_indices_]))
            positions.extend(list(keep_positions[bucket_start+kmedoids_result.medoid_indices_]))
            vectors.extend(list(keep_vectors[bucket_start+kmedoids_result.medoid_indices_]))
            are_vectors_non_keep.extend([0]*len(kmedoids_result.medoid_indices_))
            print(f"Bucket {r} processed")
    return np.array(indexes), np.array(positions), np.array(vectors), are_vectors_non_keep

def find_closest_sentences(hidden_states, probs, storage, storage_sentences, 
                           keep_index, n_closest=10, n_neighbours=10, 
                           source_sentences=None, allow_same=True,
                           cosine=False, n_non_keep=3,  min_prob=0.1,
                           to_return_indexes=False, bucket_size=10000):
    answer, positions_answer = [], []
    to_search, non_keep_positions, offsets, bucket_start = [], [], [0], 0
    for i, (curr_states, curr_probs) in enumerate(zip(tqdm(hidden_states), probs)):
        keep_probs = curr_probs[:,keep_index]
        curr_non_keep_positions = np.where(keep_probs <= 1.0-min_prob)[0]
        if len(curr_non_keep_positions) < n_non_keep:
            curr_non_keep_positions = np.argsort(keep_probs)[:n_non_keep]
        to_search.extend(curr_states[curr_non_keep_positions])
        non_keep_positions.append(curr_non_keep_positions)
        offsets.append(offsets[-1]+len(curr_non_keep_positions))
        if len(to_search) >= bucket_size:
            to_search = np.asarray(to_search)
            if cosine:
                normalize_L2(to_search)
            distances, indices = storage.search(to_search, n_neighbours)
            for j, (offset, curr_non_keep_positions) in enumerate(zip(offsets, non_keep_positions)):
                curr_answer, curr_positions_answer, curr_indexes_answer = dict(), [], []
                flat_distances, flat_indices = np.ravel(distances[offset:offsets[j+1]]), np.ravel(indices[offset:offsets[j+1]])
                order = np.argsort(flat_distances, axis=None)
                for distance, sentence_index, pos in zip(flat_distances[order], flat_indices[order], curr_non_keep_positions[order // n_neighbours]):
                    sentence = storage_sentences[sentence_index]
                    if not allow_same and source_sentences is not None and source_sentences[bucket_start+j] == sentence:
                        continue
                    if isinstance(sentence, list):
                        sentence = " ".join(sentence)
                    if sentence not in curr_answer:
                        curr_answer[sentence] = distance
                        curr_positions_answer.append(pos)
                        curr_indexes_answer.append(sentence_index)
                        if len(curr_answer) >= n_closest:
                            break
                answer.append(sorted(curr_answer.items(), key=lambda x: x[1]))
                positions_answer.append((curr_positions_answer, curr_indexes_answer))
            to_search, non_keep_positions, offsets, bucket_start = [], [], [0], i+1
    if len(to_search) > 0:
        to_search = np.asarray(to_search)
        if cosine:
            normalize_L2(to_search)
        distances, indices = storage.search(to_search, n_neighbours)
        for i, (offset, curr_non_keep_positions) in enumerate(zip(offsets, non_keep_positions)):
            curr_answer, curr_positions_answer, curr_indexes_answer = dict(), [], []
            flat_distances, flat_indices = np.ravel(distances[offset:offsets[i+1]]), np.ravel(indices[offset:offsets[i+1]])
            order = np.argsort(flat_distances, axis=None)
            for distance, sentence_index, pos in zip(flat_distances[order], flat_indices[order], curr_non_keep_positions[order // n_neighbours]):
                sentence = storage_sentences[sentence_index]
                if not allow_same and source_sentences is not None and source_sentences[i] == sentence:
                    continue
                if isinstance(sentence, list):
                    sentence = " ".join(sentence)
                if sentence not in curr_answer:
                    curr_answer[sentence] = distance
                    curr_positions_answer.append(pos)
                    curr_indexes_answer.append(sentence_index)
                    if len(curr_answer) >= n_closest:
                        break
            answer.append(sorted(curr_answer.items(), key=lambda x: x[1]))
            positions_answer.append((curr_positions_answer, curr_indexes_answer))
    return (answer, positions_answer) if to_return_indexes else answer

def make_retrieval_train_indexes(corr_labels, pred_labels, positions_answer, indexes, positions,
                                 max_num_samples=1, default_positive_indexes=None, all_corr_labels=None):
    answer = []
    for i, (corr_label, curr_pred_labels) in enumerate(zip(corr_labels, pred_labels)):
        curr_positive_labels = all_corr_labels[i] if all_corr_labels is not None else [corr_label]
        curr_positions_answer, curr_indexes_answer = positions_answer[i]
        if corr_label not in curr_pred_labels:
            if default_positive_indexes is None:
                continue
            position_query_positive, index_positive, position_positive = default_positive_indexes[i]
            selected_pairs = 0
            for j, pred_label in enumerate(curr_pred_labels):
                if pred_label not in curr_positive_labels:
                    position_query_negative = int(curr_positions_answer[j])
                    index_negative, position_negative = int(indexes[curr_indexes_answer[j]]), int(positions[curr_indexes_answer[j]])
                    answer.append({
                        "index": i, 
                        "position_query_positive": position_query_positive, "position_query_negative": position_query_negative,
                        "index_positive": index_positive, "position_positive": position_positive,
                        "index_negative": index_negative, "position_negative": position_negative,
                    })
                selected_pairs += 1
                if selected_pairs >= max_num_samples:
                    break
        else:
            corr_label_index = curr_pred_labels.index(corr_label)
            wrong_indexes = [j for j, pred_label in enumerate(curr_pred_labels) if pred_label not in curr_positive_labels]
            for wrong_index in wrong_indexes[:max_num_samples]:
                pos_index, neg_index = curr_indexes_answer[corr_label_index], curr_indexes_answer[wrong_index]
                position_query_positive, position_query_negative = curr_positions_answer[corr_label_index], curr_positions_answer[wrong_index]
                index_positive, position_positive = indexes[pos_index], positions[pos_index]
                index_negative, position_negative = indexes[neg_index], positions[neg_index]
                # assert (position_query_positive, index_positive, position_positive == default_positive_indexes[i])
                answer.append({
                    "index": i, 
                    "position_query_positive": position_query_positive, "position_query_negative": position_query_negative,
                    "index_positive": index_positive, "position_positive": position_positive,
                    "index_negative": index_negative, "position_negative": position_negative,
                })
    return answer






    