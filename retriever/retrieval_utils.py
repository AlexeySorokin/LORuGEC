from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.metrics import pairwise_distances as pairwise_distances_func
from faiss import IndexFlatL2, normalize_L2

from collate import collate_fn
from retrieval import extract_encoder_states_to_save, find_closest_sentences, make_retrieval_train_indexes
from retrieval_common import make_retrieval_train_dataset
from evaluation import measure_metrics
from utils import predict_with_model


def prepare_storage(model, processed_data, tokenizer, keep_index,
                   classifier_model=None, to_prepare_dataset=False, cosine=False, 
                   max_num_samples=1, do_additional_search_for_positives=False,
                   max_non_keep_per_text=3, min_non_keep_prob=0.1,
                   text_column="text", label_column="class_labels",
                   labels_mask_column="labels_mask", all_labels_column=None,
                   retrieval_variant_keys=None,
                   verbose=True):
    model_collate_fn = partial(collate_fn, bert_tokenizer=tokenizer, device="cuda")
    predictions = predict_with_model(processed_data, model, collate_func=model_collate_fn, 
                                     return_probs=(classifier_model is None), output_hidden_states=True,
                                     labels_mask_column=labels_mask_column)
    if classifier_model is not None:
        classifier_predictions = predict_with_model(
            processed_data, model, collate_func=model_collate_fn, return_probs=True, 
            output_hidden_states=False, labels_mask_column=labels_mask_column)
    else:
        classifier_predictions = predictions    
    # test_data = {i: str(list(pred)) for i, pred in enumerate(predictions["labels"])}
    # if args.save_hidden_states is not None:
    indexes_for_states, positions, states, non_keep_flags = extract_encoder_states_to_save(
        predictions["hidden_states"], classifier_predictions["probs"], keep_index, 
        n_non_keep=max_non_keep_per_text, min_prob=min_non_keep_prob
    )
    train_data = [elem[text_column] for elem in processed_data]
    sentences_storage = [train_data[i] for i in indexes_for_states]
    # labels_storage =  [processed_data[int(i)][label_column] for i in indexes_for_states]
    vectors_storage = IndexFlatL2(len(states[0]))
    if cosine:
        normalize_L2(states)
    vectors_storage.add(np.array(states))
    if label_column is not None:
        train_labels = [elem[label_column] for elem in processed_data]
        train_labels_by_sents = dict(zip(train_data, train_labels))
        all_train_labels = [elem[all_labels_column] for elem in processed_data] if all_labels_column is not None else None
    if not to_prepare_dataset:
        to_return = (vectors_storage, sentences_storage)
        if label_column is not None:
            to_return += (train_labels_by_sents, )
        return to_return
    answer, positions_answer = find_closest_sentences(
        predictions["hidden_states"], predictions["probs"], vectors_storage, sentences_storage, 
        keep_index, cosine=cosine, source_sentences=train_data, allow_same=False,
        to_return_indexes=True
    )
    pred_labels = [[train_labels_by_sents[sent] for sent, d in elem] for elem in answer]
    if do_additional_search_for_positives:
        default_positive_indexes = [(None, None) for _ in train_data]
        indexes_by_classes = defaultdict(lambda: {"sentence_indexes": [], "state_indexes": []})
        for i, label in enumerate(train_labels):
            indexes_by_classes[label]["sentence_indexes"].append(i)
        for i, index in enumerate(indexes_for_states):
            indexes_by_classes[train_labels[index]]["state_indexes"].append(i)
        for label, curr_data in indexes_by_classes.items():
            curr_state_indexes = curr_data["state_indexes"]
            pairwise_distances = pairwise_distances_func(states[curr_state_indexes])
            state_order = np.argsort(pairwise_distances, axis=1)
            best_distances = dict()
            for i, (index, curr_state_order) in enumerate(zip(indexes_for_states[curr_state_indexes], state_order)):
                for close_state_index in curr_state_order:
                    real_close_state_index = curr_state_indexes[close_state_index]
                    close_index, dist = indexes_for_states[real_close_state_index], pairwise_distances[i, close_state_index]
                    if index in best_distances and best_distances[index] <= dist:
                        break
                    if close_index != index:
                        default_positive_indexes[index] = (positions[curr_state_indexes[i]], close_index, positions[real_close_state_index])
                        best_distances[index] = dist
                        break
    else:
        default_positive_indexes = None
    if verbose:  
        metrics = measure_metrics(train_labels, pred_labels, d=5)
        for key, value in metrics.items():
            print(f"{key} {100*value:.2f}", end="\t")
        print("")
    # if args.closest_sentences_outfile is not None:
    #     dump_sentences(train_data, answer, args.closest_sentences_outfile, sentence_labels=train_labels_by_sents)
    retrieval_train_indexes = make_retrieval_train_indexes(train_labels, pred_labels, positions_answer, indexes_for_states, positions,
                                                           max_num_samples=max_num_samples, default_positive_indexes=default_positive_indexes,
                                                           all_corr_labels=all_train_labels)
    retrieval_train_dataset = make_retrieval_train_dataset(processed_data, retrieval_train_indexes, variant_keys=retrieval_variant_keys)
    return vectors_storage, sentences_storage, train_labels_by_sents, answer, retrieval_train_dataset

def predict_closest_sentences(model, processed_data, vectors_storage, sentences_storage,
                              tokenizer, keep_index, cosine=False,  
                              text_column="text", labels_mask_column="labels_mask"):
    model_collate_fn = partial(collate_fn, bert_tokenizer=tokenizer, device="cuda")
    predictions = predict_with_model(processed_data, model, collate_func=model_collate_fn, 
                                     return_probs=True, output_hidden_states=True,
                                     labels_mask_column=labels_mask_column)
    test_data = [elem[text_column] for elem in processed_data]
    answer = find_closest_sentences(
        predictions["hidden_states"], predictions["probs"], vectors_storage, sentences_storage, 
        keep_index, cosine=cosine, source_sentences=test_data, allow_same=False,
        to_return_indexes=False
    )
    return answer

