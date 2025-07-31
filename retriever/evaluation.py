import numpy as np


def measure_metrics(true_labels, pred_labels, d=5, evaluate_per_class=False):
    are_equal = [int(len(elem) > 0 and label == elem[0]) for label, elem in zip(true_labels, pred_labels)]
    ranks = [1+elem.index(label) if label in elem else -1 for label, elem in zip(true_labels, pred_labels)]
    inverse_ranks = [1.0/r if r > 0 else 0.0 for r in ranks]
    are_covered = [int(0 <= r <= d) for r in ranks]
    answer = {"accuracy": np.mean(are_equal), "mrr": np.mean(inverse_ranks), f"coverage_{d}": np.mean(are_covered)}
    if evaluate_per_class:
        per_class_data = {label: {"correct": 0, "mrr": 0, f"coverage_{d}": 0, "total": 0} for label in set(true_labels)}
        for is_equal, inverse_rank, is_covered, label in zip(are_equal, inverse_ranks, are_covered, true_labels):
            per_class_data[label]["correct"] += int(is_equal)
            per_class_data[label]["mrr"] += inverse_rank
            per_class_data[label][f"coverage_{d}"] += is_covered
            per_class_data[label]["total"] += 1
        for label, label_data in per_class_data.items():
            for key in ["correct", "mrr", f"coverage_{d}"]:
                label_data[key] /= label_data["total"]
        answer["per_class"] = per_class_data
    return answer


def measure_metrics_multilabel(true_labels, pred_labels, d=5, evaluate_per_class=False):
    are_equal = [int(any(x in elem[0] for x in label)) for label, elem in zip(true_labels, pred_labels)]
    ranks, inverse_ranks = [], []
    for elem, curr_pred_labels in zip(true_labels, pred_labels):
        curr_ranks = dict()
        for r, pred_elem in enumerate(curr_pred_labels, 1):
            for label in elem:
                if label not in ranks and label in pred_elem:
                    curr_ranks[label] = r
            if len(curr_ranks) == len(elem):
                break
        inverse_ranks.append(sum(1/r for r in curr_ranks.values()) / len(elem))
        ranks.append(curr_ranks)
    are_covered = [sum(int(0 <= r <= d) for r in elem.values())/max(len(elem), 1) for elem in ranks]
    answer = {"accuracy": np.mean(are_equal), "mrr": np.mean(inverse_ranks), f"coverage_{d}": np.mean(are_covered)}
    if evaluate_per_class:
        per_class_data = {label: {"correct": 0, "mrr": 0, f"coverage_{d}": 0, "total": 0} for label in set(true_labels)}
        for curr_labels, pred_labels in zip(true_labels, pred_labels):
            for label in curr_labels:
                per_class_data[label]["correct"] += int(label == pred_labels[0])
                rank = 1+pred_labels.index(label) if label in pred_labels else 0
                if rank > 0:
                    per_class_data[label]["mrr"] += 1.0 / rank
                    per_class_data[label][f"coverage_{d}"] += (rank <= d)
                per_class_data[label]["total"] += 1
        for label, label_data in per_class_data.items():
            for key in ["correct", "mrr", f"coverage_{d}"]:
                label_data[key] /= label_data["total"]
        answer["per_class"] = per_class_data
    return answer


def evaluate(test_labels, pred_labels, d=5, evaluate_per_class=False, multilabel=False, verbose=True):
    func = measure_metrics_multilabel if multilabel else measure_metrics
    metrics = func(test_labels, pred_labels, d=5, evaluate_per_class=evaluate_per_class)
    if verbose:
        for key, value in metrics.items():
            if "per_class" in key:
                if not evaluate_per_class:
                    continue
                print("")
                for subkey, subkey_data in value.items():
                    print(subkey, end="\t")
                    if isinstance(subkey_data, dict):
                        print("\t".join(f"{x}:{100*y:.2f}" if x != "total" else f"{x}:{y}" for x, y in subkey_data.items()))
                    else:
                        print(subkey_data)
            else:
                print(f"{key} {100*value:.2f}", end="\t")
        print("")
    return metrics