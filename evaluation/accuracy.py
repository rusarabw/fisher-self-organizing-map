from collections import defaultdict

def evaluate_accuracy(som, X, y_true):
    """
    Map inputs to SOM and evaluate cluster accuracy by majority-vote labeling.
    """
    mapped = som.map_vects(X)
    bmu_labels = defaultdict(list)
    for coord, label in zip(mapped, y_true):
        bmu_labels[tuple(coord)].append(label)
    
    bmu_class = {coord: max(set(labels), key=labels.count) for coord, labels in bmu_labels.items()}
    correct = sum(1 for coord, label in zip(mapped, y_true) if bmu_class.get(tuple(coord)) == label)
    return correct / len(y_true)