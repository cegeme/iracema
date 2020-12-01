import numpy as np


def evaluate_onsets(targets, predictions, tolerance=0.01):
    """
    Evaluation metrics for onset detection methods.

    Arguments
    ---------
    target : numpy array
        Target onsets (annotations).
    prediction : numpy array
        Predicted onsets (to be evaluated).
    tolerance : float
        Maximum time tolerance in seconds.

    Returns
    -------
    metrics : dict
        A dictionary with the calculated metrics.
    """
    # generate distances matrix between targets and predictions
    mesh = np.array(np.meshgrid(targets, predictions))
    combinations = mesh.T.reshape(-1, len(predictions), 2)
    distances = np.abs(np.diff(combinations, axis=2)).squeeze()

    # search for best matches within the tolerance distance
    target_found = np.full(len(targets), np.nan)
    while True:
        ix_min_dist = np.unravel_index(
            np.argmin(distances, axis=None), distances.shape)
        distance = distances[ix_min_dist]
        if np.isnan(target_found[ix_min_dist[0]]):
            target_found[ix_min_dist[0]] = ix_min_dist[1]
        distances[ix_min_dist] = np.inf
        if distance > tolerance:
            break

    arange_predictions = np.arange(0, len(predictions))
    true_positives = np.sum(~np.isnan(target_found))
    false_negatives = np.sum(np.isnan(target_found))
    false_positives = len(
        np.setdiff1d(
            arange_predictions,
            target_found[~np.isnan(target_found)],
            assume_unique=True))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    fmeasure = (2 * precision * recall) / (precision + recall)

    metrics = {
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'precision': precision,
        'recall': recall,
        'f-measure': fmeasure,
    }

    return metrics
