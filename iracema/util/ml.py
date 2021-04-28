"Useful methods for ML models."
import warnings

import numpy as np
import scipy.signal

import iracema as ir


def points_to_activations(point_list,
                          output_fs,
                          output_length,
                          encoding='binary',
                          warn_lost_onset=True,
                          activation_type='single',
                          triang_value=0.4):
    """
    Convert a point list to an array with the corresponding activations.

    Arguments
    ---------
    point_list : PointList
        List of points
    output_fs : float
        Sampling frequency for the output array.
    output_length : int
        Length of the output array.
    encoding : str
        Type of encoding for the activations:
        - 'binary': a single activation per time step, with 0 representing
          one class and 1 the other.
        - 'onehot': two complementary activations per time step. The first
          correspond to class 1, and the second to class 0.
    warn_lost_onset : bool
        If True, the method will throw a warning whenever an onset is lost due
        to the resolution of the output array. This situation will happen when
        two or more onsets are closer than the resolution of the output array.
    activation_type : str
        Type of activation function to be generated:
        - 'single' will yield single activation points equal to ``1.0`` for the
           given time instants;
        - 'triangular' will yield a triangular activation around the point
           corresponding to each time instant.
    """
    _validate_encoding(encoding)

    time_array = point_list.to_numpy()
    activations = time_array_to_activations(time_array,
                                            output_fs,
                                            output_length,
                                            warn_lost_onset=warn_lost_onset,
                                            activation_type=activation_type,
                                            triang_value=triang_value)
    if encoding == 'onehot':
        activations = binary_to_onehot_activations(activations)
    elif encoding == 'binary':
        activations = np.expand_dims(activations, axis=-1)
    return activations


def activations_to_points(activations, input_fs, encoding='binary', threshold=0.5, peak_pick=False):
    """
    Convert an array with activations to a point list.

    Arguments
    ---------
    activations : np.array
        Activations array.
    encoding : str
        Type of encoding for the activations:
        - 'binary': a single activation per time step, with 0 representing
          one class and 1 the other.
        - 'onehot': two complementary activations per time step. The first
          represent the probability of class 0, and the second the probability
          of class 1. The second values are equivalent to the values produced
          using 'binary' encoding.
    input_fs : float
        Sampling frequency of the activations array.
    threshold : float
        Activation threshold to add a time instant to the output array.
    """
    _validate_encoding(encoding)
    if encoding == 'onehot':
        activations = onehot_to_binary_activations(activations)
    elif encoding == 'binary':
        activations = activations.squeeze()
    time_array = activations_to_time_array(activations, input_fs, threshold=threshold, peak_pick=peak_pick)
    points = ir.PointList.from_numpy(time_array)
    return points


def time_array_to_activations(time_array,
                              output_fs,
                              output_length,
                              warn_lost_onset=True,
                              activation_type='single',
                              triang_value=0.4):
    """
    Convert an array of times to an array of activations.

    Arguments
    ---------
    time_array : np.array
        List of time instants.
    output_fs : float
        Sampling frequency for the output array.
    output_length : int
        Length of the output array.
    warn_lost_onset : bool
        If True, the method will throw a warning whenever an onset is lost due
        to the resolution of the output array. This situation will happen when
        two or more onsets are closer than the resolution of the output array.
    activation_type : str
        Type of activation function to be generated:
        - 'single' will yield single activation points equal to ``1.0`` for the
           given time instants;
        - 'triangular' will yield a triangular activation around the point
           corresponding to each time instant.
        - 'gaussian' will yield a gaussian activation around the point
           corresponding to each time instant.
    """
    _validate_activation_type(activation_type)

    time_indexes = np.round(time_array.astype(float) *
                            float(output_fs)).astype(int)
    activations = np.zeros(output_length)

    if activation_type == 'triangular':
        time_indexes_last = np.clip(time_indexes - 1, 0, None)
        time_indexes_next = np.clip(time_indexes + 1, None, output_length - 1)
        neighbours = np.append(time_indexes_last, time_indexes_next)
        neighbours = np.unique(neighbours)
        activations[neighbours] = triang_value
    elif activation_type == 'gaussian':
        activations = _activations_to_gaussian(activations)

    activations[time_indexes] = 1.

    num_onsets_original = len(time_array)
    num_onsets_final = np.sum(activations == 1.)
    if warn_lost_onset and (num_onsets_final != num_onsets_original):
        onsets_lost = num_onsets_original - num_onsets_final
        onsets_lost_msg = (
            f"Lost {onsets_lost} onsets during conversion from point list to "
            "probability array. The distances between succesive onsets are "
            "probably too small for the sampling frequency being used.")
        warnings.warn(onsets_lost_msg)

    return activations


def activations_to_time_array(activations, input_fs, threshold=0.5, peak_pick=False):
    """
    Convert an array of activations to an array of times.

    Arguments
    ---------
    activations : np.array
        Activations array.
    input_fs : float
        Sampling frequency of the activations array.
    threshold : float
        Activation threshold to add a time instant to the output array.
    """
    if peak_pick:
        activation_indexes, _ = scipy.signal.find_peaks(activations, height=threshold)
        time_array = activation_indexes.astype(float) / float(input_fs)
    else:
        time_array = np.where(activations > threshold)[0] / float(input_fs)
    return time_array


def binary_to_onehot_activations(activations):
    """
    Convert an array containing a single activation per time step to a
    bidimensional array, containing two complementary activations per time
    step: the first corresponds to the probability of non-onset, and the
    second to the probability of onset.
    """
    if activations.ndim != 1:
        raise ValueError(
            "The array `activations` must have only one dimension.")
    activations = np.expand_dims(activations, axis=-1)
    activations = np.repeat(activations, 2, axis=-1)
    activations[..., 1] = 1 - activations[..., 1]  # non-onset
    return activations


def onehot_to_binary_activations(activations):
    """
    Convert a bidimensional array containing two activations per time step to
    an unidimensional array, containing a single activation per time step,
    representing the probability of onset.
    """
    activations = np.squeeze(activations)
    if activations.ndim != 2:
        raise ValueError("The array `activations` must have two dimensions "
                         "with more than one element.")
    activations = activations[..., 0]  # onset
    return activations


def _activations_to_gaussian(activations, std=0.6, length=5):
    g = scipy.signal.windows.gaussian(length, std=std)
    half = length // 2
    activations_gaussian = np.zeros_like(activations)
    for ix in np.where(activations==1.)[0]:
        span = len(activations_gaussian[ix-half:ix-half+length])
        activations_gaussian[ix-half:ix-half+length] = \
            np.clip(activations_gaussian[ix-half:ix-half+length] + g[0:span], None, 1.)
    return activations_gaussian


def _validate_activation_type(activation_type):
    if activation_type not in ('single', 'triangular', 'gaussian'):
        raise ValueError('Invalid value for `activation_type`.')


def _validate_encoding(encoding):
    if encoding not in ('binary', 'onehot'):
        raise ValueError('Invalid value for argument `encoding`.')
