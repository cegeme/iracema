from iracema.core.segment import Segment


def get_notes_list(audio, onsets, offsets):
    """
    Generate a list of note segments using the specified `onsets` and `offsets`
    arrays.

    Args
    ----
    audio : Audio
        Audio object
    onsets : array
        Indexes of the onset occurrences in `audio`.
    offsets : array
        Indexes of the offset occurrences in `audio`.

    Return
    ------
    notes : list
        List of segments.
    """
    if onsets.shape != offsets.shape:
        raise ValueError("the number of onsets and offsets must the same")

    return [Segment(audio, onsets[i], offsets[i]) for i in range(len(onsets))]
