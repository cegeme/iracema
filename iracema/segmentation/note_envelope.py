"""
Note envelope segmentation methods.
"""
import librosa
import numpy as np

import iracema.features
import iracema.spectral
import iracema.pitch

from iracema.core.point import Point
from iracema.core.segment import Segment


def segment_notes(audio, onsets, window=1024, hop=441):
    """
    Extract the note envelope for each note in the excerpt.

    Estimate the following points, whitin each two adjacent onsets:
    end of attack, start of release, offset.

    Arguments
    ---------
    audio : ir.Audio
        Audio object.
    onsets : ir.PointList
        Point list containing the instants of note onset.
    window : int
        Window length.
    hop : int
        Hop length.

    Return
    ------
    note_list: list
        A list of dictionaries with the keys 'onset', 'attack_end',
        'release_start' and 'offset', containing the points that
        correspond to those instants.
    """
    stft = iracema.spectral.STFT(audio, window, hop)
    pitch = iracema.pitch.crepe(audio, min_confidence=0)
    pitch.pad(0, stft.nsamples - pitch.nsamples)

    pitch_midi = pitch.copy()
    pitch_midi.data = librosa.hz_to_midi(pitch.data)
    pitch_midi.unit = 'note'
    pitch_diff = pitch_midi.diff()
    rms = iracema.features.rms(audio, window, hop)
    spf_diff = iracema.features.spectral_flux(stft)

    note_list = []
    for onset_0, onset_1 in zip(onsets[0:-1], onsets[1:]):
        ioi = Segment(onset_0, onset_1)
        this_spf_dif = spf_diff[ioi]

        # release
        release_start = (
            onset_0 + (np.argmax(this_spf_dif.data) / this_spf_dif.fs)
        )

        this_rms = rms[ioi]
        this_pitch_diff = pitch_diff[ioi]

        release_start_point = Point(release_start)

        # attack
        seg_onset_release = Segment(onset_0, release_start)
        rms_onset_release = this_rms[seg_onset_release]
        if rms_onset_release.nsamples <= 1:
            attack_end = release_start
        else:
            attack_end = (
                onset_0 + (np.argmax(rms_onset_release.data) / this_rms.fs)
            )

        attack_end_point = Point(attack_end)

        # offset
        seg_release_end_ioi = Segment(release_start, onset_1)
        pitch_diff_end = this_pitch_diff[seg_release_end_ioi]

        pitch_change_idxs = np.where(pitch_diff_end.data > 0.5)[0]
        if not np.any(pitch_change_idxs):
            offset = onset_1
        else:
            idx = pitch_change_idxs[0] - 1
            offset = release_start +(idx / this_pitch_diff.fs)

        offset_point = Point(offset)

        # save points to dictionary list
        points = [attack_end_point, release_start_point]
        if offset_point != onset_1:
            points.append(offset_point)

        note = {
            'onset': onset_0,
            'attack_end': attack_end_point,
            'release_start': release_start_point,
            'offset': offset_point
        }

        note_list.append(note)
    
    return note_list
