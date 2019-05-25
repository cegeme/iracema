"""
Useful stuff for using iracema in jupyter.
"""

import IPython.display

def audio_player(audio):
    """
    """
    return IPython.display.Audio(data=audio.data, rate=audio.fs, autoplay=True)
