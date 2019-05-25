"""
Default recipes to run Iracema.
"""


class Run:
    """
    Execute the iracema algorithms for a set of audio files.
    """

    def process_directory(self):
        """
        Load and process all the audio files within the specified directory.
        """

        pass

    def process_file(
            self,
            filepath,
            min_ioi=0.05,
            min_rms=0.005,
            ws=1024,
            hs=256,
            fft_len=None,
            instrument=None,
    ):
        """
        Load and process the specified audio file.

        Arguments
        ---------
        filepath: str
            Path and name of the audio file to be loaded.
        ws: int
            Default window size.
        hs: int
            Default hop size.
        min_ioi: float
            Inter-onset interval threshold.
        min_rms: float
            RMS threshold for the onset detection (avoids background noise).
        instrument: string
            Name of the musical instrument in the recording
        caption: str
            Text description of the file (for plotting purposes)
        """

        pass
