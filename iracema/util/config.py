"""
This method implements the configuration loading funcionalities for iracema.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigLoader():
    """
    Configuration loader class.
    """
    samples_path: str
    _samples_path: str = field(init=False, repr=False)

    @property
    def samples_path(self):  # pylint: disable=function-redefined
        """
        Path to the directory where the sample audio files should be located.
        """
        return self._samples_path

    @samples_path.setter
    def samples_path(self, path: str):
        self._samples_path = os.path.abspath(Path(path))

    @classmethod
    def get_default_config_loader(cls, iracema_root):
        """
        Factory method to instantiate a default config object.
        """
        relative_samples_path = Path('..', 'audio', 'iracema-audio')
        return cls(Path(iracema_root / relative_samples_path))
