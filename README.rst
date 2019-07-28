.. figure:: img/iracema-logo.png
  :width: 70%
  :alt: Iracema

############
Introduction
############

Iracema is a Python package aimed at the empirical research on music
performance, with focus on the analysis of expressiveness and individuality
from audio recordings. It is developed and maintaned by researchers at
|CEGeME|, and contains computational models of music information extraction
that were developed for supporting research projects in music performance. It
was strongly inspired by Expan, a Matlab tool that had been previously
developed at CEGeME.

.. |CEGeME| raw:: html

   <a href="http://musica.ufmg.br/cegeme" target="_blank">CEGeME</a>


**********
Installing
**********

Linux
=====

1. Clone the repository from github:


.. code-block:: bash

   git clone --recurse-submodules https://github.com/taironemagalhaes/iracema.git


The command shown above will clone the project including some example audio
files. If you don't want to download those files, you should omit the
parameter `--recurse-submodules`, like this:

.. code-block:: bash

   git clone https://github.com/taironemagalhaes/iracema.git


2. We strongly recommend that you create a virtual environment to install the
   dependencies for iracema, since it is always a good practice to keep 
   project-specific dependencies isolated from your base Python installation.
   To install the required dependecies using pip, simply go to the directory
   where the repository was cloned and type in your command line:

.. code-block:: bash

   pip install -r requirements.txt

3. In order to play audio you will need to manually install an audio I/O library
   called PortAudio. In Debian / Ubuntu you can install it using apt:

.. code-block:: bash

   sudo apt install libportaudio2

Dependencies
============

- Python packages (required):

  * numpy (1.15.0)
  * scipy (1.0.1)
  * matplotlib (???) 
  * audioread (2.1.4)

- Optional dependencies:

  * ffmpeg, libav, gstreamer or core audio (for opening different audio file
    formats)
  * CFFI, sounddevice and libportaudio2 (only if you want to play audio)

- To compile the docs:

  * sphinx (1.6.4)
  * sphinxcontrib-napoleon (0.6.1)
  * sphinx-rtd-theme (0.2.4)
