.. figure:: img/iracema-logo.png
  :width: 60%
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

   git clone --recurse-submodules https://github.com/cegeme/iracema.git


The command shown above will clone the project including some example audio
files. If you don't want to download those files, you should omit the
parameter ``--recurse-submodules``, like this:

.. code-block:: bash

   git clone https://github.com/cegeme/iracema.git


2. We strongly recommend that you create a virtual environment to install the
   dependencies for iracema, since it is always a good practice to keep
   project-specific dependencies isolated from your base Python installation
   (see the instructions bellow). If you have already created and activated 
   the virtual environment, you may procceed to install the required dependencies.
   In the directory where the repository was cloned, type the following command:

.. code-block:: bash

   pip3 install -r requirements.txt
   pip3 install -e .


4. In order to play audio you will need to manually install an audio I/O library
   called PortAudio. In Debian / Ubuntu you can install it using apt:

.. code-block:: bash

   sudo apt install libportaudio2


Virtual environment (venv)
==========================

To create a virtual environment to use iracema, go to the project's folder
and use the command

.. code-block:: bash

   python3 -m venv venv

A folder called `venv` will be created, where you will be able to install
all the project's dependencies, isolated from your base Python installation.
To activate this newly created environment, type the following command:

.. code-block:: bash

   source venv/bin/activate


Dependencies
============

- Python packages (required):

  * numpy
  * scipy
  * matplotlib
  * audioread

- Optional dependencies:

  * ffmpeg, libav, gstreamer or core audio (for opening different audio file
    formats)
  * CFFI, sounddevice and libportaudio2 (only if you want to play audio)

- To compile the docs:

  * sphinx
  * sphinxcontrib-napoleon
  * sphinx-rtd-theme

