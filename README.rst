.. figure:: img/iracema-logo.png
  :width: 50%
  :alt: Iracema

############
Introduction
############

Iracema is a Python package aimed at the empirical research on music
performance, with focus on the analysis of expressiveness and individuality
from audio recordings. It is developed and maintaned by researchers at
CEGeME_, and contains computational models of music information extraction
that were developed for supporting research projects in music performance. It
was strongly inspired by Expan, a Matlab tool that had been previously
developed at CEGeME.

.. _CEGeME: http://musica.ufmg.br/cegeme


************
Installation
************

We strongly recommend that you install iracema into a separate virtual environment,
since it is always a good practice to keep project-specific dependencies isolated
from your base Python installation (check instructions in the section Virtual
environment). After activating your virtual environment, you can install iracema
by running the following command:

.. code-block:: bash

   pip3 install iracema


If you're a Linux user, you will need to manually install an audio I/O library 
called PortAudio. If you are a MacOS X user, this library is probably already
installed. In Debian / Ubuntu you can install it using apt:

.. code-block:: bash

   sudo apt install libportaudio2


Virtual environment
===================

To create a virtual environment to use iracema, go to the project's folder
and run:

.. code-block:: bash

   python3 -m venv venv

A folder called `venv` will be created, where you will be able to install
all the project's dependencies, isolated from your base Python installation.
To activate this newly created environment, type the following command:

.. code-block:: bash

   source venv/bin/activate

**********
Developing
**********

To contribute with the development of iracema, clone the repository from github:

.. code-block:: bash

   git clone --recurse-submodules https://github.com/cegeme/iracema.git


The command shown above will also clone some example audio files. 
If you don't want to download those files, you should omit the
parameter ``--recurse-submodules``, like this:

.. code-block:: bash

   git clone https://github.com/cegeme/iracema.git

  
To install the required dependencies and the cloned project in pip, go to the directory 
where the repository was cloned and type in your command line:

.. code-block:: bash

   pip3 install -r requirements.txt
   pip3 install -e .



Dependencies
============

- Python packages (required):

  * numpy
  * scipy
  * matplotlib
  * sounddevice
  * audioread

- Optional dependencies:

  * ffmpeg, libav, gstreamer or core audio (for opening different audio file
    formats)
  * libportaudio2 (if you want to play audio)
