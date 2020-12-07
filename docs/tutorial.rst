========
Tutorial
========

This tutorial aims to introduce the basics of *iracema*'s architecture_ and usage_.

.. _usage:

-----
Usage
-----

This section presents a quickstart guide to *iracema*.

Import conventions
==================

Throughout this documentation, all the code examples will assume that iracema has been
imported using the following convention.

>>> import iracema as ir

Loading audio files
===================

To load an audio file, you must provide a string that specifies the location
where it is stored. Iracema accepts local paths to load files stored in your
computer or http URL to download remote files. The class ``ir.Audio``
provides the method ``load`` for this purpose.

.. plot::
   :include-source:
   :context: close-figs
   
   import iracema as ir
   audio = ir.Audio.load("05 - Trumpet - Haydn.wav")

To play the loaded audio:

.. code:: python
  
   audio.play()

The audio object has a plot method available that displays its waveform:

.. plot::
   :include-source:
   :context: close-figs

   audio.plot()

Calculating basic features
==========================

As most features will need an object containing an :abbr:`FFT (Fast Fourier
Transform)` as input, our next step is calculating it using the method
:obj:`ir.spectral.fft` for the loaded audio. You
must specify the sliding window and hop size values (in samples). After
calculating the FFT you're now able to plot a spectogram!

Other useful methods are RMS and Peak Envelope, which will be extracted and plotted in the example.

.. plot::
   :include-source:
   :context: close-figs
  
   # specifying window and hop sizes
   window, hop = 2048, 1024
  
   # calculating the FFT
   fft = ir.spectral.fft(audio, window, hop)
  
   # plotting the spectrogram
   ir.plot.spectrogram(fft)
  
   # calculating the RMS
   rms = ir.features.rms(audio, window, hop)
  
   # plotting the RMS
   rms.plot()

   # calculating the Peak Envelope
   peak = ir.features.peak_envelope(audio, window, hop)

   # plotting the Peak Envelope
   peak.plot()


Extracting pitch and harmonics
==============================

Another important step is to extract pitch. One possible way of doing it is using the Harmonic
Product Spectrum method. But you can check other methods in the *pitch* module.
Now you can extract the harmonics, as it's dependent on a pitch method. Iracema already has a 
bulit-in function for plotting the harmonics over the spectrogram.
Notice that the harmonics methods return a dictionary, with it's keys corresponding to three TimeSeries objetcs: 'frequency', 'magnitude' and 'phase'.

.. code:: python
  
  # extract pitch
  hps_pitch = ir.pitch.hps(fft, minf0=1, maxf0=1000)

  #extract harmonics
  harmonics = ir.harmonics.extract(fft, hps_pitch)

.. _architecture:

------------
Architecture
------------

This section will discuss some import aspects of *iracema*’s architecture and offer an overview of the elements that compose the core functionalities of the library.

*iracema* relies on the manipulation of dynamic data, i.e., data that represent an attribute’s changes over time. Thus, *time series* is a fundamental element in *iracema*’s architecture. The starting point for any task is the *audio* time series, from which other kinds of time-related data will be extracted. *iracema* applies transformations called *feature extraction* to time series to obtain new time series. The implementation of such extractors depends on some recurrent types of operations, like applying sliding windows to a series of data. In Iracema, these operations are called *aggregation* methods.

To deal with a specific excerpt of a time series, such as a musical phrase or even a note. There is another important element in the architecture, called *segment*, which is used to delimit such excerpts. A user can specify the limits for a segment within the *time series* if he is already aware of its beginning and end; however, most of the time, users will expect the system to identify such limits by itself, a common kind of task in audio content extraction, known as *segmentation*.

Elements, like audio, time series and segments have been implemented as classes, since they have intrinsic data and behaviour. The ``Audio`` class inherits the functionalities from ``TimeSeries``, and add some specific behaviours (such as loading wave files). ``Segments`` provide a handy way to extract corresponding  excerpts from time series of different sampling rates, since it performs all the necessary index conversion operations to extract data that coincide with the same time interval.

Other elements have been implemented as methods that take objects of those classes as input and output another object. For example, the method *fft* takes as input an *audio* object, a *window_size*, and a *hop_size*, and generates a time series in which each sample contains all the bins of the FFT for the interval corresponding to *hop_size*. The method *spectral_flux* will take a time series containing the result of an FFT operation as input and generate another time series containing the calculated spectral flux. 

Segmentation  methods  will  usually  take *time_series* objects as input to output a list of segments. Then, these segments can be used to extract excerpts from time series objects, using square brackets (the same operator used in Python to perform indexing/slicing operations).

Modules
=======

These are the modules that compose iracema, and their respective functionalities:

- timeseries: contains the definition of the classesTimeSeriesandAudio.
- segment:  contains the definition of the classesSegmentandSegmentList.
- spectral: contains methods for frequency domain analysis (currently the FFT);
- pitch: a few different models for pitch detection.
- harmonics: a model for extracting harmonic components from audio.
- features: contains methods with the implementation of several classic feature extractors.
- segmentation: methods for automatic audio segmentation.
- plot: contains several different methods for plotting time series data.
- aggregation: contains some common aggregation methods that can be useful for implementing feature extractors.
- io:  subpackage containing IO methods, for loading/writing files, playing audio, etc.
- util: subpackage containing some useful modules for unit conversion, DSP, windowing operations, etc.

For more information, please read our article_ on SBCM's 2019 Proceedings. 

.. _article: https://compmus.ime.usp.br/sbcm/2019/papers/sbcm-2019-3.pdf
