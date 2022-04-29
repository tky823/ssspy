APIs
====

Introduction
------------

.. code-block:: python

   import soundfile as sf
   import scipy.signal as ss
   from ssspy.bss.ica import NaturalGradICA

   waveform_mix, _ = sf.read("sample-2ch.wav")
   print(waveform_mix.shape) # (160000, 2)
   ica = NaturalGradICA()
   waveform_est = ica(waveform_mix)
   print(waveform_mix.shape, waveform_est.shape) # (2, 160000)
