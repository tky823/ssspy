APIs
====

Introduction
------------

.. code-block:: python

   import soundfile as sf
   import scipy.signal as ss
   from ssspy.bss.ica import NaturalGradLaplaceICA

   waveform_mix, _ = sf.read("sample-2ch.wav")
   waveform_mix = waveform_mix.T
   print(waveform_mix.shape) # (2, 160000)
   ica = NaturalGradLaplaceICA()
   waveform_est = ica(waveform_mix)
   print(waveform_mix.shape, waveform_est.shape) # (2, 160000), (2, 160000)

Submodules
----------

.. toctree::
   :maxdepth: 1

   ssspy.bss