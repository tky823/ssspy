APIs
====

Introduction
------------

.. code-block:: python

   import numpy as np
   import scipy.signal as ss
   import IPython.display as ipd
   import matplotlib.pyplot as plt

   from ssspy.utils.dataset import download_sample_speech_data
   from ssspy.transform import whiten
   from ssspy.algorithm import projection_back
   from ssspy.bss.fdica import AuxFDICA

   n_fft, hop_length = 4096, 2048
   window = "hann"

   waveform_src_img = download_sample_speech_data(n_sources=3)
   waveform_mix = np.sum(waveform_src_img, axis=1)
   _, _, spectrogram_mix = ss.stft(
      waveform_mix,
      window=window,
      nperseg=n_fft,
      noverlap=n_fft-hop_length
   )
   _, _, spectrogram_mix = ss.stft(
      waveform_mix,
      window=window,
      nperseg=n_fft,
      noverlap=n_fft-hop_length
   )

   def contrast_fn(y):
    return 2 * np.abs(y)

   def d_contrast_fn(y):
      return 2 * np.ones_like(y)

   fdica = AuxFDICA(
      contrast_fn=contrast_fn,
      d_contrast_fn=d_contrast_fn,
   )
   spectrogram_mix_whitened = whiten(spectrogram_mix)
   spectrogram_est = fdica(spectrogram_mix_whitened)
   spectrogram_est = projection_back(spectrogram_est, reference=spectrogram_mix)

   _, waveform_est = ss.istft(
      spectrogram_est,
      window=window,
      nperseg=n_fft,
      noverlap=n_fft-hop_length
   )

   for idx, waveform in enumerate(waveform_est):
      print("Estimated source: {}".format(idx + 1))
      ipd.display(ipd.Audio(waveform, rate=16000))
      print()

   plt.figure()
   plt.plot(fdica.loss)
   plt.show()

Submodules
----------

.. toctree::
   :maxdepth: 1

   ssspy.algorithm
   ssspy.bss
