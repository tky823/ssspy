.. ssspy documentation master file, created by
   sphinx-quickstart on Fri Apr 29 20:59:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ssspy's documentation!
=================================

.. image:: https://readthedocs.org/projects/sound-source-separation-python/badge/?version=latest
   :target: https://sound-source-separation-python.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/tky823/ssspy/actions/workflows/lint.yaml/badge.svg
   :target: https://github.com/tky823/ssspy/actions/workflows/lint.yaml

.. image:: https://github.com/tky823/ssspy/actions/workflows/test_package.yaml/badge.svg
   :target: https://github.com/tky823/ssspy/actions/workflows/test_package.yaml

.. image:: https://codecov.io/gh/tky823/ssspy/branch/main/graph/badge.svg?token=IZ89MTV64G
   :target: https://codecov.io/gh/tky823/ssspy

``ssspy`` is a Python toolkit for sound source separation.

Installation
------------

You can install ``ssspy`` by pip.

.. code-block:: shell

   pip install git+https://github.com/tky823/ssspy.git

or clone the repository.

.. code-block:: shell

   git clone https://github.com/tky823/ssspy.git
   cd ssspy
   pip install -e .

Build Documentation Locally (optional)
--------------------------------------
To build the documentation locally, you have to include ``docs`` when installing ``ssspy``.

.. code-block:: shell

   pip install -e ".[docs]"

When you build the documentation, run the following command.

.. code-block:: shell

   cd docs/
   make html

Or, you can build the documentation automatically using ``sphinx-autobuild``.

.. code-block:: shell

   # in ssspy/
   sphinx-autobuild docs docs/_build/html

Quick Example of Blind Source Separation
----------------------------------------

.. code-block:: python

   import numpy as np
   import scipy.signal as ss
   import IPython.display as ipd
   import matplotlib.pyplot as plt

   from ssspy.utils.dataset import download_sample_speech_data
   from ssspy.bss.iva import AuxLaplaceIVA


   n_fft, hop_length = 4096, 2048
   window = "hann"

   waveform_src_img, sample_rate = download_sample_speech_data(n_sources=3)
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

   iva = AuxLaplaceIVA()
   spectrogram_est = iva(spectrogram_mix)

   _, waveform_est = ss.istft(
      spectrogram_est,
      window=window,
      nperseg=n_fft,
      noverlap=n_fft-hop_length
   )

   for idx, waveform in enumerate(waveform_est):
      print("Estimated source: {}".format(idx + 1))
      ipd.display(ipd.Audio(waveform, rate=sample_rate))
      print()

   plt.figure()
   plt.plot(iva.loss)
   plt.show()


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
