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

   import os
   import urllib.request
   import zipfile

   import numpy as np
   import soundfile as sf
   import matplotlib.pyplot as plt

   from ssspy.bss.ica import NaturalGradLaplaceICA


   def download_sisec2011():
      filename = "dev1.zip"
      save_dir = "./data/SiSEC2011"
      save_path = os.path.join(save_dir, filename)
      url = "http://www.irisa.fr/metiss/SiSEC10/underdetermined/{}".format(filename)
      data = urllib.request.urlopen(url).read()

      os.makedirs(save_dir, exist_ok=True)

      if not os.path.exists(save_path):
         with open(save_path, mode="wb") as f:
               f.write(data)

      with zipfile.ZipFile(save_path) as f:
         f.extractall(save_dir)

      wav_path = os.path.join(save_dir, "dev1_female3_src_{src_idx}.wav")
      wav_paths = [wav_path.format(src_idx=src_idx + 1) for src_idx in range(n_sources)]

      return wav_paths


   n_channels = n_sources = 2
   wav_paths = download_sisec2011()

   waveform_src = []

   for wav_path in wav_paths:
      waveform, _ = sf.read(wav_path)
      waveform_src.append(waveform)

   waveform_src = np.stack(waveform_src, axis=0)
   mixing_matrix = np.random.randn(n_channels, n_sources)
   waveform_mix = mixing_matrix @ waveform_src

   ica = NaturalGradLaplaceICA(is_holonomic=True)
   waveform_est = ica(waveform_mix, n_iter=100)

   plt.figure()
   plt.plot(ica.loss)
   plt.show()


.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
