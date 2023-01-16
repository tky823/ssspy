# ssspy
[![Documentation Status](https://readthedocs.org/projects/sound-source-separation-python/badge/?version=latest)](https://sound-source-separation-python.readthedocs.io/en/latest/?badge=latest)
[![black](https://github.com/tky823/ssspy/actions/workflows/lint.yaml/badge.svg)](https://github.com/tky823/ssspy/actions/workflows/lint.yaml)
[![tests](https://github.com/tky823/ssspy/actions/workflows/test_package.yaml/badge.svg)](https://github.com/tky823/ssspy/actions/workflows/test_package.yaml)
[![codecov](https://codecov.io/gh/tky823/ssspy/branch/main/graph/badge.svg?token=IZ89MTV64G)](https://codecov.io/gh/tky823/ssspy)

A Python toolkit for sound source separation.

## Installation
You can install by pip.
```shell
pip install git+https://github.com/tky823/ssspy.git
```
or clone this repository.
```shell
git clone https://github.com/tky823/ssspy.git
cd ssspy
pip install -e .
```

## Build Documentation Locally (optional)
To build the documentation locally, you have to include ``docs`` when installing ``ssspy``.
```shell
pip install -e ".[docs]"
```

When you build the documentation, run the following command.
```shell
cd docs/
make html
```

Or, you can build the documentation automatically using `sphinx-autobuild`.
```shell
# in ssspy/
sphinx-autobuild docs docs/_build/html
```

## Blind Source Separation Methods

| Method | Notebooks |
|:-:|:-:|
| Independent Component Analysis (ICA) [1-3] | Gradient-descent-based ICA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ICA/GradICA.ipynb) <br> Natural-gradient-descent-based ICA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ICA/NaturalGradICA.ipynb) <br> Fast ICA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ICA/FastICA.ipynb) |
| Frequency-Domain Independent Component Analysis (FDICA) [4-6] | Gradient-descent-based FDICA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/GradFDICA.ipynb) <br> Natural-gradient-descent-based FDICA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/NaturalGradFDICA.ipynb) <br> Auxiliary-function-based FDICA (IP1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/AuxFDICA-IP1.ipynb) <br> Auxiliary-function-based FDICA (IP2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/AuxFDICA-IP2.ipynb) <br> Gradient-descent-based Laplace-FDICA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/GradLaplaceFDICA.ipynb) <br> Natural-gradient-descent-based Laplace-FDICA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/NaturalGradLaplaceFDICA.ipynb) <br> Auxiliary-function-based Laplace-FDICA (IP1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/AuxLaplaceFDICA-IP1.ipynb) <br> Auxiliary-function-based Laplace-FDICA (IP2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/FDICA/AuxLaplaceFDICA-IP2.ipynb) |
| Independent Vector Analysis (IVA) [7-13] | Gradient-descent-based IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/GradIVA.ipynb) <br> Natural-gradient-descent-based IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/NaturalGradIVA.ipynb) <br> Fast IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/FastIVA.ipynb) <br> Faster IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/FasterIVA.ipynb) <br> Auxiliary-function-based IVA (IP1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxIVA-IP1.ipynb) <br> Auxiliary-function-based IVA (IP2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxIVA-IP2.ipynb) <br> Auxiliary-function-based IVA (ISS1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxIVA-ISS1.ipynb) <br> Auxiliary-function-based IVA (ISS2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxIVA-ISS2.ipynb) <br> Gradient-descent-based Laplace-IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/GradLaplaceIVA.ipynb) <br> Natural-gradient-descent-based Laplace-IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/NaturalGradLaplaceIVA.ipynb) <br> Auxiliary-function-based Laplace-IVA (IP1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxLaplaceIVA-IP1.ipynb) <br> Auxiliary-function-based Laplace-IVA (IP2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxLaplaceIVA-IP2.ipynb) <br> Auxiliary-function-based Laplace-IVA (ISS1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxLaplaceIVA-ISS1.ipynb) <br> Auxiliary-function-based Laplace-IVA (ISS2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxLaplaceIVA-ISS2.ipynb) <br> Gradient-descent-based Gauss-IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/GradGaussIVA.ipynb) <br> Natural-gradient-descent-based Gauss-IVA: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/NaturalGradGaussIVA.ipynb) <br> Auxiliary-function-based Gauss-IVA (IP1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxGaussIVA-IP1.ipynb) <br> Auxiliary-function-based Gauss-IVA (IP2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxGaussIVA-IP2.ipynb) <br> Auxiliary-function-based Gauss-IVA (ISS1): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxGaussIVA-ISS1.ipynb) <br> Auxiliary-function-based Gauss-IVA (ISS2): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IVA/AuxGaussIVA-ISS2.ipynb) |
| Independent Low-Rank Matrix Analysis (ILRMA) [14-16] | Gauss-ILRMA (IP1 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GaussILRMA-IP1-MM.ipynb) <br> Gauss-ILRMA (IP2 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GaussILRMA-IP2-MM.ipynb) <br> Gauss-ILRMA (ISS1 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GaussILRMA-ISS1-MM.ipynb) <br> Gauss-ILRMA (ISS2 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GaussILRMA-ISS2-MM.ipynb) <br> *t*-ILRMA (IP1 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/TILRMA-IP1.ipynb) <br> *t*-ILRMA (IP2 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/TILRMA-IP2.ipynb) <br> *t*-ILRMA (ISS1 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/TILRMA-ISS1.ipynb) <br> *t*-ILRMA (ISS2 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/TILRMA-ISS2.ipynb) <br> GGD-ILRMA (IP1 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GGDILRMA-IP1.ipynb) <br> GGD-ILRMA (IP2 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GGDILRMA-IP2.ipynb) <br> GGD-ILRMA (ISS1 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GGDILRMA-ISS1.ipynb) <br> GGD-ILRMA (ISS2 / MM): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/ILRMA/GGDILRMA-ISS2.ipynb) |
| Independent Positive Semidefinite Tensor Analysis (IPSDTA) [17, 18] | Gauss-IPSDTA (VCD): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IPSDTA/GaussIPSDTA-VCD.ipynb) <br> *t*-IPSDTA (VCD): [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/IPSDTA/TIPSDTA-VCD.ipynb) |
| Multichannel Nonnegative Matrix Factorization (MNMF) [19-22] | Gauss-MNMF: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/MNMF/GaussMNMF.ipynb) <br> *t*-MNMF: soon <br> Fast Gauss-MNMF: soon |
| Blind Source Separation via Primal-Dual Splitting Algorithm (PDS-BSS) [23] | PDS-BSS: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/PDSBSS/PDSBSS.ipynb) <br> PDS-BSS-multiPenalty: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/PDSBSS/PDSBSS_multi-penalty.ipynb) |
| Complex Angular Central Gaussian Mixture Model (cACGMM) [24] | cACGMM: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/ssspy/blob/main/notebooks/BSS/CACGMM/CACGMM.ipynb) |

- [1] [P. Comon, "Independent component analysis, a new concept?" Signal Processing, vol. 36, no. 3, pp. 287-314, 1994.](https://www.sciencedirect.com/science/article/pii/0165168494900299)
- [2] [S. Amari, A. Cichocki, and H. H. Yang, "A new learning algorithm forblind signal separation," in *Proc. NIPS*, 1996, pp. 757-763.](https://proceedings.neurips.cc/paper/1995/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html)
- [3] [A. Hyvärinen, "Fast and robust fixed-point algorithms for independent component analysis," *IEEE Trans. on Neural Netw.*, vol. 10, no. 3, pp. 626-634, 1999.](https://www.cs.helsinki.fi/u/ahyvarin/papers/TNN99new.pdf)
- [4] [N. Murata, S. Ikeda, and A. Ziehe, "An approach to blind source separation based on temporal structure of speech signals," 2001.](https://www.sciencedirect.com/science/article/pii/S0925231200003453)
- [5] [H. Sawada, S. Araki, and S. Makino, "Underdetermined convolutive blind source separation via frequency bin-wise clustering and permutation alignment," 2011.](https://ieeexplore.ieee.org/document/5473129)
- [6] [N. Ono and S. Miyabe, "Auxiliary-function-based independent componentanalysis for super-Gaussian sources," in *Proc. LVA/ICA*, 2010, pp. 165-172.](https://link.springer.com/chapter/10.1007/978-3-642-15995-4_21)
- [7] [T. Kim, T. Attias, S.-Y. Lee, and T.-W. Lee, "Blind source separation exploiting higher-order frequency dependencies," *IEEE trans. ASLP*, vol. 15, no.1, pp. 70-79, 2006.](https://link.springer.com/chapter/10.1007/11679363_21)
- [8] [I. Lee, T. Kim, and T.-W. Lee, "Fast fixed-point independent vector analysis algorithms for convolutive blind source separation," *Signal Processing*, vol. 87, no. 8, pp. 1859-1871, 2007.]()
- [9] [N. Ono, "Stable and fast update rules for independent vector analysis based on auxiliary function technique," in *Proc. WASPAA*, 2011, p.189-192.](https://ieeexplore.ieee.org/document/6082320)
- [10] [N. Ono, "Auxiliary-function-based independent vector analysis with power of vector-norm type weighting functions," in *Proc. APSIPA ASC*, 2012, pp. 1-4.](https://ieeexplore.ieee.org/document/6411886)
- [11] [R. Scheibler and N. Ono, "Fast and stable blind source separation with rank-1 updates," in *Proc. ICASSP*, 2020, pp. 236-240.](https://ieeexplore.ieee.org/document/9053556)
- [12] [A. Brendel and W. Kellermann, "Faster IVA: update rules for independent vector analysis based on negentropy and the majorize-minimize principle," *arXiv:2003.09531*, 2020.](https://arxiv.org/abs/2003.09531)
- [13] [R. Ikeshita and T. Nakatani, "ISS2: An extension of iterative source steering algorithm for majorization-minimization-based independent vector analysis," *arXiv:2202.00875*, 2022.](https://arxiv.org/abs/2202.00875)
- [14] [D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari, "Determined blind source separation unifying independent vector analysis and nonnegative matrix factorization," *IEEE/ACM Trans. ASLP*, vol. 24, no. 9, pp. 1626-1641, 2016.](https://ieeexplore.ieee.org/document/7486081)
- [15] [D. Kitamura, S. Mogami, Y. Mitsui, N. Takamune, H. Saruwatari, N. Ono, Y. Takahashi, and K. Kondo, "Generalized independent low-rank matrix analysis using heavy-tailed distributions for blind source separation," *EURASIP J. Adv. in Signal Processing*, vol. 2018, no. 28, 25 pages, 2018.](https://link.springer.com/article/10.1186/s13634-018-0549-5)
- [16] [T. Nakashima, R. Scheibler, Y. Wakabayashi, and N. Ono, "Faster independent low-rank matrix analysis with pairwise updates of demixing vectors," in *Proc. EUSIPCO*, 2021, pp. 301-305.](https://ieeexplore.ieee.org/document/9287508)
- [17] [R. Ikeshita, "Independent positive semidefinite tensor analysis in blind source separation," in *Proc. EUSIPCO*, 2018, pp. 1652-1656.](https://ieeexplore.ieee.org/document/8553546)
- [18] [T. Kondo, K. Fukushige, N. Takamune, D. Kitamura, H. Saruwatari, R. Ikeshita, and T. Nakatani, "Convergence-guaranteed independent positive semidefinite tensor analysis based on Student's t distribution," in *Proc ICASSP*, 2020, pp. 681-685.](https://ieeexplore.ieee.org/document/9054150)
- [19] [A. Ozerov and C. Fevotte, "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation," *IEEE Trans. ASLP*, vol. 18, no. 3, pp. 550-563, 2010.](https://ieeexplore.ieee.org/document/5229304)
- [20] [H. Sawada, H. Kameoka, S. Araki, and N. Ueda, "Multichannel extensions of non-negative matrix factorization with complex-valued data," *IEEE Trans. ASLP*, vol. 21, no. 5, pp. 971-982, 2013.](https://ieeexplore.ieee.org/document/6410389)
- [21] [K. Yoshii, K. Itoyama, and M. Goto, "Student's T nonnegative matrix factorization and positive semidefinite tensor factorization for single-channel audio source separation," in *Proc. ICASSP*, 2016, pp. 51-55.](https://ieeexplore.ieee.org/document/7471635)
- [22] [K. Sekiguchi, A. A. Nugraha, Y. Bando, and K. Yoshii, "Fast multichannel source separation based on jointly diagonalizable spatial covariance matrices," *arXiv:1903.03237*, 2019.](https://arxiv.org/abs/1903.03237)
- [23] [K. Yatabe and D. Kitamura, "Determined blind source separation via proximal splitting algorithm," in *Proc. ICASSP*, 2018, pp. 776-780.](https://ieeexplore.ieee.org/document/8462338)
- [24] [N. Ito, S. Araki, and T. Nakatani. "Complex angular central Gaussian mixture model for directional statistics in mask-based microphone array signal processing," in *Proc. EUSIPCO*, 2016, pp. 1153-1157.](https://ieeexplore.ieee.org/document/7760429)

## LICENSE
Apache License 2.0
