ssspy.bss.pdsbss
================

In this module, we separate multichannel signals
using blind source separation via primal dual splitting algorithm.
We denote the number of sources and microphones as :math:`N` and :math:`M`, respectively.
We also denote short-time Fourier transforms of source, observed, and separated signals
as :math:`\boldsymbol{s}_{ij}`, :math:`\boldsymbol{x}_{ij}`, and :math:`\boldsymbol{y}_{ij}`,
respectively.

.. math::
   \boldsymbol{s}_{ij}
   &= (s_{ij1},\ldots,s_{ijn},\ldots,s_{ijN})^{\mathsf{T}}\in\mathbb{C}^{N}, \\
   \boldsymbol{x}_{ij}
   &= (x_{ij1},\ldots,x_{ijm},\ldots,x_{ijM})^{\mathsf{T}}\in\mathbb{C}^{M}, \\
   \boldsymbol{y}_{ij}
   &= (y_{ij1},\ldots,y_{ijn},\ldots,y_{ijN})^{\mathsf{T}}\in\mathbb{C}^{N},

where :math:`i=1,\ldots,I` and :math:`j=1,\ldots,J` are indices of frequency bins and time frames, respectively.
When a mixing system is time-invariant, :math:`\boldsymbol{x}_{ij}` is represented as follows:

.. math::
   \boldsymbol{x}_{ij}
   = \boldsymbol{A}_{i}\boldsymbol{s}_{ij},

where :math:`\boldsymbol{A}_{i}=(\boldsymbol{a}_{i1},\ldots,\boldsymbol{a}_{in},\ldots,\boldsymbol{a}_{iN})\in\mathbb{C}^{M\times N}` is
a mixing matrix.
If :math:`M=N` and :math:`\boldsymbol{A}_{i}` is non-singular, a demixing system is represented as

.. math::
   \boldsymbol{y}_{ij}
   = \boldsymbol{W}_{i}\boldsymbol{x}_{ij},

where :math:`\boldsymbol{W}_{i}=(\boldsymbol{w}_{i1},\ldots,\boldsymbol{w}_{in},\ldots,\boldsymbol{w}_{iN})^{\mathsf{H}}\in\mathbb{C}^{N\times M}` is
a demixing matrix.
The negative log-likelihood of observed signals (divided by :math:`2J`) is computed as follows:

.. math::
   \mathcal{L}
   &= \mathcal{P}(\mathcal{V}(\mathcal{Y}))
   + \sum_{i}\mathcal{I}(\boldsymbol{W}_{i}), \\
   \mathcal{V}(\mathcal{Y})
   &:= (y_{111},\ldots,y_{11N},\ldots,y_{1JN},\ldots,y_{IJN})^{\mathsf{T}}
   \in\mathbb{C}^{IJN} \\
   \mathcal{I}(\boldsymbol{W}_{i})
   &= - \log|\det\boldsymbol{W}_{i}|,

where :math:`\mathcal{P}` is a penalty funcion that is determined by the source model.

Let us consider independent vector analysis.
In this case, :math:`\mathcal{P}` can be written by

.. math::
   \mathcal{P}(\mathcal{V}(\mathcal{Y}))
   = C\sum_{j,n}\left(
   \sum_{i}\left|\boldsymbol{w}_{in}^{\mathsf{H}}\boldsymbol{x}_{ij}\right|^{2}
   \right)^{\frac{1}{2}},

where :math:`C` is a positive constant.

To the above formulation, we can apply the primal-dual splitting algorithm.
On the basis of this algorithm, the demixing filter is updated as follows:

.. math::
   \tilde{\boldsymbol{W}}_{i}
   &\leftarrow\mathrm{prox}_{\mu_{1}\mathcal{I}}
   \left[\boldsymbol{W}_{i} - \mu_{1}\mu_{2}\sum_{j}\boldsymbol{u}_{ij}\boldsymbol{x}_{ij}^{\mathsf{H}}\right] \\
   \boldsymbol{z}_{ij}
   &\leftarrow\boldsymbol{u}_{ij} + \left(2 * \tilde{\boldsymbol{W}}_{i} - \boldsymbol{W}_{i}\right)\boldsymbol{x}_{ij} \\
   \mathcal{V}(\tilde{\mathcal{U}})
   &\leftarrow\mathcal{V}(\mathcal{Z})
   - \mathrm{prox}_{\mathcal{P}/\mu_{2}}\left[\mathcal{V}(\mathcal{Z})\right] \\
   \boldsymbol{u}_{ij}
   &\leftarrow\alpha\tilde{\boldsymbol{u}}_{ij} + (1 - \alpha)\boldsymbol{u}_{ij}, \\
   \boldsymbol{W}_{i}
   &\leftarrow\alpha\tilde{\boldsymbol{W}}_{i} + (1 - \alpha)\boldsymbol{W}_{i}.

:math:`\boldsymbol{u}_{ij}` is a dual variable, which should be initialized by a certain value.
:math:`\mathrm{prox}_{g}` is a proximal operator defined as

.. math::
   \mathrm{prox}_{g}[\boldsymbol{z}]
   = \mathrm{argmin}_{\boldsymbol{y}}
   ~~g(\boldsymbol{y}) + \frac{1}{2}\|\boldsymbol{z} - \boldsymbol{y}\|_{2}^{2}.
