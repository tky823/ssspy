ssspy.bss.fdica
===============

In this module, we separate multichannel signals
using frequency-domain independent component analysis (FDICA).
We denote the number of sources and microphones as :math:`N` and :math:`M`, respectively.
We also denote source, observed, and separated signals (in time-domain)
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
The negative log-likelihood of observed signals (divided by :math:`J`) is computed as follows:

.. math::
   \mathcal{L}
   &= -\frac{1}{J}\log p(\mathcal{X}) \\
   &= -\frac{1}{J}\left(\log p(\mathcal{Y}) \
   + \sum_{i}\log|\det\boldsymbol{W}_{i}|^{2J} \right) \\
   &= -\frac{1}{J}\sum_{i,j,n}\log p(y_{ijn})
   - 2\sum_{i}\log|\det\boldsymbol{W}_{i}| \\
   &= \sum_{i}\mathcal{L}^{[i]}, \\
   \mathcal{L}^{[i]} \
   &= \frac{1}{J}\sum_{j,n}G(y_{ijn})
   - 2\log|\det\boldsymbol{W}_{i}|, \\
   G(y_{ijn})
   &= -\log p(y_{ijn}),

where :math:`G(y_{ijn})` is a contrast function.
The derivative of :math:`G(y_{ijn})` is called a score function.

.. math::
   \phi(y_{ijn})
   = \frac{\partial G(y_{ijn})}{\partial y_{ijn}^{*}}.

Algorithms
~~~~~~~~~~

.. autoclass:: ssspy.bss.fdica.GradFDICA
   :special-members: __call__
   :members:
   :undoc-members:

.. autoclass:: ssspy.bss.fdica.NaturalGradFDICA
   :special-members: __call__
   :members:
   :undoc-members:

.. autoclass:: ssspy.bss.fdica.AuxFDICA
   :special-members: __call__
   :members:
   :undoc-members: