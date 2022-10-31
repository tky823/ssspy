ssspy.bss.iva
=============

In this module, we separate multichannel signals
using independent vector analysis (IVA).
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
We also define the following vector:

.. math::
   \vec{\boldsymbol{y}}_{jn}
   = (y_{1jn},\ldots,y_{ijn},\ldots,y_{Ijn})^{\mathsf{T}}\in\mathbb{C}^{I}.

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
   &= -\frac{1}{J}\sum_{j,n}\log p(\vec{\boldsymbol{y}}_{jn})
   - 2\sum_{i}\log|\det\boldsymbol{W}_{i}| \\
   &= \frac{1}{J}\sum_{j,n}G(\vec{\boldsymbol{y}}_{jn})
   - 2\sum_{i}\log|\det\boldsymbol{W}_{i}|, \\
   G(\vec{\boldsymbol{y}}_{jn})
   &= -\log p(\vec{\boldsymbol{y}}_{jn}),

where :math:`G(\vec{\boldsymbol{y}}_{jn})` is a contrast function.
The derivative of :math:`G(\vec{\boldsymbol{y}}_{jn})` is called a score function.

.. math::
   \phi_{i}(\vec{\boldsymbol{y}}_{jn})
   = \frac{\partial G(\vec{\boldsymbol{y}}_{jn})}{\partial y_{ijn}^{*}}.

Algorithms
~~~~~~~~~~
.. autoclass:: ssspy.bss.iva.IVAbase
   :special-members: __call__
   :members: separate, update_once, compute_loss, compute_logdet, restore_scale, apply_projection_back

.. autoclass:: ssspy.bss.iva.GradIVAbase

.. autoclass:: ssspy.bss.iva.FastIVAbase
   :members: separate, compute_loss, apply_projection_back

.. autoclass:: ssspy.bss.iva.AuxIVAbase
   :special-members: __call__
   :members: separate, compute_loss, apply_projection_back

.. autoclass:: ssspy.bss.iva.GradIVA
   :members: update_once

.. autoclass:: ssspy.bss.iva.NaturalGradIVA
   :members: update_once

.. autoclass:: ssspy.bss.iva.FastIVA
   :special-members: __call__
   :members: update_once

.. autoclass:: ssspy.bss.iva.FasterIVA
   :special-members: __call__
   :members: update_once

.. autoclass:: ssspy.bss.iva.AuxIVA
   :special-members: __call__
   :members: update_once, update_once_ip1, update_once_ip2, update_once_iss1, update_once_iss2

.. autoclass:: ssspy.bss.iva.GradLaplaceIVA
   :members: update_once, compute_loss

.. autoclass:: ssspy.bss.iva.GradGaussIVA
   :members: update_once, update_source_model

.. autoclass:: ssspy.bss.iva.NaturalGradLaplaceIVA
   :members: update_once, compute_loss

.. autoclass:: ssspy.bss.iva.NaturalGradGaussIVA
   :members: update_once, compute_loss

.. autoclass:: ssspy.bss.iva.AuxLaplaceIVA

.. autoclass:: ssspy.bss.iva.AuxGaussIVA
   :members: update_once, update_source_model
