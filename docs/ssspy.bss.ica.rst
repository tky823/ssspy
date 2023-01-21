ssspy.bss.ica
=============

In this module, we separate time-domain multichannel signals
using independent component analysis (ICA) [#comon1994independent]_.
We denote the number of sources and microphones as :math:`N` and :math:`M`, respectively.
We also denote source, observed, and separated signals (in time-domain)
as :math:`\boldsymbol{s}_{t}`, :math:`\boldsymbol{x}_{t}`, and :math:`\boldsymbol{y}_{t}`,
respectively.

.. math::
   \boldsymbol{s}_{t}
   &= (s_{t1},\ldots,s_{tn},\ldots,s_{tN})^{\mathsf{T}}\in\mathbb{R}^{N}, \\
   \boldsymbol{x}_{t}
   &= (x_{t1},\ldots,x_{tm},\ldots,x_{tM})^{\mathsf{T}}\in\mathbb{R}^{M}, \\
   \boldsymbol{y}_{t}
   &= (y_{t1},\ldots,y_{tn},\ldots,y_{tN})^{\mathsf{T}}\in\mathbb{R}^{N},

where :math:`t=1,\ldots,T` is an index of time samples.
When a mixing system is time-invariant, :math:`\boldsymbol{x}_{t}` is represented as follows:

.. math::
   \boldsymbol{x}_{t}
   = \boldsymbol{A}\boldsymbol{s}_{t},

where :math:`\boldsymbol{A}=(\boldsymbol{a}_{1},\ldots,\boldsymbol{a}_{n},\ldots,\boldsymbol{a}_{N})\in\mathbb{R}^{M\times N}` is
a mixing matrix.
If :math:`M=N` and :math:`\boldsymbol{A}` is non-singular, a demixing system is represented as

.. math::
   \boldsymbol{y}_{t}
   = \boldsymbol{W}\boldsymbol{x}_{t},

where :math:`\boldsymbol{W}=(\boldsymbol{w}_{1},\ldots,\boldsymbol{w}_{n},\ldots,\boldsymbol{w}_{N})^{\mathsf{T}}\in\mathbb{R}^{N\times M}` is
a demixing matrix.
The negative log-likelihood of observed signals (divided by :math:`T`) is computed as follows:

.. math::
   \mathcal{L}
   &= -\frac{1}{T}\log p(\mathcal{X}) \\
   &= -\frac{1}{T}\left(\log p(\mathcal{Y}) \
   + \log|\det\boldsymbol{W}|^{T} \right) \\
   &= -\frac{1}{T}\sum_{t,n}\log p(y_{tn})
   - \log|\det\boldsymbol{W}| \\
   &= \frac{1}{T}\sum_{t,n}G(y_{tn})
   - \log|\det\boldsymbol{W}|, \\
   G(y_{tn})
   &= -\log p(y_{tn}),

where :math:`G(y_{tn})` is a contrast function.
The derivative of :math:`G(y_{tn})` is called a score function.

.. math::
   \phi(y_{tn})
   = \frac{\partial G(y_{tn})}{\partial y_{ijn}}.

.. [#comon1994independent] P. Comon,
   "Independent component analysis, a new concept?"
   *Signal Processing*, vol. 36, no. 3, pp. 287-314, 1994.

Algorithms
~~~~~~~~~~

.. autoclass:: ssspy.bss.ica.GradICABase
   :special-members: __call__
   :members: separate, compute_loss, compute_logdet

.. autoclass:: ssspy.bss.ica.FastICABase
   :special-members: __call__
   :members: separate, compute_loss

.. autoclass:: ssspy.bss.ica.GradICA
   :members: update_once

.. autoclass:: ssspy.bss.ica.NaturalGradICA
   :members: update_once

.. autoclass:: ssspy.bss.ica.FastICA
   :members: update_once

.. autoclass:: ssspy.bss.ica.GradLaplaceICA
   :members: update_once, compute_loss

.. autoclass:: ssspy.bss.ica.NaturalGradLaplaceICA
   :members: update_once, compute_loss
