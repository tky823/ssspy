ssspy.bss.mnmf
==============


Algorithms
~~~~~~~~~~
.. autoclass:: ssspy.bss.mnmf.FastMNMFBase
    :special-members: __call__
    :members: normalize, normalize_by_power

.. autoclass:: ssspy.bss.mnmf.FastGaussMNMF
    :special-members: __call__
    :members: separate,
        compute_loss, compute_logdet,
        update_once, update_basis, update_activation, update_diagonalizer, update_spatial,
        update_diagonalizer_ip1, update_diagonalizer_ip2
