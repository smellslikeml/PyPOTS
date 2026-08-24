"""
The implementation of MDTIM for the partially-observed time-series imputation task.

Refer to the paper
`Dongbin Kim, Seungyun Lee, Geonwoo Shin, and Jaewook Lee.
Discretizing Continuous Time Series for Imputation with Masked Diffusion Training.
arXiv preprint arXiv:2608.19119, 2026.
<https://arxiv.org/abs/2608.19119>`_

Notes
-----
This implementation is inspired by the paper; no official code was released.
MDTIM is a discrete masked-diffusion imputer, complementary to the continuous
score-based CSDI already shipped in PyPOTS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import MDTIM

__all__ = [
    "MDTIM",
]
