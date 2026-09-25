"""
The implementation of CoIFNet for the partially-observed time-series forecasting task.

Refer to the paper
`Kai Tang, Shengsheng Lin, Chacha Chen, Cen Chen, Zhennan Feng, Yadong Zhang, and Jun Zhou.
"A Unified Framework for Multivariate Time Series Forecasting with Missing Values".
arXiv:2506.13064.
<https://arxiv.org/abs/2506.13064>`_

Notes
-----
This implementation is a port with attribution from the official one https://github.com/KaiTang-eng/CoIFNet
(MIT-licensed). Only the model architecture and its forward/loss computation are ported, and
the training loop, data loading, and configuration management of the reference repository are
not ported because ``BaseNNForecaster`` already owns them.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import CoIFNet

__all__ = [
    "CoIFNet",
]
