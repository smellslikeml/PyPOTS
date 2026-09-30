"""
Dataset class for the forecasting model CoIFNet.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ...data.dataset.base import BaseDataset


class DatasetForCoIFNet(BaseDataset):
    """Dataset for CoIFNet forecasting model.

    CoIFNet consumes exactly the standard pypots forecasting sample structure, i.e.
    ``[index, X, missing_mask, X_pred, X_pred_missing_mask]`` when ``X_pred`` is available
    (training/validation stages), and ``[index, X, missing_mask]`` for testing. X is returned
    zero-filled at missing positions together with the missingness mask, which is what CoIFNet's
    mask-aware fusion modules and RevON normalization expect. Hence this class simply inherits
    the fetching logic from :class:`pypots.data.dataset.base.BaseDataset` and fixes the flags
    for the forecasting task, following the pattern of the CSDI dataset class.
    """

    def __init__(
        self,
        data: Union[dict, str],
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=False,
            return_X_pred=True,
            return_y=False,
            file_type=file_type,
        )
