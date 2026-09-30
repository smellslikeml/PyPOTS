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
(MIT-licensed). Only the model architecture (the Cross-Timestep Fusion and Cross-Variate Fusion
modules, and the reversible normalization layers RevIN/RevON) and its forward/loss computation
are ported. The training loop, data loading, and configuration management of the reference
repository are not ported because ``BaseNNForecaster`` already owns them.

CoIFNet forecasts directly on incomplete input as a joint imputation-forecasting model, rather
than imputing first and forecasting afterwards. The training objective couples the forecasting
loss on the prediction horizon with a reconstruction loss on the observed window, weighted by
``loss_lambda`` as in the reference implementation.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from .core import _CoIFNet
from .data import DatasetForCoIFNet
from ..base import BaseNNForecaster
from ...data.checking import key_in_data_set
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class CoIFNet(BaseNNForecaster):
    """The PyTorch implementation of the CoIFNet forecasting model.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_pred_steps :
        The number of steps in the forecasting time series.

    n_pred_features :
        The number of features in the forecasting time series. CoIFNet fuses and denormalizes
        variates one by one, so if it differs from ``n_features``, a linear projection is
        appended to map the forecast back to the target feature dimension.

    use_reconstruct :
        Whether to enable the joint imputation-forecasting objective, i.e. to let the model
        also output the reconstruction of the observed window and add the reconstruction loss
        to the training loss. This is the ``use_reconstruct`` option in the reference
        implementation and defaults to True there.

    use_mask :
        Whether to concatenate the missingness mask to the model input, the key mechanism
        letting CoIFNet forecast directly on incomplete input. Default as True.

    d_model :
        The dimension of the hidden layer in the Cross-Timestep Fusion and Cross-Variate
        Fusion blocks, named ``hidden`` in the reference implementation.

    dropout :
        The dropout rate for the model.

    intra_type :
        The type of the Cross-Timestep Fusion (CTF) block. It has to be one of
        ['TSBlock', 'LinearBlock', 'AttentionBlock'].

    inter_type :
        The type of the Cross-Variate Fusion (CVF) block. It has to be one of
        ['TSBlock', 'LinearBlock', 'AttentionBlock'].

    use_head :
        Whether to map the fusion output to the horizon through an auxiliary linear head,
        rather than letting the fusion module directly output the horizon length.

    loss_lambda :
        The weight of the forecasting loss in the joint objective, i.e. the training loss is
        ``loss_lambda * forecasting_loss + (1 - loss_lambda) * reconstruction_loss``.
        Default as 0.1, the value used by the reference implementation.

    use_reversible_norm :
        Whether to apply reversible instance normalization (RevIN/RevON) to the input data.

    use_revon :
        Whether to use RevON, the missing-value-aware variant of RevIN that computes
        normalization statistics over observed entries only. Only effective when
        ``use_reversible_norm`` is True.

    revin_affine :
        Whether RevIN/RevON use learnable affine parameters.

    use_time_features :
        Whether to concatenate timestamp features to the model input, as the reference
        implementation does when time features are available. Note pypots forecasting
        datasets do not carry timestamps, so this should be left as False unless time
        features are provided in the model input dictionary under the key ``time_features``.

    use_time_feature_embedding :
        Whether to embed the time-in-day and time-in-week features with learnable embeddings
        rather than concatenating the raw features. Only effective when ``use_time_features``
        is True.

    temp_dim_tid :
        The embedding dimension for the time-in-day feature.

    temp_dim_tiw :
        The embedding dimension for the time-in-week (day-of-week) feature.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    training_loss:
        The customized loss function designed by users for training the model.
        If not given, will use the default masked MAE loss as claimed in the original paper.

    validation_metric:
        The customized metric function designed by users for validating the model.
        If not given, will use the default MSE metric.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose :
        Whether to print out the training logs during the training process.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        use_reconstruct: bool = True,
        use_mask: bool = True,
        d_model: int = 256,
        dropout: float = 0.1,
        intra_type: str = "TSBlock",
        inter_type: str = "TSBlock",
        use_head: bool = True,
        loss_lambda: float = 0.1,
        use_reversible_norm: bool = True,
        use_revon: bool = True,
        revin_affine: bool = True,
        use_time_features: bool = False,
        use_time_feature_embedding: bool = True,
        temp_dim_tid: int = 8,
        temp_dim_tiw: int = 8,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.use_reconstruct = use_reconstruct
        self.use_mask = use_mask
        self.d_model = d_model
        self.dropout = dropout
        self.intra_type = intra_type
        self.inter_type = inter_type
        self.use_head = use_head
        self.loss_lambda = loss_lambda
        self.use_reversible_norm = use_reversible_norm
        self.use_revon = use_revon
        self.revin_affine = revin_affine
        self.use_time_features = use_time_features
        self.use_time_feature_embedding = use_time_feature_embedding
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_tiw = temp_dim_tiw

        # set up the model
        self.model = _CoIFNet(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_pred_steps=self.n_pred_steps,
            n_pred_features=self.n_pred_features,
            use_reconstruct=self.use_reconstruct,
            use_mask=self.use_mask,
            use_time_features=self.use_time_features,
            use_time_feature_embedding=self.use_time_feature_embedding,
            temp_dim_tid=self.temp_dim_tid,
            temp_dim_tiw=self.temp_dim_tiw,
            d_model=self.d_model,
            dropout=self.dropout,
            intra_type=self.intra_type,
            inter_type=self.inter_type,
            use_head=self.use_head,
            loss_lambda=self.loss_lambda,
            use_reversible_norm=self.use_reversible_norm,
            use_revon=self.use_revon,
            revin_affine=self.revin_affine,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the forecaster on the given data.

        Parameters
        ----------
        train_set :
            The dataset for model training, should be a dictionary including the keys 'X' and 'X_pred',
            or a path string locating a data file.
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for training and can contain missing values, and X_pred should
            be array-like with shape [n_samples, n_pred_steps, n_features].
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the keys 'X' and 'X_pred'.

        val_set :
            The dataset for model validating, should be a dictionary including the keys 'X' and 'X_pred',
            or a path string locating a data file.
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for validating and can contain missing values, and X_pred should
            be array-like with shape [n_samples, n_pred_steps, n_features].
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the keys 'X' and 'X_pred'.

        file_type :
            The type of the given file if train_set and val_set are path strings.

        """
        # Step 1: wrap the input data with classes Dataset and DataLoader
        train_dataset = DatasetForCoIFNet(
            train_set,
            file_type=file_type,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_pred", val_set):
                raise ValueError("val_set must contain 'X_pred' for model validation.")
            val_dataset = DatasetForCoIFNet(
                val_set,
                file_type=file_type,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
