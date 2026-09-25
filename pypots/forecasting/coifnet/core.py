"""
The core wrapper assembles the submodules of CoIFNet forecasting model
and takes over the forward progress of the algorithm.

This is a port with attribution from the MIT-licensed official implementation
https://github.com/KaiTang-eng/CoIFNet (files ``model/CoIFNet.py``, ``model/RevIN.py``,
``model/RevON.py``, and ``model/attend.py``) of the paper
`A Unified Framework for Multivariate Time Series Forecasting with Missing Values
<https://arxiv.org/abs/2506.13064>`_ (Tang et al., arXiv:2506.13064).

Only the model architecture and its forward/loss computation are ported. The training loop,
data loading, and configuration management of the reference repository are deliberately NOT
ported because :class:`pypots.forecasting.base.BaseNNForecaster` already owns them.

Notes
-----
CoIFNet is a joint imputation-forecasting model that forecasts directly on incomplete input,
rather than imputing first and forecasting afterwards. It fuses the observed values, the
missingness mask (and optionally timestamp features) through two modules:

1. Cross-Timestep Fusion (CTF, ``intra_model`` in the reference code) that mixes information
   along the time axis and maps the lookback window to the prediction horizon;
2. Cross-Variate Fusion (CVF, ``inter_model`` in the reference code) that mixes information
   across variates at every time step.

Reversible normalization (RevIN, or the missing-value-aware RevON variant) is applied before
the fusion modules and reversed afterwards. RevON computes its statistics over observed
entries only, hence missing positions must be zero-filled in ``X`` (this is exactly what
``pypots.data.dataset.base.BaseDataset`` does together with the missing mask).

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Ported from ``model/RevIN.py`` of https://github.com/KaiTang-eng/CoIFNet (MIT),
    which itself derives from https://github.com/ts-kim/RevIN with minor modifications.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"RevIN mode {mode} is not implemented.")
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class RevON(nn.Module):
    """Reversible normalization with missing-value-aware (Observed-only) statistics, i.e. RevON.

    Ported from ``model/RevON.py`` of https://github.com/KaiTang-eng/CoIFNet (MIT).
    Different from RevIN, RevON computes the mean and standard deviation over observed entries
    only (given the missingness mask), and zeroes out missing positions after normalization, so
    the normalization never gets contaminated by imputed/filled values.

    Parameters
    ----------
    num_features :
        The number of features or channels.

    eps :
        A value added for numerical stability.

    affine :
        Whether to use learnable affine parameters.

    subtract_last :
        Whether to subtract the last value instead of the mean. Kept for parity with the
        reference implementation, but note the reference only uses the mean-based mode.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str, mask: torch.Tensor = None) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"RevON mode {mode} is not implemented.")
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor, mask: torch.Tensor = None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            # the reference's subtract_last branch is not mask-aware, hence only used when no mask is given
            self.last = x[:, -1, :].unsqueeze(1)
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            if mask is not None:
                # statistics over observed entries only, i.e. x is supposed to be zero-filled at
                # missing positions, exactly as pypots datasets provide (X, missing_mask) pairs
                valid_sum = x.sum(dim=dim2reduce, keepdim=True)
                valid_count = mask.sum(dim=dim2reduce, keepdim=True)
                self.mean = (valid_sum / (valid_count + self.eps)).detach()

                squared_deviation = ((x - self.mean) ** 2) * mask
                variance = squared_deviation.sum(dim=dim2reduce, keepdim=True) / (valid_count + self.eps)
                self.stdev = torch.sqrt(variance + self.eps).detach()
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
                self.stdev = torch.sqrt(
                    torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
                ).detach()

    def _normalize(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight
                x = x + self.affine_bias
            # zero out missing positions so that the fusion modules never see normalized values
            # at unobserved positions, mirroring the reference implementation
            return x * mask
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class GEGLU(nn.Module):
    """Gated GELU activation, ported from the reference's ``model/CoIFNet.py``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class Attend(nn.Module):
    """Scaled dot-product attention with optional value gating.

    A dependency-light port of ``model/attend.py`` of https://github.com/KaiTang-eng/CoIFNet (MIT).
    The reference version probes the GPU type and dispatches to flash/math/mem-efficient kernels
    accordingly, which is unnecessary here, so this port simply uses
    ``torch.nn.functional.scaled_dot_product_attention`` when available (torch>=2.0) and falls
    back to an explicit einsum implementation otherwise.
    """

    def __init__(self, dropout: float = 0.0, flash: bool = True, causal: bool = False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.flash = flash and hasattr(F, "scaled_dot_product_attention")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.flash:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=self.causal,
                dropout_p=self.dropout if self.training else 0.0,
            )

        scale = q.shape[-1] ** -0.5
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=sim.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


class TSBlock(nn.Module):
    """The MLP block with GEGLU gating used by the fusion modules, ported from the reference."""

    def __init__(self, input_dim: int, output_dim: int, mid_hidden: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, mid_hidden * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LinearBlock(nn.Module):
    """The plain linear block, ported from the reference."""

    def __init__(self, input_dim: int, output_dim: int, mid_hidden: int, dropout: float):
        super().__init__()
        self.block = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionBlock(nn.Module):
    """The attention block, ported from the reference (einops rearrange included)."""

    def __init__(self, dim_in: int, dim_out: int, dim_head: int = 32, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_qkv = nn.Sequential(
            nn.Linear(dim_in, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads),
        )
        self.to_v_gates = nn.Sequential(
            nn.Linear(dim_in, heads, bias=False),
            nn.Sigmoid(),
            Rearrange("b n h -> b h n 1", h=heads),
        )
        self.attend = Attend(dropout=dropout)
        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(dim_inner, dim_out, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x)
        out = self.attend(q, k, v)
        out = out * self.to_v_gates(x)
        return self.to_out(out)


def get_block(block_name: str, input_dim: int, output_dim: int, mid_hidden: int, dropout: float) -> nn.Module:
    if block_name == "TSBlock":
        return TSBlock(input_dim, output_dim, mid_hidden, dropout)
    elif block_name == "LinearBlock":
        return LinearBlock(input_dim, output_dim, mid_hidden, dropout)
    elif block_name == "AttentionBlock":
        return AttentionBlock(input_dim, output_dim)
    else:
        raise NotImplementedError(f"block {block_name} not implemented")


class SharedModule(nn.Module):
    """The fusion module assembling the Cross-Timestep Fusion (CTF) and Cross-Variate Fusion (CVF).

    Ported from ``SharedModule`` in the reference's ``model/CoIFNet.py``. It concatenates the
    missingness mask (and optionally time-feature embeddings) to the input along the feature
    dimension, then applies the CTF block (``intra_model`` in the reference) over the time axis
    and the CVF block (``inter_model`` in the reference) over the variate axis.

    Parameters
    ----------
    n_steps :
        The number of time steps in the input window, i.e. the length fed to the CTF block.

    out_steps :
        The number of time steps the CTF block should output (the horizon length, or the hidden
        length when an auxiliary head is used afterwards).

    n_features :
        The number of features (variates) in the time-series data sample.

    use_mask :
        Whether to concatenate the missingness mask to the input, the key mechanism making
        CoIFNet directly forecast on incomplete input.

    use_time_features :
        Whether to concatenate timestamp features to the input.

    use_time_feature_embedding :
        Whether to embed the day-of-week and time-of-day features with learnable embeddings,
        as done in the reference implementation. If False while ``use_time_features`` is True,
        the raw time features (4 dimensions) are concatenated directly.

    temp_dim_tid :
        The embedding dimension for the time-in-day feature.

    temp_dim_tiw :
        The embedding dimension for the time-in-week (day-of-week) feature.

    d_model :
        The dimension of the hidden layer inside the fusion blocks, named ``hidden`` in the
        reference implementation.

    dropout :
        The dropout rate for the fusion blocks.

    intra_type :
        The type of the Cross-Timestep Fusion block, one of ["TSBlock", "LinearBlock", "AttentionBlock"].

    inter_type :
        The type of the Cross-Variate Fusion block, one of ["TSBlock", "LinearBlock", "AttentionBlock"].
    """

    def __init__(
        self,
        n_steps: int,
        out_steps: int,
        n_features: int,
        use_mask: bool,
        use_time_features: bool,
        use_time_feature_embedding: bool,
        temp_dim_tid: int,
        temp_dim_tiw: int,
        d_model: int,
        dropout: float,
        intra_type: str,
        inter_type: str,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.use_time_features = use_time_features
        self.use_time_feature_embedding = use_time_feature_embedding

        self.time_of_day_size = 24
        self.day_of_week_size = 7

        in_channel_dim = n_features
        if self.use_mask:
            in_channel_dim += n_features  # mask

        feat_emb_dim = 0
        if self.use_time_features:
            if self.use_time_feature_embedding:
                feat_emb_dim = temp_dim_tid + temp_dim_tiw
                self.time_in_day_emb = nn.Embedding(self.time_of_day_size, temp_dim_tid)
                self.day_in_week_emb = nn.Embedding(self.day_of_week_size, temp_dim_tiw)
            else:
                feat_emb_dim = 4

        in_channel_dim += feat_emb_dim

        # Cross-Timestep Fusion (CTF), `intra_model` in the reference code
        self.intra_model = get_block(
            intra_type,
            n_steps,
            out_steps,
            d_model,
            dropout,
        )
        # Cross-Variate Fusion (CVF), `inter_model` in the reference code
        self.inter_model = get_block(
            inter_type,
            in_channel_dim,
            n_features,
            d_model,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        feat: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_mask:
            x = torch.concat([x, mask], dim=-1)  # [B, L, C] -> [B, L, C*2]

        if self.use_time_features:
            if self.use_time_feature_embedding:
                feat_emb = []
                feat_emb.append(self.day_in_week_emb(feat[..., 0].long()))
                feat_emb.append(self.time_in_day_emb(feat[..., 1].long()))
            else:
                feat_emb = [feat]
            x = torch.cat([x] + feat_emb, dim=-1)

        # fuse across time steps, then across variates
        x = self.intra_model(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.inter_model(x)
        return x


class _CoIFNet(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        use_reconstruct: bool,
        use_mask: bool,
        use_time_features: bool,
        use_time_feature_embedding: bool,
        temp_dim_tid: int,
        temp_dim_tiw: int,
        d_model: int,
        dropout: float,
        intra_type: str,
        inter_type: str,
        use_head: bool,
        loss_lambda: float,
        use_reversible_norm: bool,
        use_revon: bool,
        revin_affine: bool,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.use_reconstruct = use_reconstruct
        self.use_mask = use_mask
        self.use_time_features = use_time_features
        self.use_head = use_head
        self.loss_lambda = loss_lambda
        self.use_reversible_norm = use_reversible_norm

        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # the joint imputation-forecasting objective outputs the reconstruction of the observed
        # window followed by the forecast of the horizon, as CoIFNetTask does in the reference
        horizon_len = n_steps + n_pred_steps if use_reconstruct else n_pred_steps

        if use_head:
            self.shared_model = SharedModule(
                n_steps=n_steps,
                out_steps=d_model,
                n_features=n_features,
                use_mask=use_mask,
                use_time_features=use_time_features,
                use_time_feature_embedding=use_time_feature_embedding,
                temp_dim_tid=temp_dim_tid,
                temp_dim_tiw=temp_dim_tiw,
                d_model=d_model,
                dropout=dropout,
                intra_type=intra_type,
                inter_type=inter_type,
            )
            self.aux_head = nn.Linear(d_model, horizon_len)
        else:
            self.shared_model = SharedModule(
                n_steps=n_steps,
                out_steps=horizon_len,
                n_features=n_features,
                use_mask=use_mask,
                use_time_features=use_time_features,
                use_time_feature_embedding=use_time_feature_embedding,
                temp_dim_tid=temp_dim_tid,
                temp_dim_tiw=temp_dim_tiw,
                d_model=d_model,
                dropout=dropout,
                intra_type=intra_type,
                inter_type=inter_type,
            )

        if use_reversible_norm:
            if use_revon:
                # the missing-value-aware variant, normalizing over observed entries only
                self.norm_layer = RevON(num_features=n_features, affine=revin_affine)
            else:
                self.norm_layer = RevIN(num_features=n_features, affine=revin_affine)

        if n_pred_features != n_features:
            # CoIFNet fuses and denormalizes variates one by one, hence an extra projection is
            # needed only when the forecasting targets carry a different number of features
            self.output_projection = nn.Linear(n_features, n_pred_features)
        else:
            self.output_projection = None

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        # X is already zero-filled at missing positions by the dataset, and missing_mask marks
        # observed positions as 1, exactly as the reference implementation expects

        # keep the raw X for the reconstruction loss below, because the reference computes the
        # reconstruction loss between the raw input and the denormalized reconstruction
        X_raw = X

        if self.use_reversible_norm:
            if isinstance(self.norm_layer, RevON):
                X = self.norm_layer(X, "norm", missing_mask)
            else:
                X = self.norm_layer(X, "norm")

        reconstruction, forecasting_result = self._forward_shared(X, missing_mask, inputs.get("time_features", None))

        if self.output_projection is not None:
            forecasting_result = self.output_projection(forecasting_result)
            if reconstruction is not None:
                reconstruction = self.output_projection(reconstruction)

        results = {
            "forecasting": forecasting_result,
        }
        if reconstruction is not None:
            # CoIFNet is a joint imputation-forecasting model, so also expose the imputed
            # lookback window for users, though only the forecast is the mandatory output
            results["reconstruction"] = reconstruction

        if calc_criterion:
            X_pred, X_pred_missing_mask = inputs["X_pred"], inputs["X_pred_missing_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # the joint objective of CoIFNet: forecasting loss on the prediction horizon
                # plus the reconstruction loss on the observed window, weighted by loss_lambda
                loss_forecast = self.training_loss(X_pred, forecasting_result, X_pred_missing_mask)
                if self.use_reconstruct:
                    loss_reconstruct = self.training_loss(X_raw, reconstruction, missing_mask)
                    # `loss` is always the item for backward propagating to update the model
                    results["loss"] = self.loss_lambda * loss_forecast + (1 - self.loss_lambda) * loss_reconstruct
                else:
                    results["loss"] = loss_forecast
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(X_pred, forecasting_result, X_pred_missing_mask)

        return results

    def _forward_shared(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        time_features: torch.Tensor = None,
    ):
        """Run the fusion modules and the reversible denormalization, then split the output."""

        if self.use_time_features and time_features is None:
            raise ValueError(
                "CoIFNet is configured with use_time_features=True but no time features are given. "
                "Note pypots forecasting datasets do not carry timestamp features, hence "
                "use_time_features should be left as False unless a dataset yielding "
                "'time_features' is passed through a customized model input."
            )

        # the fusion part, `impute()` in the reference's CoIFNet model
        output = self.shared_model(X, missing_mask, time_features)
        if self.use_head:
            output = self.aux_head(output.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_reversible_norm:
            output = self.norm_layer(output, "denorm")
        output = output.contiguous()

        if self.use_reconstruct:
            # the output covers the reconstruction of the observed window and the forecast
            reconstruction, forecasting_result = output[:, : self.n_steps], output[:, self.n_steps :]
        else:
            reconstruction, forecasting_result = None, output
        return reconstruction, forecasting_result
