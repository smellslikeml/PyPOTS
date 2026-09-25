# CoIFNet

CoIFNet (*A Unified Framework for Multivariate Time Series Forecasting with Missing Values*,
Tang et al., [arXiv:2506.13064](https://arxiv.org/abs/2506.13064)) is a **joint
imputation-forecasting** model: it forecasts directly on incomplete input instead of
imputing first and forecasting afterwards. It fuses observed values, the missingness mask,
and (optionally) timestamp features through two modules — **Cross-Timestep Fusion (CTF)** and
**Cross-Variate Fusion (CVF)** — with reversible normalization (RevIN, or the
missing-value-aware RevON that normalizes over observed entries only).

## Provenance

Port with attribution from the official MIT-licensed implementation
[KaiTang-eng/CoIFNet](https://github.com/KaiTang-eng/CoIFNet):

- ported: `model/CoIFNet.py` (the fusion architecture), `model/RevIN.py`, `model/RevON.py`,
  and `model/attend.py` (attention), plus the joint objective of `model/CoIFNetTask.py`;
- deliberately **not** ported: the reference's `conf/`, `dataset/`, `exp/`, `trainer/`
  scaffolding and its Task/config classes — `BaseNNForecaster` owns the training loop, data
  loading, and configuration in PyPOTS.

## Usage

```python
from pypots.forecasting import CoIFNet

coifnet = CoIFNet(
    n_steps=24,
    n_features=7,
    n_pred_steps=6,
    n_pred_features=7,
    # fusion hidden dim ("hidden" in the reference), defaults shown below
    d_model=256,
    dropout=0.1,
    use_reconstruct=True,   # joint imputation-forecasting objective
    use_mask=True,          # concatenate the missingness mask, the key POTS mechanism
    use_reversible_norm=True,
    use_revon=True,         # observed-only normalization statistics
    loss_lambda=0.1,        # weight of the forecasting loss in the joint objective
    batch_size=32,
    epochs=100,
)
coifnet.fit(train_set, val_set)  # train_set/val_set contain keys "X" and "X_pred"
forecasting = coifnet.forecast(test_set)  # [n_samples, n_pred_steps, n_features]
```

`X` may contain missing values (NaN). The model also returns its imputed lookback window
under the key `reconstruction` when `use_reconstruct=True`.

Note: the reference optionally consumes timestamp features (time-of-day / day-of-week
embeddings). PyPOTS forecasting datasets do not carry timestamps, so `use_time_features`
defaults to `False`; enable it only if you pass `time_features` in the model input dictionary.

## Validation status

See [VALIDATION.md](VALIDATION.md) for the BenchPOTS benchmark-parity plan (deferred to a
human/GPU run). The CPU smoke test lives at `tests/forecasting/coifnet.py`.
