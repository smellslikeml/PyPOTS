# CoIFNet validation plan

This file records the validation status of the CoIFNet port in PyPOTS and describes the
benchmark-parity run that must happen **before upstreaming**. No benchmark parity is claimed
here — that run is deferred to a human/GPU step.

## What is already verified (CPU smoke, CI-runnable)

`tests/forecasting/coifnet.py` runs on CPU as part of the regular test suite and asserts:

1. `fit` runs for 2 epochs on partially-observed synthetic data (random-walk data with
   injected missingness, missing rate 0.1) and produces model parameters / best loss /
   checkpoints exactly like the sibling forecasting models;
2. `predict`/`forecast` return finite results with shape
   `[n_samples, n_pred_steps, n_features]`, and the joint model also returns its
   `reconstruction` of the observed window with shape `[n_samples, n_steps, n_features]`;
3. the training data is confirmed partially observed and 30 optimizer steps on a fixed
   batch **decrease** the joint imputation-forecasting loss (smoke-level convergence);
4. saving/loading and lazy loading (h5) round-trip.

What the smoke test proves: the port is wired correctly into the PyPOTS forecaster contract
(input keys `X`/`missing_mask`/`X_pred`/`X_pred_missing_mask`, train-mode `loss` / eval-mode
`metric` returns), the mask path is exercised on partially-observed data, and the model
learns. What it does **not** prove: benchmark parity with the paper's numbers.

## The real gate: BenchPOTS forecasting parity run (human/GPU, deferred)

Objective: confirm the port reproduces the paper's claim (≈24% improvement over SOTA at 0.6
missing rate) before merging upstream. Suggested protocol, mirroring the reference paper's
setup (ETT-style datasets with point and block missingness):

| Item | Value |
| --- | --- |
| Datasets | ETTh1, ETTh2, ETTm1, ETTm2 (and optionally Weather / Electricity) |
| Missingness | point (MAR) at rates {0.1, 0.3, 0.6} and block missingness, via `pygrinder` / `benchpots` |
| Lookback / horizon | seq_len 96, pred_len {96, 192, 336, 720} (reference defaults) |
| Baselines | the existing PyPOTS forecasters `CSDI` and `TimeMixer` (impute-then-forecast vs joint), plus SAITS→forecast as an impute-first reference |
| Metrics | masked MSE and masked MAE on the horizon, computed only on observed positions (BenchPOTS defaults) |
| Hyperparameters | reference defaults: hidden 256, dropout 0.1, TSBlock×TSBlock, use_head, use_reconstruct, RevON(affine), loss_lambda 0.1, masked-MAE joint objective; tune only lr/epochs/batch_size per dataset |
| Seeds | ≥3 seeds, report mean ± std |
| Hardware | single GPU (the model is small: a few MLP blocks) |

Command sketch (BenchPOTS):

```python
from benchpots.datasets import preprocess_ett  # e.g. ETTh1

for rate in [0.1, 0.3, 0.6]:
    for pred_len in [96, 192, 336, 720]:
        dataset = preprocess_ett(subset="etth1", rate=rate, n_steps=96, n_pred_steps=pred_len, ...)
        for model in [CoIFNet, CSDI, TimeMixer]:
            ...  # fit/predict, collect masked MSE/MAE
```

Acceptance bar: CoIFNet's masked MAE/MSE on the horizon beats the impute-then-forecast
baselines at the 0.6 missing rate by a margin consistent with the paper (direction and rough
magnitude), and degrades gracefully (not catastrophically) at low missingness.

## Highest-risk areas for the parity run (what to check first if numbers look off)

1. **Mask handling** — `X` must stay zero-filled at missing positions with `missing_mask`
   marking observed entries; the mask is concatenated channel-wise into the CVF input
   (`use_mask=True`). A silently-dropped or inverted mask is the most likely silent failure.
2. **RevON observed-only normalization** — statistics are computed over observed entries only
   and missing positions are zeroed after normalization; denormalization reuses the input
   window's statistics. Check a fully-missing variate degenerates without NaNs.
3. **CTF/CVF fidelity** — CTF mixes along the time axis (`intra_model`, Linear over the
   sequence dimension) and CVF along the variate axis (`inter_model`); the order
   (mask/feature concatenation → CTF → CVF) and the `use_head` auxiliary projection
   (hidden → seq_len+pred_len) must match the reference.
