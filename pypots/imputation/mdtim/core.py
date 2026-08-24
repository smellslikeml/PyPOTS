"""
The core wrapper assembles the submodules of the MDTIM imputation model
and takes over the forward progress of the algorithm.

Refer to the paper
`Dongbin Kim, Seungyun Lee, Geonwoo Shin, and Jaewook Lee.
Discretizing Continuous Time Series for Imputation with Masked Diffusion Training.
arXiv preprint arXiv:2608.19119, 2026.
<https://arxiv.org/abs/2608.19119>`_
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules import ModelCore


class TemporalVariateDiTBlock(nn.Module):
    """One interleaved temporal/variate attention block of the MDTIM backbone.

    Follows Sec. 4.1 of arXiv:2608.19119: the hidden state H ∈ R^(B×T×C×D) is reshaped
    to (B·C, T, D) for temporal self-attention and to (B·T, C, D) for variate
    self-attention, both conditioned on the diffusion timestep through adaptive
    layer-norm (adaLN-Zero, as in DiT of Peebles & Xie).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2, d_conditioning: int = 16):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}."
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        # adaLN-Zero: the timestep embedding yields scale/shift/gate for the attention
        # and the MLP sublayers; gates are zero-initialized so each block is an identity
        self.conditioning = nn.Sequential(nn.SiLU(), nn.Linear(d_conditioning, 6 * d_model))
        nn.init.zeros_(self.conditioning[-1].weight)
        nn.init.zeros_(self.conditioning[-1].bias)

        self.norm_attention = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def _sublayer(self, H: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply the modulated attention+MLP sublayer to a (N, len, D) tensor."""
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.conditioning(t_emb).unsqueeze(1).chunk(6, dim=-1)
        )
        normed = self.norm_attention(H) * (1 + scale_attn) + shift_attn
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        H = H + gate_attn * attended
        normed = self.norm_mlp(H) * (1 + scale_mlp) + shift_mlp
        return H + gate_mlp * self.mlp(normed)

    def forward(self, H: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward the block.

        Parameters
        ----------
        H :
            Hidden state of shape (B, T, C, D).

        t_emb :
            Diffusion-timestep embedding of shape (B, d_conditioning).

        Returns
        -------
        H :
            The updated hidden state of shape (B, T, C, D).
        """
        B, T, C, D = H.shape
        # temporal attention over T, batched over samples and variates
        H_temporal = H.permute(0, 2, 1, 3).reshape(B * C, T, D)
        H_temporal = self._sublayer(H_temporal, t_emb.repeat_interleave(C, dim=0))
        # variate attention over C, batched over samples and time steps
        H_variate = H_temporal.reshape(B, C, T, D).permute(0, 2, 1, 3).reshape(B * T, C, D)
        H_variate = self._sublayer(H_variate, t_emb.repeat_interleave(T, dim=0))
        return H_variate.reshape(B, T, C, D)


class _MDTIM(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_bins: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_conditioning: int = 16,
        d_time_embedding: int = 32,
        dropout: float = 0.2,
        spectral_weight: float = 1.0,
        ordinal_sigma: float = 1.0,
        ordinal_window: int = 2,
        input_range: float = 1.5,
    ):
        super().__init__()
        assert n_bins >= 2, f"n_bins must be at least 2, but got {n_bins}."
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_bins = n_bins
        self.spectral_weight = spectral_weight
        self.ordinal_sigma = ordinal_sigma
        self.ordinal_window = ordinal_window

        # The output vocabulary K_out = 1.5K bins covers [-1.5, 1.5] (Table 10), giving
        # the head room for the distribution shift in unobserved regions (Sec. 4.1).
        self.input_range = input_range
        self.vocab_size = int(round(n_bins * input_range))
        # bin centers v_k over [-input_range, input_range], used for expectation decoding
        self.register_buffer(
            "bin_centers",
            torch.linspace(-input_range, input_range, self.vocab_size),
            persistent=False,
        )

        # token embeddings for the K observed bins plus one [MASK] absorbing state,
        # added to per-variate channel embeddings and time-point encodings (Sec. 4.1)
        self.mask_token_id = self.vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size + 1, d_model)
        self.channel_embedding = nn.Parameter(torch.randn(1, 1, n_features, d_model) * 0.02)
        self.d_time_embedding = d_time_embedding
        self.time_embedding = nn.Linear(d_time_embedding, d_model)
        self.cond_embedding = nn.Linear(1, d_model)
        self.timestep_embedding = nn.Sequential(
            nn.Linear(d_conditioning, d_conditioning),
            nn.SiLU(),
            nn.Linear(d_conditioning, d_conditioning),
        )

        self.blocks = nn.ModuleList(
            [
                TemporalVariateDiTBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    d_conditioning=d_conditioning,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.vocab_size))

        # Ordinal soft-label kernel s(l) (Sec. 3.2): a truncated Gaussian over adjacent
        # bins centered at the ground-truth token, with zero mass on [MASK]. Indexed by
        # ground truth at loss time.
        offsets = torch.arange(self.vocab_size).unsqueeze(0) - torch.arange(self.vocab_size).unsqueeze(1)
        kernel = torch.exp(-((offsets.float() / ordinal_sigma) ** 2))
        kernel = kernel.masked_fill(offsets.abs() > ordinal_window, 0.0)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("ordinal_kernel", kernel, persistent=False)

    @staticmethod
    def sinusoidal_encoding(positions: torch.Tensor, d_embedding: int) -> torch.Tensor:
        """Sinusoidal encoding of the observed time points, as in CSDI's side information.

        Parameters
        ----------
        positions :
            Time points of shape (B, T).

        d_embedding :
            The embedding dimension.

        Returns
        -------
        encoding :
            The encoded time points of shape (B, T, d_embedding).
        """
        device = positions.device
        pe = torch.zeros(positions.shape[0], positions.shape[1], d_embedding, device=device)
        position = positions.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_embedding, 2, device=device) / d_embedding)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Embed the diffusion timesteps with a sinusoidal table (Sec. 4.1).

        Parameters
        ----------
        t :
            Diffusion timesteps of shape (B,).

        Returns
        -------
        t_emb :
            The timestep embedding of shape (B, d_conditioning).
        """
        half = self.timestep_embedding[0].in_features // 2
        frequencies = 10.0 ** (torch.arange(half, device=t.device) / max(half - 1, 1) * 4.0).unsqueeze(0)
        table = t.float().unsqueeze(1) * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=-1)
        return self.timestep_embedding(table)

    def instance_normalize(self, X: torch.Tensor, mask: torch.Tensor):
        """Per-instance normalization computed from observed positions only (Sec. 3.1).

        Parameters
        ----------
        X :
            The filled (zero-imputed) data of shape (B, T, C).

        mask :
            The observed mask of shape (B, T, C), 1 for observed.

        Returns
        -------
        X_norm, mean, scale :
            The normalized data and the per-instance statistics for de-normalization.
        """
        safe_mask = mask.clone()
        safe_mask[safe_mask.sum(dim=(1, 2)) == 0] = 1.0  # guard fully-missing instances
        counts = safe_mask.sum(dim=(1, 2)).clamp_min(1.0)
        mean = (X * safe_mask).sum(dim=(1, 2)) / counts
        centered = (X - mean[:, None, None]) * safe_mask
        scale = (centered.pow(2).sum(dim=(1, 2)) / counts).sqrt().clamp_min(1e-5)
        return (X - mean[:, None, None]) / scale[:, None, None], mean, scale

    def discretize(self, X_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Stochastic discretization into ordinal bins (Sec. 3.1).

        The normalized value is mapped to grid coordinates over K bins, then uniform
        noise ε∼U(−0.5,0.5) is added before rounding. The noise makes tokenization
        unbiased in expectation (a coordinate of 3.6 maps to token 4 with probability
        0.6 and to token 3 with probability 0.4) and so preserves sub-grid information.

        Parameters
        ----------
        X_norm :
            The normalized data of shape (B, T, C).

        mask :
            The observed mask of shape (B, T, C).

        Returns
        -------
        tokens :
            Token indices of shape (B, T, C), `mask_token_id` at missing positions.
        """
        coords = (X_norm + self.input_range) / (2 * self.input_range) * (self.vocab_size - 1)
        noise = (torch.rand_like(coords) - 0.5) * mask  # only perturb observed values
        tokens = torch.round(coords + noise).long().clamp(0, self.vocab_size - 1)
        return torch.where(mask > 0, tokens, torch.full_like(tokens, self.mask_token_id))

    def soft_labels(self, tokens: torch.Tensor) -> torch.Tensor:
        """Ordinal-aware soft labels s(l) for the given ground-truth tokens (Sec. 3.2).

        Ground-truth tokens live on the output grid of `vocab_size` bins; positions whose
        ground truth is the [MASK] state get an all-zero label since s_0 = 0.

        Parameters
        ----------
        tokens :
            Ground-truth token indices of shape (B, T, C).

        Returns
        -------
        labels :
            Soft label distributions of shape (B, T, C, vocab_size).
        """
        labels = self.ordinal_kernel[tokens.clamp(max=self.vocab_size - 1)]
        return labels.masked_fill((tokens == self.mask_token_id).unsqueeze(-1), 0.0)

    def backbone(
        self,
        tokens: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_tp: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the factorized temporal-variate DiT backbone.

        Parameters
        ----------
        tokens :
            Token indices of shape (B, T, C), possibly containing `mask_token_id`.

        cond_mask :
            The conditioning (observed) mask of shape (B, T, C), injected as side
            information the same way CSDI feeds its conditional mask.

        observed_tp :
            The observed time points of shape (B, T).

        t :
            The diffusion timesteps of shape (B,).

        Returns
        -------
        logits :
            The predicted logits over the output vocabulary, shape (B, T, C, vocab_size).
        """
        H = self.token_embedding(tokens) + self.channel_embedding
        H = H + self.time_embedding(self.sinusoidal_encoding(observed_tp, self.d_time_embedding)).unsqueeze(2)
        H = H + self.cond_embedding(cond_mask.unsqueeze(-1))
        t_emb = self.embed_timestep(t)
        for block in self.blocks:
            H = block(H, t_emb)
        return self.output_head(H)

    def calc_loss(self, X_ori: torch.Tensor, cond_mask: torch.Tensor, indicating_mask: torch.Tensor) -> torch.Tensor:
        """The masked-diffusion training objective (Sec. 4.2).

        L_diff = E_t[ w(t) Σ_{l∈M_t} −⟨s(l), log p_θ(z_t(l))⟩ ] with t ~ U(0,1) under
        the log-linear schedule σ(t) = −log(1−t), i.e. α_t = 1−t, hence the weight
        w(t) = α_t'/(1−α_t) = 1/t. A spectral-consistency term L_FFT (adapted from
        DiffusionTS) is added with weight `spectral_weight`.

        Parameters
        ----------
        X_ori :
            The observed data of shape (B, T, C).

        cond_mask :
            The conditioning mask of shape (B, T, C); observed anchors are never corrupted.

        indicating_mask :
            The mask of artificially-held-out positions, i.e. where supervision applies.

        Returns
        -------
        loss :
            The scalar training loss.
        """
        # CSDI-style layout (B, n_features, n_steps) -> the (B, T, C) layout of the paper
        X_ori = X_ori.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        indicating_mask = indicating_mask.permute(0, 2, 1)
        B, T, C = X_ori.shape
        device = X_ori.device
        observed_tp = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        X_norm, _, _ = self.instance_normalize(X_ori, cond_mask)
        tokens = self.discretize(X_norm, cond_mask)  # ground-truth tokens

        # Corrupt the non-anchored (held-out) positions with the absorbing [MASK] state
        # with probability 1−α_t = t; observed anchors are carried over untouched.
        t = torch.rand(B, device=device).clamp(1e-4, 1 - 1e-4)
        corrupt = (torch.rand(B, T, C, device=device) > (1 - t).view(B, 1, 1)).to(X_ori.dtype)
        target_mask = indicating_mask * corrupt  # M_t = {l : z_t^l = m}
        z_t = torch.where(target_mask > 0, torch.full_like(tokens, self.mask_token_id), tokens)

        logits = self.backbone(z_t, cond_mask, observed_tp, t)
        log_p = F.log_softmax(logits, dim=-1)
        token_loss = -(self.soft_labels(tokens) * log_p).sum(dim=-1)  # −⟨s(l), log p_θ(z_t(l))⟩

        # w(t) = α_t'/(1−α_t); with α_t = 1−t the numerator α_t' = 1, so w(t) = 1/t.
        # The weight is per-sample, so it scales each sample's masked positions together.
        w = (1.0 / t).view(B, 1, 1)
        diff_loss = (token_loss * target_mask * w).sum() / (target_mask * w).sum().clamp_min(1e-8)

        loss = diff_loss

        if self.spectral_weight > 0:
            # L_FFT = E_t[ ‖𝓕(x̂_t) − 𝓕(x_0)‖₁ ] on the expectation-decoded estimate
            X_hat = torch.softmax(logits, dim=-1) @ self.bin_centers
            fft_loss = (torch.fft.rfft(X_hat, dim=1) - torch.fft.rfft(X_norm * cond_mask, dim=1)).abs().sum(
                dim=(1, 2)
            ).mean() / (T * C)
            loss = loss + self.spectral_weight * fft_loss

        return loss

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
        n_sampling_times: int = 1,
    ) -> dict:
        results = {}
        if calc_criterion:
            (X_ori, indicating_mask, cond_mask) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
            )
            if self.training:  # for training
                results["loss"] = self.calc_loss(X_ori, cond_mask, indicating_mask)
            else:  # for validating
                with torch.no_grad():
                    results["metric"] = self.calc_loss(X_ori, cond_mask, indicating_mask)
        else:
            (X, cond_mask, observed_tp) = (inputs["X"], inputs["cond_mask"], inputs["observed_tp"])
            # CSDI-style layout (B, n_features, n_steps) -> the (B, T, C) layout of the paper
            X = X.permute(0, 2, 1)
            cond_mask = cond_mask.permute(0, 2, 1)
            B, T, C = X.shape
            device = X.device

            X_norm, mean, scale = self.instance_normalize(X, cond_mask)

            # Decoding via expectation (Sec. 4.3): run M passes with independent
            # tokenization noise, average the predicted token distributions
            # p̄ = (1/M) Σ_m p^(m), then reconstruct continuously as
            # x̂ = Σ_k p̄(k)·v_k over the bin centers v_k and de-normalize.
            prob_sum = torch.zeros(B, T, C, self.vocab_size, device=device)
            for _ in range(n_sampling_times):
                tokens = self.discretize(X_norm, cond_mask)
                # start the reverse process from the fully-masked state (t=1 → α_t=0) on
                # missing positions, then refine over a decreasing ladder of timesteps
                z = torch.where(cond_mask > 0, tokens, torch.full_like(tokens, self.mask_token_id))
                for t_step in [0.75, 0.5, 0.25]:
                    t = torch.full((B,), t_step, device=device)
                    logits = self.backbone(z, cond_mask, observed_tp, t)
                    sampled = torch.multinomial(F.softmax(logits, dim=-1).reshape(B * T * C, -1), 1).reshape(B, T, C)
                    # carry-over: unmasked tokens are preserved, only masked positions update
                    z = torch.where(cond_mask > 0, tokens, sampled)

                prob_sum += F.softmax(self.backbone(z, cond_mask, observed_tp, torch.zeros(B, device=device)), dim=-1)

            p_mean = prob_sum / n_sampling_times
            imputed_norm = p_mean @ self.bin_centers
            imputed = imputed_norm * scale[:, None, None] + mean[:, None, None]
            # keep the per-sample estimates for the sampling dimension of the output
            imputed_samples = imputed.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)

            # observed values are kept as-is and only missing positions take the estimate
            imputed_data = cond_mask.unsqueeze(1) * X.unsqueeze(1) + (1 - cond_mask.unsqueeze(1)) * imputed_samples
            results["imputation"] = imputed_data  # (n_samples, M, n_steps, n_features)
            # the continuous estimates are taken as the reconstruction output to keep
            # the result dict aligned with the other imputation models
            results["reconstruction"] = imputed_samples

        return results
