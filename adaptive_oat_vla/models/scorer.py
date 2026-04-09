"""AdapTok-inspired z_q-only scorer for OAT prefix-quality prediction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ActionQualityScorer(nn.Module):
    """Predict one reconstruction score per candidate prefix length from quantized OAT latents.

    This keeps the AdapTok-style "score along the latent sequence, then read out the
    candidate positions" idea, but intentionally uses only `z_q` so train and inference
    see the same information in the OAT-VLA pipeline.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        latent_horizon: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        candidate_ks: Sequence[int] = (1, 2, 4, 8),
        causal: bool = True,
    ):
        super().__init__()
        self.candidate_ks = [int(k) for k in candidate_ks]
        if any(k < 1 or k > latent_horizon for k in self.candidate_ks):
            raise ValueError(
                f"candidate_ks must be within [1, {latent_horizon}], got {self.candidate_ks}"
            )
        self.candidate_indices = [k - 1 for k in self.candidate_ks]
        self.latent_horizon = int(latent_horizon)
        self.causal = bool(causal)

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_emb = nn.Parameter(
            (hidden_dim ** -0.5) * torch.randn(1, latent_horizon, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, 1),
        )

    def _build_attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward_per_token(self, z_q: torch.Tensor) -> torch.Tensor:
        """Return one predicted score per latent position."""
        seq_len = z_q.shape[1]
        x = self.input_proj(z_q) + self.pos_emb[:, :seq_len, :]
        attn_mask = self._build_attn_mask(seq_len, z_q.device)
        x = self.encoder(x, mask=attn_mask)
        x = self.norm(x)
        return self.score_head(x).squeeze(-1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Return one predicted reconstruction score per candidate prefix length."""
        per_token_scores = self.forward_per_token(z_q)
        index = torch.as_tensor(self.candidate_indices, device=z_q.device)
        return per_token_scores.index_select(dim=1, index=index)


if __name__ == "__main__":
    scorer = ActionQualityScorer()
    dummy = torch.randn(3, 8, 4)
    out = scorer(dummy)
    per_token = scorer.forward_per_token(dummy)
    print("output_shape", tuple(out.shape))
    print("per_token_shape", tuple(per_token.shape))
