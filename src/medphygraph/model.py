"""P0-10: Compact DyPhyGraph-Health support predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# Feature layout from features.FEATURE_NAMES
IDX_DISP = 0
IDX_DISP_Z = 1
IDX_VEL_Z = 2
IDX_CUM_Z = 3
IDX_CONTACT = 4
IDX_HOST_RM = 5
IDX_GEOM_SEP = 6
IDX_GEOM_GAP = 7
IDX_MASK = 8

TEMPORAL_IDX = (IDX_DISP, IDX_DISP_Z, IDX_VEL_Z, IDX_CUM_Z, IDX_CONTACT, IDX_HOST_RM)
GEOM_IDX = (IDX_GEOM_SEP, IDX_GEOM_GAP)


@dataclass
class ModelConfig:
    temporal_dim: int = 6
    geom_dim: int = 2
    hidden: int = 64
    gru_layers: int = 1
    dropout: float = 0.1


def gather_observed_sequence(
    temp: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact each batch row's temporal features down to only the frames
    where ``mask == 1``, preserving original temporal order, so the result is
    safe to feed to ``pack_padded_sequence`` (which assumes the first
    ``length`` positions of each row ARE the valid sequence in order).

    ``apply_incompleteness`` (features.py) marks observed frames at
    *scattered* temporal indices (e.g. ``0 0 1 0 1 0 0 1``), not necessarily a
    contiguous prefix. Naively zeroing unobserved frames and calling
    ``pack_padded_sequence(..., lengths=mask.sum(...))`` is therefore
    semantically wrong: it silently treats the first ``length`` raw frames
    (a mix of observed and unobserved positions) as if they were a
    contiguous observed prefix, discarding real observations that land at
    later temporal indices and feeding never-observed (zeroed) frames to the
    GRU as if they were data.

    This function fixes that by explicitly gathering the ``mask == 1`` rows
    (in ascending temporal order -- never reordered/shuffled) into a new,
    shorter, contiguous sequence per batch element.

    Args:
        temp: ``[B, T, F]`` temporal feature channels only.
        mask: ``[B, T]`` observation mask (1 = observed, 0 = unobserved).

    Returns:
        compact: ``[B, L, F]`` zero-padded, gathered observed-only sequence
            (``L = max over the batch of the true observed-frame count``).
        lengths: ``[B]`` true observed-frame count per batch element (>= 1).
        last_observed_idx: ``[B]`` the actual temporal index (into the
            ORIGINAL ``[0, T)`` axis) of each batch element's last observed
            frame -- NOT ``lengths - 1``, which is only correct when the
            mask happens to be a contiguous prefix.
    """
    if temp.dim() != 3 or mask.dim() != 2:
        raise ValueError(f"expected temp [B,T,F] and mask [B,T], got {tuple(temp.shape)} / {tuple(mask.shape)}")
    b_size, t_size, f_size = temp.shape
    observed = mask > 0.5
    lengths = observed.sum(dim=1).clamp(min=1).long()
    max_len = int(lengths.max().item()) if b_size else 0
    compact = temp.new_zeros(b_size, max(max_len, 1), f_size)
    last_observed_idx = torch.zeros(b_size, dtype=torch.long, device=temp.device)
    for b in range(b_size):
        obs_idx = torch.nonzero(observed[b], as_tuple=True)[0]  # ascending -> original order preserved
        n = int(obs_idx.numel())
        if n == 0:
            # Defensive only: apply_incompleteness() always keeps >= 1 frame,
            # so this never triggers on data produced by the frozen dataset
            # pipeline. Falls back to frame 0 rather than raising.
            last_observed_idx[b] = 0
            continue
        compact[b, :n] = temp[b, obs_idx]
        last_observed_idx[b] = obs_idx[-1]
    return compact, lengths, last_observed_idx


class HealthDyPhyGraph(nn.Module):
    """Static geometry encoder + GRU over partial CF observations + fusion head.

    Novelty is the incomplete-counterfactual + graph pipeline, not architecture size.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        self.geom_enc = nn.Sequential(
            nn.Linear(self.cfg.geom_dim, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, h // 2),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            self.cfg.temporal_dim,
            h,
            num_layers=self.cfg.gru_layers,
            batch_first=True,
            dropout=self.cfg.dropout if self.cfg.gru_layers > 1 else 0.0,
        )
        self.fuse = nn.Sequential(
            nn.Linear(h + h // 2, h),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.cls = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, F] with F matching FEATURE_NAMES (mask in last col).
        Returns (logits [B], probs [B]).

        Pre-training integrity repair (Issue 1): observed frames are gathered
        into a contiguous, order-preserving sequence (see
        ``gather_observed_sequence``) before packing, so `pack_padded_sequence`
        only ever sees genuinely observed frames -- correct for the scattered
        (non-prefix) masks `apply_incompleteness` produces. For ratio=1.0
        (mask all-ones) this is numerically equivalent to the previous
        implementation; see `_forward_legacy_prefix_packing` and
        `tests/test_sequence.py` for the regression
        proof.
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B,T,F], got {tuple(x.shape)}")
        mask = x[:, :, IDX_MASK]  # [B,T]
        temp = x[:, :, list(TEMPORAL_IDX)]

        compact, lengths, last_observed_idx = gather_observed_sequence(temp, mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_t = h_n[-1]  # [B,H]

        # geometry from any observed frame (constant across t); use the TRUE
        # last observed temporal index, not lengths-1 (Issue 1B).
        geom = x[:, :, list(GEOM_IDX)]
        b = torch.arange(x.size(0), device=x.device)
        geom_v = geom[b, last_observed_idx]
        h_g = self.geom_enc(geom_v)

        h = self.fuse(torch.cat([h_t, h_g], dim=-1))
        logits = self.cls(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def _forward_legacy_prefix_packing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """DEPRECATED reference implementation -- kept ONLY for the
        pre-training integrity repair's full-observation equivalence
        regression test. Do NOT use for training or evaluation.

        This is a byte-for-byte copy of the original (buggy) forward pass:
        it zeroes unobserved frames in place and calls
        `pack_padded_sequence(temp, mask.sum(...))`, which implicitly
        assumes the observed frames form a contiguous PREFIX of the
        sequence. That assumption is false for the scattered masks
        `apply_incompleteness` actually produces (ratio < 1.0), so this path
        is scientifically incorrect for partial observation and must never
        be used outside of the mask-all-ones (ratio=1.0) equivalence check.
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B,T,F], got {tuple(x.shape)}")
        mask = x[:, :, IDX_MASK]  # [B,T]
        temp = x[:, :, list(TEMPORAL_IDX)]
        temp = temp * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            temp, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_t = h_n[-1]  # [B,H]

        geom = x[:, :, list(GEOM_IDX)]
        idx = (lengths - 1).clamp(min=0)
        b = torch.arange(x.size(0), device=x.device)
        geom_v = geom[b, idx]
        h_g = self.geom_enc(geom_v)

        h = self.fuse(torch.cat([h_t, h_g], dim=-1))
        logits = self.cls(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> HealthDyPhyGraph:
    return HealthDyPhyGraph(cfg)
