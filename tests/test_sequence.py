"""Pre-training integrity repair -- Issue 1: mask-aware sequence packing.

Regression tests proving that scattered (non-prefix) observation masks are
gathered correctly -- observed frames only, original order preserved, true
count used, no late-observation discarding -- and that the corrected
`HealthDyPhyGraph.forward` is numerically equivalent to the previous
(pre-repair) implementation when the mask is all-ones (ratio=1.0).
"""

from __future__ import annotations

import torch

from medphygraph.model import (
    HealthDyPhyGraph,
    ModelConfig,
    gather_observed_sequence,
)

SCATTERED_MASKS = [
    [0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


def _make_temp(t: int, f: int, offset: float = 0.0) -> torch.Tensor:
    """Distinct per-frame values (frame index encoded in the values) so we
    can verify exactly which original frames were gathered."""
    base = torch.arange(t, dtype=torch.float32).unsqueeze(-1).expand(t, f).clone()
    return base + offset


def test_gather_observed_sequence_contains_exactly_masked_frames_in_order() -> None:
    f = 3
    for mask_list in SCATTERED_MASKS:
        t = len(mask_list)
        temp = _make_temp(t, f).unsqueeze(0)  # [1,T,F], value at frame i is i (broadcast over F)
        mask = torch.tensor([mask_list], dtype=torch.float32)  # [1,T]

        compact, lengths, last_idx = gather_observed_sequence(temp, mask)

        observed_frame_indices = [i for i, m in enumerate(mask_list) if m == 1]
        true_count = len(observed_frame_indices)

        # true observed-frame count is used (not len(mask) and not len-based on a prefix guess)
        assert int(lengths[0]) == true_count, mask_list

        # gathered sequence contains exactly the mask==1 frames, in original order,
        # with no reordering and no late-observation discarding.
        gathered_values = compact[0, :true_count, 0].tolist()
        assert gathered_values == [float(i) for i in observed_frame_indices], mask_list

        # padded tail (if any) must be zero, not garbage / not a repeated frame.
        if compact.shape[1] > true_count:
            assert torch.all(compact[0, true_count:] == 0.0)

        # last observed index must be the TRUE last mask==1 position, not lengths-1.
        assert int(last_idx[0]) == observed_frame_indices[-1], mask_list


def test_gather_observed_sequence_late_observation_not_discarded() -> None:
    """[0,0,1,0,1,0,0,1]: last observed frame is index 7, count is 3.
    lengths-1 (=2) would WRONGLY point at frame index 2 under the old logic.
    """
    mask_list = [0, 0, 1, 0, 1, 0, 0, 1]
    t = len(mask_list)
    temp = _make_temp(t, 2).unsqueeze(0)
    mask = torch.tensor([mask_list], dtype=torch.float32)

    compact, lengths, last_idx = gather_observed_sequence(temp, mask)

    assert int(lengths[0]) == 3
    assert int(last_idx[0]) == 7  # true last observed frame, NOT lengths-1==2
    # the frame at t=7 (the late observation) must be present in the gathered sequence
    assert 7.0 in compact[0, :, 0].tolist()
    assert compact[0, 2, 0].item() == 7.0  # last gathered slot holds frame 7's data


def test_gather_observed_sequence_batched_variable_counts() -> None:
    """Different batch elements may have different true observed counts;
    padding must not corrupt shorter sequences or their `lengths`."""
    masks = [
        [0, 0, 1, 0, 1, 0, 0, 1],  # count 3
        [1, 0, 0, 0, 0, 0, 0, 1],  # count 2
        [1, 1, 1, 1, 1, 1, 1, 1],  # count 8
    ]
    t, f = 8, 2
    temp = torch.stack([_make_temp(t, f, offset=100.0 * b) for b in range(len(masks))], dim=0)
    mask = torch.tensor(masks, dtype=torch.float32)

    compact, lengths, last_idx = gather_observed_sequence(temp, mask)

    assert compact.shape == (3, 8, 2)  # padded to max true count (8)
    assert lengths.tolist() == [3, 2, 8]
    assert last_idx.tolist() == [7, 7, 7]

    # batch element 0: frames 2,4,7 gathered in that order, offset 0 -> values 2,4,7
    assert compact[0, :3, 0].tolist() == [2.0, 4.0, 7.0]
    assert torch.all(compact[0, 3:] == 0.0)

    # batch element 1: frames 0,7 gathered in that order, offset 100 -> values 100,107
    assert compact[1, :2, 0].tolist() == [100.0, 107.0]
    assert torch.all(compact[1, 2:] == 0.0)

    # batch element 2: fully observed, all 8 frames present in original order, offset 200
    assert compact[2, :, 0].tolist() == [200.0 + i for i in range(8)]


def test_full_observation_ratio_equivalence_corrected_vs_legacy() -> None:
    """ratio=1.0 (all-ones mask): corrected forward() must be numerically
    equivalent to the pre-repair `_forward_legacy_prefix_packing`, given
    identical weights and identical input tensors."""
    torch.manual_seed(0)
    m = HealthDyPhyGraph(ModelConfig(hidden=32))
    m.eval()
    x = torch.randn(5, 20, 9)
    x[:, :, -1] = 1.0  # all-ones mask column

    with torch.no_grad():
        logits_new, probs_new = m(x)
        logits_old, probs_old = m._forward_legacy_prefix_packing(x)

    assert torch.allclose(logits_new, logits_old, atol=1e-5, rtol=1e-5)
    assert torch.allclose(probs_new, probs_old, atol=1e-5, rtol=1e-5)


def test_full_observation_ratio_equivalence_across_batch_sizes_and_lengths() -> None:
    torch.manual_seed(7)
    m = HealthDyPhyGraph(ModelConfig(hidden=16))
    m.eval()
    for b, t in [(1, 1), (3, 5), (8, 60)]:
        x = torch.randn(b, t, 9)
        x[:, :, -1] = 1.0
        with torch.no_grad():
            logits_new, _ = m(x)
            logits_old, _ = m._forward_legacy_prefix_packing(x)
        assert torch.allclose(logits_new, logits_old, atol=1e-5, rtol=1e-5), (b, t)


def test_scattered_mask_forward_runs_and_shapes_correct() -> None:
    """Sanity: corrected forward() must run end-to-end on scattered masks
    (not just the pure gather helper) and produce correctly shaped output."""
    torch.manual_seed(1)
    m = HealthDyPhyGraph(ModelConfig(hidden=16))
    x = torch.randn(4, 8, 9)
    mask_rows = torch.tensor(SCATTERED_MASKS, dtype=torch.float32)
    x[:, :, -1] = mask_rows
    logits, probs = m(x)
    assert logits.shape == (4,)
    assert probs.shape == (4,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
