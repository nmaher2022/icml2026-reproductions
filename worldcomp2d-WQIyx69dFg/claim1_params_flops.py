"""Claim 1: params/FLOPs reduction vs PoPos (Table 1, Sec 4.3 of the paper).

Dataset-independent: loads the *official* WorldComp2D model classes and pretrained checkpoints
directly (no reimplementation) and counts parameters + FLOPs, comparing against the paper's own
reported numbers:
  Total params: 2.4M (PdEnc 1.1M, Loc 1.3M, AuxLoc 4.0K)
  Total FLOPs (per Sec 4.3 formula, FLOPs_tot = 9*FLOPs_PdEnc + FLOPs_Loc + |C_tot|*FLOPs_AuxLoc):
    COFW 293.7M, 300W 546.8M, AFLW 256.9M
  Per-module FLOPs (single forward pass, one fixation point / one landmark):
    PdEnc ~15.7M, Loc ~3.0M, AuxLoc ~5.9M
Run with: <repo-root>/.venv/bin/python worldcomp2d-WQIyx69dFg/claim1_params_flops.py
"""
import sys
import torch
from thop import profile

sys.path.insert(0, "WorldComp2D/Test")
from Models import framework_making, Proximity_dependent_encoder, Localizer, Auxiliary_localizer

TASKS = {
    "COFW": dict(n_l=29, enc_channel=2, aux_loc_channel=2),
    "300W": dict(n_l=68, enc_channel=6, aux_loc_channel=4),
    "AFLW": dict(n_l=19, enc_channel=6, aux_loc_channel=4),
}


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    print(f"{'task':6s} {'PdEnc params':>14s} {'Loc params':>12s} {'AuxLoc params':>14s} {'total':>10s}")
    totals = {}
    for task, cfg in TASKS.items():
        enc = Proximity_dependent_encoder(cfg["enc_channel"])
        loc = Localizer(cfg["n_l"])
        aux = Auxiliary_localizer(cfg["aux_loc_channel"])

        p_enc, p_loc, p_aux = count_params(enc), count_params(loc), count_params(aux)
        total = p_enc + p_loc + p_aux
        totals[task] = dict(p_enc=p_enc, p_loc=p_loc, p_aux=p_aux, total=total)
        print(f"{task:6s} {p_enc:14,d} {p_loc:12,d} {p_aux:14,d} {total:10,d}")

        # FLOPs (thop reports MACs; paper's convention checked against both MACs and 2*MACs below)
        enc_in = torch.randn(1, cfg["enc_channel"], 27, 27)
        loc_in = torch.randn(1, 9 * (256 + 2))
        aux_in = torch.randn(1, cfg["aux_loc_channel"], 27, 27)

        macs_enc, _ = profile(enc, inputs=(enc_in,), verbose=False)
        macs_loc, _ = profile(loc, inputs=(loc_in,), verbose=False)
        macs_aux, _ = profile(aux, inputs=(aux_in,), verbose=False)

        flops_tot_macs = 9 * macs_enc + macs_loc + cfg["n_l"] * macs_aux
        flops_tot_2x = 2 * flops_tot_macs

        print(f"       PdEnc MACs/fwd={macs_enc:,.0f}  Loc MACs/fwd={macs_loc:,.0f}  "
              f"AuxLoc MACs/fwd={macs_aux:,.0f}")
        print(f"       FLOPs_tot (paper formula, 9xPdEnc + Loc + {cfg['n_l']}xAuxLoc):")
        print(f"         as MACs:       {flops_tot_macs:,.0f}  ({flops_tot_macs/1e6:.1f}M)")
        print(f"         as 2xMACs:     {flops_tot_2x:,.0f}  ({flops_tot_2x/1e6:.1f}M)")

    grand_total = sum(t["total"] for t in totals.values()) // 3  # same arch reused per task family differs only in channel counts; report per-task
    print()
    print("Paper Table 1 target: total params 2.4M (PdEnc 1.1M / Loc 1.3M / AuxLoc 4.0K); "
          "FLOPs 293.7M(COFW) / 546.8M(300W) / 256.9M(AFLW)")
    print("Paper per-module single-fixation FLOPs targets: PdEnc ~15.7M, Loc ~3.0M, AuxLoc ~5.9M")

    # PoPos comparison (from Table 1, values as stated in the paper text/table)
    popos_params_M = 9.7
    popos_flops_worst_G = 1.2  # worst case for FLOPs ratio per paper text ("in the worst case")
    my_total_params_M = totals["300W"]["total"] / 1e6  # 300W has largest AuxLoc/enc channel count but Loc/AuxLoc scale differs per task -- see note below
    print(f"\nNote: total param count differs slightly by task (enc_channel/aux_loc_channel/n_l vary);"
          f" paper reports a single 2.4M figure -- comparing each task's total against it below.")
    for task, t in totals.items():
        ratio = popos_params_M * 1e6 / t["total"]
        print(f"  {task}: total={t['total']:,} ({t['total']/1e6:.2f}M) -> "
              f"param reduction vs PoPos 9.7M = {ratio:.2f}x")


if __name__ == "__main__":
    main()
