"""Claim 2 (FPS half only): real-time CPU inference speed (Table 2), using the official pretrained
checkpoints + the official Test.py pipeline verbatim, but with synthetic (random) images instead
of real dataset images. This measures raw compute/architecture cost, not accuracy -- the NME half
of Claim 2 needs real test data and is handled separately once dataset access is confirmed.

Paper's Table 2 (FP32, CPU i5-13400 2.5GHz, batch_size=1): 138.4 FPS (COFW), 78.48 FPS (300W),
163.55 FPS (AFLW). We use whatever CPU this environment provides and report our own hardware spec
alongside -- absolute FPS numbers are hardware-dependent and not expected to match exactly; what
matters is (a) real-time order of magnitude (>>1 FPS, ideally >30 FPS for "real-time"), and
(b) the relative ordering across datasets (300W slowest due to most landmarks -> heaviest AuxLoc
cost, matches paper's ordering 300W < COFW < AFLW... actually paper has 300W(78.48) < COFW(138.4)
< AFLW(163.55), consistent with landmark count 68 > 29 > 19 driving AuxLoc's per-landmark cost).

Run with: <repo-root>/.venv/bin/python worldcomp2d-WQIyx69dFg/claim2_fps.py
"""
import os
import sys
import time
import platform
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
WC2D_TEST = os.path.join(REPO_ROOT, "WorldComp2D", "Test")
sys.path.insert(0, WC2D_TEST)
os.chdir(WC2D_TEST)  # Utils.py's load_* functions use "../Pretrained_modules/" relative paths

import Utils as U  # noqa: E402
from Models import framework_making  # noqa: E402

device = torch.device("cpu")
BATCH = 1
N_WARMUP = 5
N_TIMED = 50

FIXATION_POINTS = torch.LongTensor([[64, 64], [64, 128], [64, 192],
                                     [128, 64], [128, 128], [128, 192],
                                     [192, 64], [192, 128], [192, 192]]).to(device)


def run_task(task):
    encoders, localizers, aux_localizers, n_l, landmark_coordinate_prior = framework_making(task)
    for m in (encoders + localizers + aux_localizers):
        m.eval()

    channel = 1 if task == "COFW" else 3
    fixation_points = FIXATION_POINTS.view(1, 9, 2).repeat(BATCH, 1, 1)
    img_size = torch.FloatTensor([256, 256]).to(device)
    fixation_points_norm = (2 * fixation_points / (img_size - 1)) - 1
    class_embedding = (2 * torch.arange(0, n_l).to(device) / (n_l - 1) - 1).view(-1, 1, 1, 1).expand(
        n_l, 1, 27, 27).repeat(BATCH, 1, 1, 1)
    delta_clamp = 2 if task == "COFW" else 1

    def one_forward(images):
        with torch.no_grad():
            o = U.extract_observation(task, images, fixation_points)
            z_1 = encoders[0](o)
            z_x_1 = torch.cat((z_1, fixation_points_norm.view(-1, 2)), dim=1).view(BATCH, 9 * (z_1.size(1) + 2))
            x_hat_1 = U.norm_coord_to_abs(landmark_coordinate_prior + localizers[0](z_x_1)).long()

            o2 = U.extract_observation(task, images, x_hat_1, False)
            o_l = torch.cat((o2, class_embedding), dim=1)
            h = aux_localizers[0](o_l)
            delta_y = torch.argmax(h.view(BATCH, n_l, -1), dim=-1) // 27 - 13
            delta_x = torch.argmax(h.view(BATCH, n_l, -1), dim=-1) % 27 - 13
            delta = torch.cat((delta_y.view(BATCH, n_l, 1), delta_x.view(BATCH, n_l, 1)), dim=-1)
            delta = torch.clamp(delta, -delta_clamp, delta_clamp)
            _ = x_hat_1 + delta
        return

    images = torch.randn(BATCH, channel, 256, 256, device=device)

    for _ in range(N_WARMUP):
        one_forward(images)

    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        one_forward(images)
    elapsed = time.perf_counter() - t0
    fps = N_TIMED / elapsed
    return fps


def main():
    torch.set_num_threads(torch.get_num_threads())  # use whatever default this box has, report it
    print(f"Host CPU: {platform.processor() or platform.machine()}, "
          f"torch threads: {torch.get_num_threads()}, torch: {torch.__version__}")
    print(f"Paper's hardware: Intel i5-13400 2.5GHz, 128GB DRAM, batch_size=1")
    print(f"{'task':6s} {'FPS (ours)':>12s} {'FPS (paper)':>12s}")
    paper_fps = {"COFW": 138.4, "300W": 78.48, "AFLW": 163.55}
    for task in ["COFW", "300W", "AFLW"]:
        fps = run_task(task)
        print(f"{task:6s} {fps:12.2f} {paper_fps[task]:12.2f}")


if __name__ == "__main__":
    main()
