"""Does annealing the denoising strength explain why in-loop denoising helped less?

Compares, on noisy Fourier magnitudes (same setup as experiments.py):
  - HIO+ER                      (no denoising)
  - HIO+ER, then TV (post)      (TV weight 0.2 applied once at the end)
  - PnP-HIO, fixed TV weight    (0.2 every 10 iterations, then 0.2 once at the end)
  - PnP-HIO, annealed TV weight (0.2 -> 0.01 geometrically over the iterations, no final denoising)
  - PnP-HIO, annealed + final   (as above, plus TV 0.2 once at the end)

    python annealing_experiment.py            # full run
    python annealing_experiment.py --quick
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data, transform
from skimage.restoration import denoise_tv_chambolle

import pr

OUT = "results"
W_START, W_END = 0.2, 0.01
N_ITER = 1000
EVERY = 10
METHODS = ["HIO+ER", "HIO+ER + TV (post)", "PnP-HIO fixed", "PnP-HIO annealed", "PnP-HIO annealed + final TV"]


def tv(z, w):
    return denoise_tv_chambolle(z, weight=w)


def pnp_hio_scheduled(mag, support, x0, w_start, w_end, final_w, n_iter=N_ITER, beta=0.9, every=EVERY):
    """HIO with a TV denoising step whose weight goes from w_start to w_end (geometric)."""
    x = x0.copy()
    for k in range(n_iter):
        xp = pr.project_fourier(x, mag)
        ok = support & (xp >= 0)
        x = np.where(ok, xp, x - beta * xp)
        if k % every == 0:
            w = w_start * (w_end / w_start) ** (k / (n_iter - 1))
            inside = np.where(support, x, 0.0)
            x = np.where(support, tv(inside, w), x)
    x = pr.project_object(x, support)
    if final_w is not None:
        x = pr.project_object(tv(x, final_w), support)
    return x


def load_image(n=64):
    img = data.camera().astype(float) / 255.0
    return transform.resize(img, (n, n), anti_aliasing=True)


def one_job(args):
    noise, seed = args
    img = load_image(64)
    prob = pr.make_problem(img, noise_level=noise, seed=1)
    m, s = prob["mag"], prob["support"]
    rng = np.random.default_rng(1000 + seed)
    x0 = rng.random(prob["obj"].shape) * prob["support"]
    res = {}
    x, _ = pr.hio_then_er(m, s, x0)
    res["HIO+ER"] = x
    res["HIO+ER + TV (post)"] = pr.project_object(tv(x, W_START), s)
    res["PnP-HIO fixed"] = pnp_hio_scheduled(m, s, x0, W_START, W_START, W_START)
    res["PnP-HIO annealed"] = pnp_hio_scheduled(m, s, x0, W_START, W_END, None)
    res["PnP-HIO annealed + final TV"] = pnp_hio_scheduled(m, s, x0, W_START, W_END, W_START)
    return noise, seed, {k: float(pr.align_to_reference(v, prob["obj"], s)[1]) for k, v in res.items()}


def main(quick):
    os.makedirs(OUT, exist_ok=True)
    n_seeds = 4 if quick else 8
    noises = [1.0, 3.0, 6.0]
    jobs = [(nl, sd) for nl in noises for sd in range(n_seeds)]
    with Pool() as pool:
        results = pool.map(one_job, jobs)
    table = {m: {"mean": [], "std": []} for m in METHODS}
    for nl in noises:
        for m in METHODS:
            vals = [r[2][m] for r in results if r[0] == nl]
            table[m]["mean"].append(float(np.mean(vals)))
            table[m]["std"].append(float(np.std(vals)))
        print("noise", nl, {m: round(table[m]["mean"][-1], 2) for m in METHODS}, flush=True)
    with open(f"{OUT}/annealing.json", "w") as f:
        json.dump({"noise_levels": noises, "n_seeds": n_seeds, "w_start": W_START, "w_end": W_END, "stats": table}, f, indent=2)
    plt.figure(figsize=(7, 4.2))
    for m in METHODS:
        plt.errorbar(noises, table[m]["mean"], yerr=table[m]["std"], marker="o", capsize=3, label=m)
    plt.xlabel("noise level (std relative to mean magnitude)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Annealed vs fixed denoising strength ({n_seeds} random starts)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_annealing.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
