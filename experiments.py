"""Run all experiments and save figures + a results table.

    python experiments.py            # full run (a few minutes)
    python experiments.py --quick    # fewer seeds, for a fast check
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data, transform
from skimage.restoration import denoise_tv_chambolle

import pr

OUT = "results"
TV_WEIGHT = 0.2          # one fixed denoiser setting, not tuned per noise level
SUCCESS_DB = 40.0        # noiseless runs above this PSNR count as "success"


def tv(z):
    return denoise_tv_chambolle(z, weight=TV_WEIGHT)


def load_image(n=64):
    img = data.camera().astype(float) / 255.0
    return transform.resize(img, (n, n), anti_aliasing=True)


def random_init(prob, seed):
    rng = np.random.default_rng(1000 + seed)
    return rng.random(prob["obj"].shape) * prob["support"]


def score(x, prob):
    return pr.align_to_reference(x, prob["obj"], prob["support"])


def run_methods(prob, seed, methods):
    x0 = random_init(prob, seed)
    m, s = prob["mag"], prob["support"]
    res = {}
    if "ER only" in methods:
        x, _ = pr.error_reduction(m, s, x0, n_iter=1100, record=False)
        res["ER only"] = x
    if "HIO+ER" in methods or "HIO+ER + TV (post)" in methods:
        x, _ = pr.hio_then_er(m, s, x0)
        res["HIO+ER"] = x
        res["HIO+ER + TV (post)"] = pr.project_object(tv(x), s)
    if "PnP-HIO (TV)" in methods:
        x, _ = pr.pnp_hio(m, s, x0, tv, n_iter=1000, denoise_every=10)
        res["PnP-HIO (TV)"] = x
    return res


def main(quick):
    os.makedirs(OUT, exist_ok=True)
    img = load_image(64)
    n_seeds = 4 if quick else 8
    table = {}

    # ---- Experiment 1: noiseless, ER vs HIO (stagnation) -------------------
    prob = pr.make_problem(img, noise_level=0.0)
    x0 = random_init(prob, 0)
    _, e_er = pr.error_reduction(prob["mag"], prob["support"], x0, n_iter=1100)
    _, e_hio = pr.hio(prob["mag"], prob["support"], x0, n_iter=1100)
    plt.figure(figsize=(6, 4))
    plt.semilogy(e_er, label="ER")
    plt.semilogy(e_hio, label="HIO (beta=0.9)")
    plt.xlabel("iteration")
    plt.ylabel("relative Fourier magnitude error")
    plt.title("Noiseless, same random start")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_er_vs_hio.png", dpi=150)
    plt.close()

    # ---- Experiment 2: success rate over random starts (noiseless) ---------
    rows = {"ER only": [], "HIO+ER": []}
    for seed in range(n_seeds):
        res = run_methods(prob, seed, ["ER only", "HIO+ER"])
        for k in rows:
            rows[k].append(score(res[k], prob)[1])
    table["noiseless"] = {
        k: {"psnr_per_seed": [round(float(v), 2) for v in vals],
            "success_rate": float(np.mean(np.array(vals) > SUCCESS_DB))}
        for k, vals in rows.items()
    }
    print("noiseless:", json.dumps(table["noiseless"]), flush=True)

    # ---- Experiment 3: noisy magnitudes ------------------------------------
    noise_levels = [1.0, 3.0, 6.0]
    methods = ["HIO+ER", "HIO+ER + TV (post)", "PnP-HIO (TV)"]
    stats = {m: {"mean": [], "std": []} for m in methods}
    example = {}
    for nl in noise_levels:
        prob_n = pr.make_problem(img, noise_level=nl, seed=1)
        per = {m: [] for m in methods}
        for seed in range(n_seeds):
            res = run_methods(prob_n, seed, methods)
            for m in methods:
                xa, p = score(res[m], prob_n)
                per[m].append(p)
                if seed == 0 and nl == 3.0:
                    example[m] = xa
        for m in methods:
            stats[m]["mean"].append(float(np.mean(per[m])))
            stats[m]["std"].append(float(np.std(per[m])))
        print("noise", nl, {m: round(stats[m]["mean"][-1], 2) for m in methods}, flush=True)
    table["noisy"] = {"noise_levels": noise_levels, "stats": stats, "n_seeds": n_seeds}

    plt.figure(figsize=(6, 4))
    for m in methods:
        plt.errorbar(noise_levels, stats[m]["mean"], yerr=stats[m]["std"], marker="o", capsize=3, label=m)
    plt.xlabel("noise level (std relative to mean magnitude)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Mean +/- std over {n_seeds} random starts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_psnr_vs_noise.png", dpi=150)
    plt.close()

    # ---- Visual example (noise level 3) ------------------------------------
    off, nn = prob["offset"], prob["n"]
    crop = lambda a: a[off:off + nn, off:off + nn]
    prob3 = pr.make_problem(img, noise_level=3.0, seed=1)
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.8))
    ax[0].imshow(crop(prob3["obj"]), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("ground truth")
    for a, m in zip(ax[1:], methods):
        a.imshow(crop(example[m]), cmap="gray", vmin=0, vmax=1)
        a.set_title(f"{m}\n{pr.psnr(prob3['obj'], example[m]):.1f} dB", fontsize=8)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    plt.savefig(f"{OUT}/fig2_examples_noise3.png", dpi=150)
    plt.close()

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(table, f, indent=2)
    print("saved to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
