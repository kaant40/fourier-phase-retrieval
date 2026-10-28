"""Classical Fourier phase retrieval, implemented from scratch with NumPy.

Problem: recover a real, non-negative image x (known support) from the
magnitude of its oversampled 2D Fourier transform, y = |F x|.

Contents
--------
- make_problem        : build object, support mask and (noisy) Fourier magnitudes
- error_reduction     : Gerchberg-Saxton / Error Reduction (ER)
- hio                 : Hybrid Input-Output (HIO, Fienup 1982)
- hio_then_er         : the usual practical recipe (many HIO steps, a few ER steps)
- pnp_hio             : HIO with a plug-and-play denoising step (simple variant)
- align_to_reference  : remove the trivial ambiguities (shift, 180-degree flip) before scoring
- psnr                : peak signal-to-noise ratio
"""
import numpy as np


# ----------------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------------
def make_problem(image, pad_factor=2, noise_level=0.0, seed=0):
    """Create a Fourier phase retrieval problem.

    image        : 2D array in [0, 1], the true object (n x n).
    pad_factor   : the object is zero-padded to (pad_factor*n) x (pad_factor*n).
                   Oversampling by >= 2 in each dimension is what makes the
                   problem (almost always) uniquely solvable.
    noise_level  : std of additive Gaussian noise on the magnitudes, relative
                   to the mean magnitude. 0 means noiseless.
    Returns a dict with the padded object, the support mask and the magnitudes.
    """
    rng = np.random.default_rng(seed)
    n = image.shape[0]
    big = pad_factor * n
    obj = np.zeros((big, big))
    off = (big - n) // 2
    obj[off:off + n, off:off + n] = image
    support = np.zeros((big, big), dtype=bool)
    support[off:off + n, off:off + n] = True

    mag = np.abs(np.fft.fft2(obj))
    if noise_level > 0:
        mag = mag + noise_level * mag.mean() * rng.standard_normal(mag.shape)
        mag = np.maximum(mag, 0.0)  # magnitudes cannot be negative
    return {"obj": obj, "support": support, "mag": mag, "offset": off, "n": n}


def fourier_error(x, mag):
    """Relative distance between |F x| and the measured magnitudes."""
    return np.linalg.norm(np.abs(np.fft.fft2(x)) - mag) / np.linalg.norm(mag)


# ----------------------------------------------------------------------------
# Projections
# ----------------------------------------------------------------------------
def project_fourier(x, mag):
    """P_M: keep the phase of F x, replace its magnitude by the measurement."""
    X = np.fft.fft2(x)
    X = mag * np.exp(1j * np.angle(X))
    return np.real(np.fft.ifft2(X))  # object is real, drop the tiny imaginary part


def project_object(x, support):
    """P_S: zero outside the support and clip negatives (object is real, >= 0)."""
    out = np.where(support, x, 0.0)
    return np.maximum(out, 0.0)


# ----------------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------------
def error_reduction(mag, support, x0, n_iter=200, record=True):
    """ER: x <- P_S(P_M(x)). Simple, but it stagnates in local minima."""
    x = x0.copy()
    errs = []
    for _ in range(n_iter):
        x = project_object(project_fourier(x, mag), support)
        if record:
            errs.append(fourier_error(x, mag))
    return x, errs


def hio(mag, support, x0, n_iter=1000, beta=0.9, record=True):
    """HIO: pixels that violate the object constraints are pushed *negatively*
    (x - beta * x'), which lets the iterate escape the stagnation of ER.
    """
    x = x0.copy()
    errs = []
    for _ in range(n_iter):
        xp = project_fourier(x, mag)
        ok = support & (xp >= 0)          # pixels that satisfy the constraints
        x = np.where(ok, xp, x - beta * xp)
        if record:
            # HIO's iterate x is not itself a valid image; track the error of
            # its projection onto the object constraints.
            errs.append(fourier_error(project_object(x, support), mag))
    return x, errs


def hio_then_er(mag, support, x0, n_hio=1000, n_er=100, beta=0.9):
    """Standard recipe: run HIO, then finish with a few ER steps."""
    x, e1 = hio(mag, support, x0, n_hio, beta)
    x = project_object(x, support)
    x, e2 = error_reduction(mag, support, x, n_er)
    return x, e1 + e2


def pnp_hio(mag, support, x0, denoiser, n_iter=1000, beta=0.9,
            denoise_every=10, final_denoise=True):
    """HIO with a plug-and-play denoising step (a deliberately simple variant).

    Every `denoise_every` iterations, the part of the HIO iterate inside the
    support is passed through `denoiser` (any function image -> image of the
    same size, e.g. a TV or Gaussian filter). The final estimate is denoised
    once more if `final_denoise` is True.

    This is NOT the method of the Deep Plug-and-Play HIO paper (which uses a
    learned denoiser and a more careful algorithm). It is only a minimal way
    to see what a denoising prior can do inside an HIO loop.
    """
    x = x0.copy()
    errs = []
    for k in range(n_iter):
        xp = project_fourier(x, mag)
        ok = support & (xp >= 0)
        x = np.where(ok, xp, x - beta * xp)
        if k % denoise_every == 0:
            inside = np.where(support, x, 0.0)
            x = np.where(support, denoiser(inside), x)
        errs.append(fourier_error(project_object(x, support), mag))
    x = project_object(x, support)
    if final_denoise:
        x = project_object(denoiser(x), support)
    return x, errs


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def psnr(ref, est, peak=1.0):
    mse = np.mean((ref - est) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def align_to_reference(est, ref, support):
    """Fourier magnitudes cannot tell an image from its translated copy or its
    180-degree rotated ("twin") copy. Before scoring, try both orientations and
    the best circular shift (via cross-correlation) and keep the better one.
    Returns the aligned estimate and its PSNR against `ref`.
    """
    best, best_p = est, -np.inf
    for cand in (est, np.roll(est[::-1, ::-1], 1, axis=(0, 1))):
        corr = np.fft.ifft2(np.fft.fft2(ref) * np.conj(np.fft.fft2(cand))).real
        dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
        shifted = np.roll(cand, (dy, dx), axis=(0, 1))
        shifted = np.where(support, shifted, 0.0)
        p = psnr(ref, shifted)
        if p > best_p:
            best, best_p = shifted, p
    return best, best_p
