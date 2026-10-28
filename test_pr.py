"""Small sanity tests. Run with:  python -m unittest -v"""
import unittest
import numpy as np
from skimage import data, transform
import pr


def small_image(n=32):
    img = data.camera().astype(float) / 255.0
    return transform.resize(img, (n, n), anti_aliasing=True)


class TestPhaseRetrieval(unittest.TestCase):
    def setUp(self):
        self.prob = pr.make_problem(small_image(32), noise_level=0.0)

    def test_forward_model_is_consistent(self):
        p = self.prob
        self.assertTrue(np.allclose(np.abs(np.fft.fft2(p["obj"])), p["mag"]))
        self.assertEqual(p["obj"].shape[0], 2 * 32)
        self.assertTrue(np.all(p["obj"][~p["support"]] == 0))

    def test_true_object_has_zero_fourier_error(self):
        p = self.prob
        self.assertLess(pr.fourier_error(p["obj"], p["mag"]), 1e-12)

    def test_error_reduction_never_increases_the_error(self):
        p = self.prob
        rng = np.random.default_rng(0)
        x0 = rng.random(p["obj"].shape) * p["support"]
        _, errs = pr.error_reduction(p["mag"], p["support"], x0, n_iter=100)
        self.assertTrue(np.all(np.diff(errs) <= 1e-9))

    def test_alignment_removes_shift_and_twin(self):
        p = self.prob
        twin = np.roll(p["obj"][::-1, ::-1], 1, axis=(0, 1))
        moved = np.roll(twin, (3, -2), axis=(0, 1))
        _, score = pr.align_to_reference(moved, p["obj"], p["support"])
        self.assertGreater(score, 60)

    def test_hio_recovers_small_image_from_some_random_start(self):
        p = self.prob
        best = -np.inf
        for seed in range(5):
            rng = np.random.default_rng(seed)
            x0 = rng.random(p["obj"].shape) * p["support"]
            x, _ = pr.hio_then_er(p["mag"], p["support"], x0, n_hio=800, n_er=50)
            best = max(best, pr.align_to_reference(x, p["obj"], p["support"])[1])
        self.assertGreater(best, 40)


if __name__ == "__main__":
    unittest.main()
