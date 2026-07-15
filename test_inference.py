"""Small CPU smoke tests for the deployed decision rule."""

import glob
import os
import unittest

import numpy as np
from PIL import Image

from inference import MIN_REFS, RECOMMENDED_REFS, decide_embeddings, verify_images


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = os.path.join("sign_data_combined", "icdar_049")
        genuine = sorted(glob.glob(os.path.join(folder, "*_g_*.png")))
        cls.refs = [Image.open(path) for path in genuine[:RECOMMENDED_REFS]]
        cls.query = Image.open(genuine[RECOMMENDED_REFS])

    @classmethod
    def tearDownClass(cls):
        for image in cls.refs + [cls.query]:
            image.close()

    def test_real_sample_returns_a_finite_decision(self):
        for count in (MIN_REFS, RECOMMENDED_REFS):
            result = verify_images(self.refs[:count], self.query)
            self.assertIn(result["verdict"], {"GENUINE", "INCONCLUSIVE", "FORGED"})
            self.assertGreater(result["threshold"], 0)
            self.assertTrue(np.isfinite(result["distance"]))

    def test_duplicate_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "3 to 5 distinct"):
            verify_images([self.refs[0]] * MIN_REFS, self.query)

    def test_uniform_embeddings_are_rejected(self):
        embeddings = np.zeros((MIN_REFS, 128), dtype="float32")
        with self.assertRaisesRegex(ValueError, "too uniform"):
            decide_embeddings(embeddings, embeddings[0])


if __name__ == "__main__":
    unittest.main()
