"""MNIST CNN digit scorer for IP4R — feature-similarity mode.

The MNIST CNN (kenil-patel-183/mnist-cnn-digit-classifier, 99.25% MNIST acc)
is used as a *feature extractor*, not a digit classifier.

Because 7-segment LCD digits look unlike handwritten MNIST digits, direct
classification (is this an '8'?) fails. Instead we compute cosine similarity
between CNN feature vectors of the golden patch and the sample patch:

    defect_score = 1 - cos_sim(feat(golden), feat(sample))

Identical LCD digits → high cos_sim → defect_score ≈ 0  → PASS
Missing/damaged seg → lower cos_sim → defect_score ↑    → FAIL

Calibrated on 3351 good-bank images across 6 cameras:
  good bank: cos_sim 0.82 – 0.99  → defect_score 0.01 – 0.18
  3 segs missing: avg cos_sim 0.74 → defect_score ≈ 0.26

Safe threshold: defect_threshold = 0.20  (cos_sim < 0.80 → FAIL).

Uses model source files bundled with this package; weights downloaded/cached
via hf_hub_download (no trust_remote_code needed, no AutoModel routing).
"""
from __future__ import annotations

import numpy as np

_HF_MODEL_ID = "kenil-patel-183/mnist-cnn-digit-classifier"
_WEIGHTS_FILE = "model.safetensors"


class MnistDigitScorer:
    """Score digit patches by comparing CNN features against the golden patch."""

    def __init__(self, device: str | None = None, invert: bool = True):
        """
        Args:
            device: PyTorch device. Auto-selects MPS → CUDA → CPU if None.
            invert: Flip pixel values (255 - patch) before inference.
                    Set True when active_is_dark=True so LCD segments
                    become bright (matching MNIST's training distribution).
        """
        import torch
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from PIL import Image

        from .mnist_modeling import MnistCNN, MnistCNNConfig
        from .mnist_image_processor import MnistCNNImageProcessor

        self._torch = torch
        self._Image = Image
        self.invert = invert

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        weights_path = hf_hub_download(_HF_MODEL_ID, _WEIGHTS_FILE)
        config = MnistCNNConfig()
        self.model = MnistCNN(config).to(self.device)
        self.model.load_state_dict(load_file(weights_path))
        self.model.eval()

        self.processor = MnistCNNImageProcessor()
        print(f"[MnistDigitScorer] loaded {_HF_MODEL_ID} on {device}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _feat(self, patch: np.ndarray) -> np.ndarray:
        """Extract CNN feature vector (3136-dim) from a grayscale patch."""
        p = (255 - patch.astype(np.uint8)) if self.invert else patch.astype(np.uint8)
        img = self._Image.fromarray(p)
        inp = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        with self._torch.no_grad():
            feat = self.model.flatten(self.model.network(inp))[0]
        return feat.cpu().numpy()

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    # ── Public interface ───────────────────────────────────────────────────────

    def score_vs_golden(self, sample: np.ndarray, golden: np.ndarray) -> float:
        """Return 1 - cos_sim(feat(sample), feat(golden)).

        0.0 = sample features identical to golden → no defect
        1.0 = completely different features → severe defect
        """
        return 1.0 - self._cos_sim(self._feat(sample), self._feat(golden))

    def score(self, patch: np.ndarray) -> float:
        """Fallback: 1 - cos_sim(feat(patch), feat(zero)).  Prefer score_vs_golden."""
        zero = np.zeros_like(patch)
        return self.score_vs_golden(patch, zero)
