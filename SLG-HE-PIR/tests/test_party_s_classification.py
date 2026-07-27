"""Smoke tests for the 7-class classification projection in PartyS.

Pure stdlib (no pytest required) — runnable via:

    python -m unittest tests.test_party_s_classification -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


_THIS = Path(__file__).resolve().parent
for p in (str(_THIS), str(_THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers — minimal PartyS stand-in (no Llama checkpoint needed)
# ---------------------------------------------------------------------------

def _make_party_s_stub():
    from src.parties.party_s import PartyS

    party = PartyS.__new__(PartyS)
    party.device = torch.device("cpu")
    party.config = {}
    party.bfv_backend = None
    party.hint_table = None
    party.prg_seed = b""
    party.crypto_s_pool = None
    # Tiny 32-token "vocab" so we can craft specific argmax targets.
    party.V_weight = torch.eye(32, dtype=torch.float32)

    def _stub_opt(device=None):
        return torch.tensor(list(range(7)), dtype=torch.long)

    party._get_option_token_ids = _stub_opt
    return party


# ---------------------------------------------------------------------------
# _get_last_nonpad_index
# ---------------------------------------------------------------------------

class TestLastNonpadIndex(unittest.TestCase):
    def test_basic(self):
        from src.parties.party_s import PartyS
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.long,
        )
        last = PartyS._get_last_nonpad_index(mask)
        self.assertEqual(last.tolist(), [3, 1, 5])

    def test_all_zero_clamped(self):
        from src.parties.party_s import PartyS
        mask = torch.zeros(2, 5, dtype=torch.long)
        last = PartyS._get_last_nonpad_index(mask)
        self.assertTrue((last >= 0).all())
        self.assertEqual(last.tolist(), [0, 0])


# ---------------------------------------------------------------------------
# _classify_from_logits
# ---------------------------------------------------------------------------

class TestClassifyFromLogits(unittest.TestCase):
    def test_predictions_in_alphabet(self):
        party = _make_party_s_stub()
        B, S, V = 4, 6, 32
        torch.manual_seed(0)
        H_M = torch.randn(B, S, party.V_weight.shape[1])
        logits = party.compute_logits_gpu(H_M)

        mask = torch.ones(B, S, dtype=torch.long)
        mask[1, 3:] = 0
        mask[2, 5:] = 0
        mask[3, 2:] = 0

        out = party.generate_predictions(
            logits, attention_mask=mask, task_type="classification",
        )
        allowed = {"a)", "b)", "c)", "d)", "e)", "f)", "g)"}
        for p in out["predictions"]:
            self.assertIn(p, allowed)
        self.assertEqual(len(out["predictions"]), B)

        self.assertEqual(len(out["logits"]), B)
        for row in out["logits"]:
            self.assertEqual(len(row), 7)
            for x in row:
                self.assertIsInstance(x, float)

    def test_argmax_correct(self):
        party = _make_party_s_stub()
        B, S, V = 3, 4, 32
        torch.manual_seed(0)
        H_M = torch.randn(B, S, party.V_weight.shape[1])
        logits = party.compute_logits_gpu(H_M)
        mask = torch.ones(B, S, dtype=torch.long)

        out = party.generate_predictions(
            logits, attention_mask=mask, task_type="classification",
        )
        expected = []
        opt_ids = party._get_option_token_ids()
        for b in range(B):
            last_pos_logits = logits[b, -1, :]
            seven = last_pos_logits[opt_ids]
            expected.append(f"{chr(ord('a') + int(seven.argmax().item()))})")
        self.assertEqual(out["predictions"], expected)

    def test_respects_attention_mask(self):
        party = _make_party_s_stub()
        B, S, V = 1, 5, 32
        H_M = torch.zeros(B, S, party.V_weight.shape[1])
        logits = party.compute_logits_gpu(H_M)
        # Strong "c)" preference (token id 2) at the *last non-pad* position.
        # mask sum = 3 → last non-pad index = 2.
        logits[0, 2, 2] = 100.0
        # Negative spikes at unmasked positions must NOT influence argmax.
        logits[0, 3, 4] = -100.0  # masked out
        logits[0, 4, 5] = -100.0  # masked out
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

        out = party.generate_predictions(
            logits, attention_mask=mask, task_type="classification",
        )
        # Token id 2 → letter "c"
        self.assertEqual(out["predictions"], ["c)"])


# ---------------------------------------------------------------------------
# compute_classification_metrics: pred_logits changes ROC AUC computation
# ---------------------------------------------------------------------------

class TestMetricsPredLogits(unittest.TestCase):
    def test_pred_logits_used(self):
        from src.training.biotriplex_metrics import compute_classification_metrics

        # Build 14 samples: each of the 7 classes gets 2 well-separated
        # samples so that ``roc_auc_score(..., multi_class='ovr',
        # average='macro')`` returns a finite number (sklearn raises
        # ``UndefinedMetricWarning`` on classes with zero true samples).
        letters = ["a)", "b)", "c)", "d)", "e)", "f)", "g)"]
        preds, labels, pred_logits = [], [], []
        for cls_idx, letter in enumerate(letters):
            for _ in range(2):
                preds.append(letter)
                labels.append(letter)
                row = [-5.0] * 7
                row[cls_idx] = 5.0
                pred_logits.append(row)
        out = compute_classification_metrics(preds, labels, pred_logits=pred_logits)
        self.assertTrue(out["has_logits"])
        self.assertEqual(out["n_parse_failures"], 0)
        self.assertGreater(out["metrics"]["micro_f1"], 0.9)
        roc = out["metrics"]["macro_roc_auc_ovr"]
        self.assertIsNotNone(roc)
        self.assertFalse(roc != roc)  # not NaN
        self.assertGreaterEqual(roc, 0.99)
        self.assertLessEqual(roc, 1.0001)

    def test_length_mismatch_fallback(self):
        from src.training.biotriplex_metrics import compute_classification_metrics

        preds = ["a)", "b)"]
        labels = ["a)", "b)"]
        out = compute_classification_metrics(
            preds,
            labels,
            pred_logits=[[5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]],
        )
        self.assertFalse(out["has_logits"])
        self.assertEqual(out["n_samples"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
