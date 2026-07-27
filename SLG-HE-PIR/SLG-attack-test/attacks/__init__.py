"""SLG-HE-PIR Attack Test Suite.

4 core attacks based on TEST_REPORT.md:
  - L1: M-side Gradient Label Inference (g_{H,t} = a_t - V_y)
  - L2: S-side Activation Label Inference (a_t analysis)
  - M1: U-side Model Inference (Evaluation Phase - S's predictions)
  - M2: S-side Hidden State Inversion (Z_t analysis)
"""

from attacks.L1_gradient_inference import L1GradientInference
from attacks.L2_activation_inference import L2ActivationInference
from attacks.M1_logits_distillation import M1ModelInference
from attacks.M2_hidden_inversion import M2HiddenStateInversion

__all__ = [
    "L1GradientInference",
    "L2ActivationInference",
    "M1ModelInference",
    "M2HiddenStateInversion",
]
