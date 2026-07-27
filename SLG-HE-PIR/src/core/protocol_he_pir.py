"""
SLG-BPL-Lite Protocol Implementation.

Provides real cryptographic implementations for the SLG three-party protocol:
  - M-side: RSA-KEM unwrap + AES-256-GCM decrypt + vector unpack -> gradient
  - U-side: PrivSelect (placeholder)
  - S-side: RSA-KEM wrap + AES-256-GCM encrypt (in worker_S)
"""

from __future__ import annotations
import numpy as np
from typing import Optional
import sys
from pathlib import Path

# Add project root to path for crypto_utils import
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from slg_core import crypto_utils as cu


class ModelM:
    """M-side protocol handler. Performs real RSA-KEM + AES-256-GCM decryption."""

    def __init__(self, secret_key=None):
        """Initialize with RSA private key from cu.KEM.keygen()."""
        self._sk = secret_key

    def decrypt_grad(
        self,
        c_K: bytes,
        ciphertext: bytes,
        hidden_dim: int,
        device: str = "cpu",
    ) -> np.ndarray:
        """Real decryption pipeline:
        1. RSA-KEM unwrap: recover AES-256-GCM symmetric key
        2. AES-256-GCM decrypt: recover packed gradient vector
        3. unpack_vector: recover float64 numpy array

        Returns: (hidden_dim,) float64 gradient vector
        """
        # Step 1: unwrap the symmetric key
        k_sym = cu.KEM.unwrap(self._sk, c_K)

        # Step 2: decrypt the ciphertext
        decrypted = cu.aes_decrypt(k_sym, ciphertext)

        # Step 3: unpack to numpy array
        grad_np = cu.unpack_vector(decrypted)

        return grad_np


class UserU:
    """U-side protocol handler. Stub."""
    pass


class ServerS:
    """S-side protocol handler. Stub (encryption done in worker_S directly)."""
    pass
