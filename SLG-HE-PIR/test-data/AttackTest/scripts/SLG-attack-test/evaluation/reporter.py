"""Attack verdict reporter — serialises results to JSON and HTML."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import AttackVerdict, Verdict

logger = logging.getLogger(__name__)


class AttackReporter:
    """Aggregates and reports AttackVerdict objects."""

    def __init__(self, output_dir: str = "SLG-attack-test/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verdicts: List[AttackVerdict] = []
        self.metadata: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "suite_version": "1.0.0",
        }

    def add(self, verdict: AttackVerdict) -> None:
        self.verdicts.append(verdict)

    def add_batch(self, verdicts: List[AttackVerdict]) -> None:
        self.verdicts.extend(verdicts)

    def set_metadata(self, **kwargs) -> None:
        self.metadata.update(kwargs)

    # ------------------------------------------------------------------------- #
    #  JSON export
    # ------------------------------------------------------------------------- #

    def to_json(self, path: Optional[str] = None) -> str:
        if path is None:
            path = str(self.output_dir / "attack_results.json")

        report = {
            "metadata": self.metadata,
            "summary": self._summary_dict(),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("JSON report written to %s", path)
        return path

    # ------------------------------------------------------------------------- #
    #  HTML report
    # ------------------------------------------------------------------------- #

    def to_html(self, path: Optional[str] = None) -> str:
        if path is None:
            path = str(self.output_dir / "attack_report.html")

        summary = self._summary_dict()
        rows = ""
        for v in self.verdicts:
            icon, color = {
                "PRIVACY_PRESERVED": ("&#x2714;", "#2ecc71"),
                "LEAK_DETECTED":      ("&#x26A0;", "#e74c3c"),
                "INCONCLUSIVE":        ("&#x2753;", "#95a5a6"),
            }.get(v.verdict, ("&#x2753;", "#95a5a6"))

            p_str = f'<span class="pval">p={v.p_value:.4f}</span>' if v.p_value is not None else ""
            ci_str = ""
            if v.confidence_interval is not None:
                ci_str = f'<span class="ci">CI=[{v.confidence_interval[0]:.3f}, {v.confidence_interval[1]:.3f}]</span>'

            rows += f"""
            <tr>
              <td class="attack-id">{v.attack_id}</td>
              <td class="sub-attack">{v.sub_attack}</td>
              <td class="metric">{v.metric}</td>
              <td class="value">{v.value:.4f}</td>
              <td class="chance">{v.chance_level:.4f}</td>
              <td class="std_err">{v.std_err:.4f}</td>
              <td>{p_str}</td>
              <td>{ci_str}</td>
              <td class="n-samples">{v.n_samples}</td>
              <td class="verdict" style="color:{color}">{icon} {v.verdict}</td>
              <td class="notes">{v.notes}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SLG-HE-PIR Attack Test Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; margin: 2rem; }}
    h1 {{ color: #2c3e50; }}
    .meta {{ background: white; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    .summary-cards {{ display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }}
    .card {{ background: white; border-radius: 8px; padding: 1rem; min-width: 140px; box-shadow: 0 1px 4px rgba(0,0,0,.1); text-align: center; }}
    .card .num {{ font-size: 2rem; font-weight: bold; }}
    .card .label {{ font-size: 0.85rem; color: #7f8c8d; }}
    .card.ok    .num {{ color: #2ecc71; }}
    .card.warn  .num {{ color: #e74c3c; }}
    .card.maybe .num {{ color: #95a5a6; }}
    table {{ border-collapse: collapse; background: white; width: 100%; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    th {{ background: #34495e; color: white; padding: 0.6rem 0.5rem; text-align: left; font-size: 0.85rem; }}
    td {{ padding: 0.5rem; border-bottom: 1px solid #ecf0f1; font-size: 0.9rem; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover {{ background: #f5f7fa; }}
    .attack-id {{ font-weight: bold; color: #2980b9; }}
    .sub-attack {{ color: #7f8c8d; font-size: 0.85rem; }}
    .metric {{ color: #8e44ad; }}
    .value {{ font-weight: bold; }}
    .chance, .std_err {{ color: #7f8c8d; }}
    .pval, .ci {{ font-family: monospace; font-size: 0.8rem; background: #ecf0f1; padding: 0.1rem 0.3rem; border-radius: 3px; }}
    .verdict {{ font-weight: bold; }}
    .notes {{ color: #7f8c8d; font-size: 0.85rem; max-width: 200px; }}
  </style>
</head>
<body>
  <h1>SLG-HE-PIR Attack Test Report</h1>

  <div class="summary-cards">
    <div class="card ok">
      <div class="num">{summary['n_preserved']}</div>
      <div class="label">Privacy Preserved</div>
    </div>
    <div class="card warn">
      <div class="num">{summary['n_leaked']}</div>
      <div class="label">Leak Detected</div>
    </div>
    <div class="card maybe">
      <div class="num">{summary['n_inconclusive']}</div>
      <div class="label">Inconclusive</div>
    </div>
    <div class="card">
      <div class="num">{summary['n_total']}</div>
      <div class="label">Total Tests</div>
    </div>
  </div>

  <div class="meta">
    <strong>Timestamp:</strong> {self.metadata.get('timestamp','')}<br/>
    <strong>Model:</strong> {self.metadata.get('model','')}<br/>
    <strong>Dataset:</strong> {self.metadata.get('dataset','')}<br/>
    <strong>N Steps:</strong> {self.metadata.get('n_steps','')}<br/>
    <strong>Seed:</strong> {self.metadata.get('seed','')}<br/>
  </div>

  <table>
    <thead>
      <tr>
        <th>Attack ID</th>
        <th>Sub-attack</th>
        <th>Metric</th>
        <th>Value</th>
        <th>Chance</th>
        <th>Std Err</th>
        <th>p-value</th>
        <th>CI</th>
        <th>N</th>
        <th>Verdict</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {rows or '<tr><td colspan="11" style="text-align:center;color:#7f8c8d">No attack results yet.</td></tr>'}
    </tbody>
  </table>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML report written to %s", path)
        return path

    # ------------------------------------------------------------------------- #
    #  Summary helpers
    # ------------------------------------------------------------------------- #

    def _summary_dict(self) -> Dict[str, Any]:
        counts = {"PRIVACY_PRESERVED": 0, "LEAK_DETECTED": 0, "INCONCLUSIVE": 0}
        for v in self.verdicts:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        return {
            "n_total": len(self.verdicts),
            "n_preserved": counts["PRIVACY_PRESERVED"],
            "n_leaked": counts["LEAK_DETECTED"],
            "n_inconclusive": counts["INCONCLUSIVE"],
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("  SLG-HE-PIR Attack Test Summary")
        print("=" * 70)
        for v in self.verdicts:
            print(f"  {v.summary()}")
        summary = self._summary_dict()
        print("-" * 70)
        print(f"  Total: {summary['n_total']}  |  "
              f"Preserved: {summary['n_preserved']}  |  "
              f"Leaked: {summary['n_leaked']}  |  "
              f"Inconclusive: {summary['n_inconclusive']}")
        print("=" * 70 + "\n")
