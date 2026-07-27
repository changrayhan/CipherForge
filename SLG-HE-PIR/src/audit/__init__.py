"""Offline audit aggregation for the dχ-privacy mechanism.

Mirrors ``src/audit/lia_h15_audit.py`` from the reference project.  Reads
``log_dir/dp_audit.jsonl`` (one JSON record per training step) and emits
both a machine-readable summary (``--output`` JSON) and a human-readable
Markdown report next to it.
"""
