"""Task-blocking PoC: a measurement harness for second-line (functional-mapping)
safeguards on protein sequence models.

The harness (data pipeline, metrics, attacker sweep, analysis) is safeguard-agnostic.
Task blocking is the first intervention run through it; only ``src.train.taskblock``
is task-blocking-specific.
"""

__version__ = "0.1.0"
