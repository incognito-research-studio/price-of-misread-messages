"""Compile, Don't Decide.

Operational compilation for LLM-assisted operations: an interpreter turns each message into
typed state deltas, an SOP layer checks them, a bitemporal state records them, and a
deterministic solver decides. Total regret then splits into interpretation regret and decision
regret.
"""

from cdd.schema import SCHEMA_VERSION, Delta, DeltaType, Message, Proposal, Span, Window

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "Delta",
    "DeltaType",
    "Message",
    "Proposal",
    "Span",
    "Window",
    "__version__",
]
