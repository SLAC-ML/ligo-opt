"""Shared power objective for optimization, training defaults, and analysis.

The input beam q remains ITM.p1.i; that is a beam parameter, not this detector.
"""

POWER_NODE = "ITM.p2.i"
POWER_DETECTOR = "p_" + POWER_NODE.replace(".", "_")
