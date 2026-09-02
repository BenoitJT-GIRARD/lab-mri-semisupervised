"""The evaluation protocol.

Everything that touches a label lives here, and everything here respects one rule: the
test fold is read once, at the end, to produce predictions. It takes part in no decision —
not the choice of clustering method, not the cluster-to-class alignment, not the
checkpoint, not the decision threshold.

The rule is not a comment. `tests/test_no_leakage.py` flips every test label and asserts
that nothing upstream moves.
"""
