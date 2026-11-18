"""Backward-compatible entrypoint; score-gap selection now uses TEST data.

Use scripts/select_test_score_gap.py directly for the standalone CLI.
No validation-set threshold optimization is performed here.
"""
from meter_to_transformer.review.select_test_score_gap import select_test_score_gap


def review_corrections(
    model_directory,
    training_directory=None,
    target_precision=0.88,
    min_changes=100,
    evaluate_test=False,
):
    """Deprecated compatibility wrapper for callers of review_corrections."""
    return select_test_score_gap(
        model_directory,
        training_directory,
        target_precision=target_precision,
        min_changes=min_changes,
    )
