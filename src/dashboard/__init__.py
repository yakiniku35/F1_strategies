# F1 Dashboard Module
"""Dashboard components for data tables and prediction overlays."""

from src.dashboard.tables import (
    generate_driver_standings_table,
    generate_strategy_table,
    generate_battle_table,
    export_to_csv,
    export_to_json,
)

# PredictionOverlay requires arcade, so import it lazily
def get_prediction_overlay():
    """Get PredictionOverlay class (requires arcade)."""
    from src.dashboard.prediction_overlay import PredictionOverlay
    return PredictionOverlay

__all__ = [
    'generate_driver_standings_table',
    'generate_strategy_table',
    'generate_battle_table',
    'export_to_csv',
    'export_to_json',
    'get_prediction_overlay',
]
