"""
Package de visualisation pour le dashboard de maintenance prédictive.
"""

from .sensor_plots import (
    plot_temporal_evolution,
    plot_temperature_scatter,
    plot_speed_torque
)

from .correlation_plots import (
    plot_wear_distribution,
    plot_correlation_heatmaps,
    plot_feature_importance_comparison
)

__all__ = [
    'plot_temporal_evolution',
    'plot_temperature_scatter',
    'plot_speed_torque',
    'plot_wear_distribution',
    'plot_correlation_heatmaps',
    'plot_feature_importance_comparison'
]
