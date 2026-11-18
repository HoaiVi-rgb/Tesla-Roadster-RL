"""
Demonstration package for expert trajectory generation and dataset handling.
"""

from demonstration.teacher import HeuristicTeacherController
from demonstration.dataset import DemonstrationTransition, DemonstrationDataset
from demonstration.generator import DemonstrationGenerator

__all__ = [
    "HeuristicTeacherController",
    "DemonstrationTransition",
    "DemonstrationDataset",
    "DemonstrationGenerator",
]
