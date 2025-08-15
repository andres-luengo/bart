"""
BART - Breakthrough Listen's RFI analysis and batching toolkit

A Python package for detecting and analyzing radio frequency interference
in astronomical observation data.
"""

#: The version string of the BART package
__version__ = "0.2.0"

#: The author/organization that developed BART
__author__ = "Breakthrough Listen"

#: A brief description of the BART package
__description__ = "BART - Breakthrough Listen's RFI analysis and batching toolkit"

from .manager import RunManager

__all__ = ["RunManager"]
