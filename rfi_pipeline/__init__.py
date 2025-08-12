"""
RFI Pipeline - Radio Frequency Interference Detection Pipeline

A Python package for detecting and analyzing radio frequency interference
in astronomical observation data.
"""

#: The version string of the RFI Pipeline package
__version__ = "0.1.0"

#: The author/organization that developed the RFI Pipeline
__author__ = "Breakthrough Listen"

#: A brief description of the RFI Pipeline package
__description__ = "Radio Frequency Interference Detection Pipeline"

from .manager import RunManager

__all__ = ["RunManager"]
