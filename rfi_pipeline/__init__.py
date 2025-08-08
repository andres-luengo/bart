"""
RFI Pipeline - Radio Frequency Interference Detection Pipeline

A Python package for detecting and analyzing radio frequency interference
in astronomical observation data.
"""

__version__ = "0.1.0"
__author__ = "Breakthrough Listen"
__description__ = "Radio Frequency Interference Detection Pipeline"

from .manager import RunManager
from .batchjob import BatchJob
from .filejob import FileJob

__all__ = ["RunManager", "BatchJob", "FileJob"]
