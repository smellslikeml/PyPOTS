"""
Dataset classes for the imputation model MDTIM.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ..csdi.data import DatasetForCSDI, TestDatasetForCSDI

# MDTIM conditions on exactly the same anchors as CSDI: observed values are carried
# over untouched and the artificially-held-out positions are the supervision targets,
# so the CSDI dataset classes (observed data, indicating mask, conditioning mask, and
# observed time points) serve MDTIM unchanged and are re-exported here for convenience.
DatasetForMDTIM = DatasetForCSDI
TestDatasetForMDTIM = TestDatasetForCSDI
