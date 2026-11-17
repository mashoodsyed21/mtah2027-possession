"""Possession segmentation: per-frame H/A/D inference and sequence decoding."""

# Settings shared by every script.
FEAT_DIR = "features_v4"   # feature cache: ../extdata/<FEAT_DIR>/ and <MTAH data>/derived/BAR-ATH_<FEAT_DIR>.npz
LGB_THREADS = 8            # LightGBM threads for every model. Fixed rather than "all cores": the thread
                           # count can change the floating-point summation order, hence the trees.
