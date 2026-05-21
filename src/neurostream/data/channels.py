"""Single source of truth for the 22-channel BCI IV 2a montage.

Open-corpus harmonisation uses canonical 10-10 names, which is what MOABB
datasets expose via MNE. Phase 1 ``loader.py`` reads the BCI IV 2a GDF files
with ``picks="eeg"`` — those channels carry an ``EEG-`` prefix in the raw GDF
and a future loader refactor (out of scope here) is the place to rename them
to match this constant.

Order matters: the encoder treats channels positionally, so harmonised
recordings must always present these channels in this exact order.
"""

# Brunner et al. 2008, "BCI Competition 2008 - Graz data set A".
# 22 channels over sensorimotor cortex in the 10-10 system.
BCI_IV_2A_22_CHANNELS: tuple[str, ...] = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)
