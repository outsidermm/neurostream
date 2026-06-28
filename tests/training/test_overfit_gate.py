"""
Overfit gate: EEGNet must memorise 288 trials for subject A01.

This test is the single most important diagnostic in the repo.
If it fails, the model, data pipeline, or loss function is broken.
Do not proceed to full training until this passes.

Run with: pytest tests/test_overfit_gate.py -v -s
"""

import pytest
import torch
import torch.nn as nn

from neurostream.data.bci_iv_loader import load_subject
from neurostream.models.eegnet import EEGNet
from neurostream.preprocessing.filters import BandpassParams
from neurostream.preprocessing.bci_pipeline import PipelineConfig, fit_pipeline
from neurostream.training.train import set_deterministic_seed


SEED = 42
EPOCHS = 200
TARGET_TRAIN_ACC = 0.99  # 100% is ideal but 99% allows for rare GDF label noise


@pytest.mark.slow  # skip in fast CI; run explicitly before full training
def test_overfit_single_subject() -> None:
    set_deterministic_seed(SEED)
    device = torch.device("cpu")  # CPU is fine for 288 trials

    epochs_arr, labels_arr = load_subject(subject_id=1, session="T")

    # Preprocessing: fit on full training session (no val split for overfit check)
    bandpass = BandpassParams(low_hz=4.0, high_hz=40.0, fs_hz=128.0, order=4)
    pipeline = fit_pipeline(epochs_arr, config=PipelineConfig(bandpass=bandpass))
    X = pipeline.transform(epochs_arr)

    # Full-batch overfit: one tensor for X, one for y, no DataLoader needed
    X_batch = torch.from_numpy(X).float().to(device)
    y_batch = torch.from_numpy(labels_arr).long().to(device)

    model = EEGNet(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        dropout=0.0,  # no regularisation — must memorise
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_batch).argmax(1) == y_batch).float().mean().item()
            print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | train_acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_batch).argmax(1) == y_batch).float().mean().item()

    assert train_acc >= TARGET_TRAIN_ACC, (
        f"Overfit gate failed: train_acc={train_acc:.4f} < {TARGET_TRAIN_ACC}. "
        "Check: label encoding, loss function, model forward pass, data shapes."
    )
