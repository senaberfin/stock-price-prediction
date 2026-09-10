"""
train.py

Shared training loop for the LSTM and GRU models.
"""
import time
from typing import List, Tuple

import torch
import torch.nn as nn


def train_model(
    model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
    num_epochs: int = 100, lr: float = 0.01, print_every: int = 10,
) -> Tuple[List[float], float]:
    """
    Train `model` on the full training set (no mini-batching, as in the
    reference implementation) using MSE loss and Adam.

    Returns the list of per-epoch training losses and the total training time (s).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{num_epochs} | Training MSE: {loss.item():.6f}")

    elapsed = time.time() - start
    return train_losses, elapsed
