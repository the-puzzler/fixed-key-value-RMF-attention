"""Baseline MLP for comparing against the spatial network."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    ImageField,
    load_image_field,
    plot_training_snapshot,
)


class BaselineMLP(nn.Module):
    """Simple feed-forward network with the same parameter budget as SpatialNet."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 14),
            nn.ReLU(inplace=True),
            nn.Linear(14, 31),
            nn.ReLU(inplace=True),
            nn.Linear(31, 26),
            nn.ReLU(inplace=True),
            nn.Linear(26, 3),
            nn.Sigmoid(),
        )

        param_count = sum(p.numel() for p in self.parameters())
        if param_count != 1420:
            raise ValueError(f"Expected 1420 parameters, found {param_count}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    data_path = os.path.join("data", "simple_image.jpg")
    coords, colors, image_shape = load_image_field(data_path)

    dataset = ImageField(coords, colors)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineMLP().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    coords_device = coords.to(device)
    height, width = image_shape
    colors_image = colors.view(height, width, 3)

    num_epochs = 4
    snapshot_every = 2500
    snapshot_dir = os.path.join("outputs", "mlp_snapshots")

    loss_history = []
    step = 0

    for epoch in range(num_epochs):
        with tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch") as progress:
            for inputs, target in progress:
                inputs = inputs.to(device)
                target = target.to(device)

                prediction = model(inputs)
                loss = criterion(prediction, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                loss_history.append(loss.item())
                progress.set_postfix(loss=f"{loss.item():.4f}")

                if step % snapshot_every == 0:
                    model.eval()
                    with torch.no_grad():
                        reconstruction = model(coords_device).view(height, width, 3).cpu()
                    model.train()

                    plot_training_snapshot(
                        save_path=os.path.join(snapshot_dir, f"step_{step:06d}.png"),
                        target_image=colors_image,
                        reconstruction=reconstruction,
                        neuron_positions=[],
                        neuron_weights=[],
                        losses=loss_history,
                        step=step,
                    )

    model.eval()
    with torch.no_grad():
        final_recon = model(coords_device).view(height, width, 3).cpu()

    plot_training_snapshot(
        save_path=os.path.join(snapshot_dir, "final.png"),
        target_image=colors_image,
        reconstruction=final_recon,
        neuron_positions=[],
        neuron_weights=[],
        losses=loss_history,
        step=step,
    )


if __name__ == "__main__":
    main()
