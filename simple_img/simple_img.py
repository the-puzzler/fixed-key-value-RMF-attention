'''
Here we will model a simple image, treating the network as a neural field:
the input space is x, y and the output space is 3 channel values
'''
#%%
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
    launch_interactive_inspector,
)

from spatialnet_model import SpatialNet
#%% 

import importlib
import utils
importlib.reload(utils)

#%%

path = 'data/simple_image.jpg'
coords, colors, image_shape = load_image_field(path)
height, width = image_shape

dataset = ImageField(coords, colors)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SpatialNet([20,20,20], [20,20], 2, 3)


# %%

#Train test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

coords_device = coords.to(device)
colors_image = colors.view(height, width, 3)

num_epochs = 4
snapshot_every = 5_000
snapshot_dir = os.path.join("outputs", "snapshots")
loss_history = []
step = 0

#%%
for epoch in range(num_epochs):
    with tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            loss_history.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % snapshot_every == 0:
                model.eval()
                with torch.no_grad():
                    recon = model(coords_device)
                model.train()

                recon_image = recon.view(height, width, 3).cpu()
                neuron_positions = [layer.positions for layer in model.layers]
                neuron_weights = [layer.weights for layer in model.layers]

                snapshot_path = os.path.join(snapshot_dir, f"step_{step:06d}.png")
                plot_training_snapshot(
                    save_path=snapshot_path,
                    target_image=colors_image,
                    reconstruction=recon_image,
                    neuron_positions=neuron_positions,
                    neuron_weights=neuron_weights,
                    losses=loss_history,
                    step=step,
                )

# %%
%matplotlib notebook
#%%

launch_interactive_inspector(model, coords, (height, width))
