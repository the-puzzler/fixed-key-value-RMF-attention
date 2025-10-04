import os
from typing import Sequence, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def load_image_field(path):
    img = Image.open(path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0       # (H, W, 3)
    H, W, _ = img_tensor.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )
    coords = torch.stack([xs, ys], dim=-1).view(-1, 2)                 # (H*W, 2)
    colors = img_tensor.view(-1, 3)                                    # (H*W, 3)
    return coords, colors, (H, W)


class ImageField(Dataset):
    def __init__(self, coords, colors):
        self.coords = coords
        self.colors = colors

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.colors[idx]


def _compute_effective_positions(
    positions: Sequence[Optional[np.ndarray]],
    weights: Sequence[Optional[np.ndarray]],
    temperature: float = 1.0,
) -> Sequence[Optional[np.ndarray]]:
    """Map higher-layer positions back into input space via soft assignments."""

    if not positions:
        return []

    effective = []
    base = positions[0]
    if base is None:
        return [None for _ in positions]

    effective.append(base[:, :2])
    prev_effective = effective[0]

    for layer_idx in range(1, len(positions)):
        current_pos = positions[layer_idx]
        current_weights = weights[layer_idx - 1] if weights else None

        if (
            current_pos is None
            or prev_effective is None
            or current_weights is None
        ):
            effective.append(None)
            prev_effective = None
        else:
            dists_sq = ((current_weights[:, None, :] - current_pos[None, :, :]) ** 2).sum(axis=2)
            coeffs = np.exp(-dists_sq / max(temperature, 1e-8))
            coeffs_sum = coeffs.sum(axis=0, keepdims=True)
            coeffs = coeffs / (coeffs_sum + 1e-8)
            eff_pos = coeffs.T @ prev_effective
            effective.append(eff_pos)
            prev_effective = eff_pos

    return effective


def plot_training_snapshot(
    save_path: str,
    target_image: torch.Tensor,
    reconstruction: torch.Tensor,
    neuron_positions: Sequence[torch.Tensor],
    losses: Sequence[float],
    step: int,
    neuron_weights: Optional[Sequence[torch.Tensor]] = None,
    smooth_window: int = 100,
) -> None:
    """Visualize training progress with images, neuron layout, and loss curve."""

    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    target_np = target_image.detach().cpu().numpy() if torch.is_tensor(target_image) else target_image
    recon_np = (
        reconstruction.detach().cpu().numpy() if torch.is_tensor(reconstruction) else reconstruction
    )

    target_np = np.clip(target_np, 0.0, 1.0)
    recon_np = np.clip(recon_np, 0.0, 1.0)

    height, width, _ = target_np.shape

    gray = target_np.mean(axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(target_np)
    axes[0, 0].set_title("Target")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(recon_np)
    axes[0, 1].set_title("Reconstruction")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(gray, cmap="gray")
    axes[1, 0].set_title("Neuron Layout")
    axes[1, 0].axis("off")

    cmap = plt.cm.get_cmap("tab10", max(len(neuron_positions), 1))
    positions_np = [
        pos.detach().cpu().numpy() if torch.is_tensor(pos) else None
        for pos in neuron_positions
    ]
    weights_np = (
        [w.detach().cpu().numpy() if torch.is_tensor(w) else None for w in neuron_weights]
        if neuron_weights is not None
        else [None] * len(neuron_positions)
    )

    effective_positions = _compute_effective_positions(positions_np, weights_np)

    for layer_idx, positions in enumerate(effective_positions):
        if positions is None:
            continue
        xs = ((positions[:, 0] + 1.0) * 0.5) * (width - 1)
        ys = ((positions[:, 1] + 1.0) * 0.5) * (height - 1)
        axes[1, 0].scatter(xs, ys, s=30, color=cmap(layer_idx), label=f"Layer {layer_idx}")

    if len(neuron_positions) > 0:
        axes[1, 0].legend(loc="upper right", fontsize="small")

    axes[1, 0].set_xlim(-0.5, width - 0.5)
    axes[1, 0].set_ylim(height - 0.5, -0.5)

    axes[1, 1].set_title("Loss")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("MSE")

    if losses:
        losses_arr = np.asarray(losses, dtype=np.float32)
        steps = np.arange(1, len(losses_arr) + 1)
        axes[1, 1].plot(steps, losses_arr, alpha=0.3, label="loss")

        if len(losses_arr) >= smooth_window:
            kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
            smooth = np.convolve(losses_arr, kernel, mode="valid")
            smooth_steps = steps[smooth_window - 1 :]
        else:
            smooth = losses_arr
            smooth_steps = steps
        axes[1, 1].plot(smooth_steps, smooth, label=f"mean({smooth_window})")
        axes[1, 1].legend()

    fig.suptitle(f"Step {step}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def launch_interactive_inspector(
    model: torch.nn.Module,
    coords: torch.Tensor,
    image_shape: Sequence[int],
    cmap_name: str = "magma",
) -> None:
    """Interactive viewer linking reconstruction to layer-wise neuron influence."""

    if coords.dim() != 2:
        raise ValueError("coords must be (N, input_dim)")

    was_training = model.training
    first_param = next(model.parameters())
    original_device = first_param.device
    param_dtype = first_param.dtype
    model.to("cpu")
    model.eval()

    coords_cpu = coords.detach().cpu()
    height, width = image_shape

    with torch.no_grad():
        recon = model(coords_cpu).view(height, width, -1).clamp(0.0, 1.0)

    recon_np = recon.numpy()
    cmap = plt.cm.get_cmap(cmap_name)

    num_layers = len(getattr(model, "layers", []))
    if num_layers == 0:
        raise ValueError("model must have at least one layer")

    fig, (ax_img, ax_net) = plt.subplots(1, 2, figsize=(12, 6))
    ax_img.imshow(recon_np)
    ax_img.set_title("Reconstruction")
    ax_img.axis("off")

    ax_net.set_title("Neuron Influence")
    ax_net.set_ylim(-0.5, 1.5)
    ax_net.set_xlim(-0.5, num_layers - 0.5)
    ax_net.set_xticks(range(num_layers))
    ax_net.set_xticklabels([f"L{i}" for i in range(num_layers)])
    ax_net.set_yticks([])

    scatter_artists = []
    plotted_indices = []
    for layer_idx, layer in enumerate(model.layers):
        num = layer.num_neurons
        if num == 0:
            continue
        ys = np.linspace(0.05, 0.95, num)
        xs = np.full(num, layer_idx, dtype=float)
        sc = ax_net.scatter(
            xs,
            ys,
            s=40,
            c=[(0.7, 0.7, 0.7, 0.4)] * num,
            edgecolors="k",
            linewidths=0.4,
        )
        scatter_artists.append(sc)
        plotted_indices.append(layer_idx)

    marker = ax_img.scatter([], [], s=120, facecolors="none", edgecolors="cyan", linewidths=1.5)
    info_text = ax_net.text(0.02, 1.05, "Click image to inspect influence", transform=ax_net.transAxes)

    def highlight(point_tensor: torch.Tensor) -> None:
        activations = []
        current = point_tensor
        with torch.no_grad():
            for layer in model.layers:
                g = layer.gaussian_weights(current)
                activations.append(g.squeeze(0).cpu().numpy())
                current = (g / layer.num_neurons) @ layer.weights

        for sc, layer_idx in zip(scatter_artists, plotted_indices):
            values = activations[layer_idx]
            if values.size == 0:
                count = sc.get_offsets().shape[0]
                sc.set_facecolors([(0.7, 0.7, 0.7, 0.2)] * count)
                sc.set_sizes(np.full(count, 40.0))
                continue
            if values.ndim:
                v = values
            else:
                v = np.array([values])
            vmax = np.max(v)
            if vmax <= 0.0:
                norm = np.zeros_like(v)
            else:
                norm = v / vmax
            colors = cmap(np.clip(norm, 0.0, 1.0))
            sc.set_facecolors(colors)
            sc.set_sizes(60 + 140 * norm)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return

        x_pix = float(event.xdata)
        y_pix = float(event.ydata)
        marker.set_offsets([[x_pix, y_pix]])

        x_norm = (x_pix / max(width - 1, 1)) * 2.0 - 1.0
        y_norm = (y_pix / max(height - 1, 1)) * 2.0 - 1.0
        point = torch.tensor([[x_norm, y_norm]], dtype=param_dtype)

        highlight(point)
        info_text.set_text(f"Clicked ({x_norm:+.2f}, {y_norm:+.2f})")
        fig.canvas.draw_idle()

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)

    def on_close(event):
        marker.remove()
        fig.canvas.mpl_disconnect(cid_click)
        model.to(original_device)
        if was_training:
            model.train()

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()
