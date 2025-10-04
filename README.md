# Spatial Net: Continuous Attention with Radial Metric Fields

A small research playground for modelling 2D neural fields with attention-style interactions between neurons that live in continuous space. The project compares three approaches that all share the same parameter budget and training recipe:

- A radial metric field (RMF) version of my spatial attention block with learnable per-neuron bandwidths.
- A dot-product attention variant that anchors queries/values to static neuron locations.
- A vanilla MLP baseline.

## Key Things I Learned
- Attention on continuous vectors still needs anchors: queries and values have to be tied to explicit neuron positions, otherwise the model has no spatial reference frame.
- Replacing dot-product similarity with the RMF distance metric gave noticeably better reconstructions. The learnable `sigma` term encourages locality, so neurons only compete with nearby neighbours instead of the entire grid.
- Parameter and compute parity matters. Holding these constant across models made the qualitative differences in the reconstructions much easier to attribute to the attention mechanism itself.

## Repository Tour
- `spatialnet_model.py` – Implements `SpatialBlock` and `SpatialNet`, including the RMF attention and an optional dot-product variant.
- `simple_img/simple_img.py` – Trains the RMF SpatialNet on a toy image-as-field task and logs snapshots.
- `simple_img/mlp_for_comparison.py` – Same training loop but with the MLP baseline (hard-checks the parameter count to 1,420).
- `simple_img/utils.py` – Data loading helpers plus plotting utilities for loss curves and neuron layouts.
- `comparison.png` – Side-by-side target, RMF output, MLP output, and dot-product output from the matched training runs.

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
The project uses PyTorch, NumPy, Matplotlib, and tqdm (see `pyproject.toml`). GPU is optional but cuts runtime for the image field experiment.

## Reproducing the Experiments
- RMF SpatialNet: `python -m simple_img.simple_img`
- MLP baseline: `python -m simple_img.mlp_for_comparison`
- Dot-product SpatialNet: switch `SpatialBlock.forward` to use the provided `dot_product_weights` implementation before running the RMF script. This keeps the rest of the pipeline and parameter counts identical.

All scripts write progress snapshots to `simple_img/outputs/`. The default schedule trains for four epochs with `Adam(lr=1e-3)` and logs reconstructions every 2,500 steps.

## Results
![Target vs. RMF vs. MLP vs. Dot-Product](comparison.png)

Going left-to-right: (1) the target image, (2) RMF attention with fixed query/value anchors, (3) the MLP baseline, and (4) dot-product attention with the same anchors. The RMF version stays sharper around edges and colour transitions, strongly suggesting that the learnable locality enforced by `sigma` helps the network specialise neurons to regions of the image.

## Next Questions I Want to Explore
- Learn the anchor positions jointly with the bandwidths instead of fixing them per layer.
- Push beyond 2D fields (e.g., 3D occupancy or video) to see whether RMF still outperforms dot-product similarity.
- Add quantitative metrics (PSNR/SSIM) alongside the qualitative snapshots to track improvements.
