# SIREN Image Fitting — Reproduction

A from-scratch PyTorch reproduction of **SIREN** (Sinusoidal Representation Networks, Sitzmann et al., CVPR 2020) on single-image fitting: the network maps pixel coordinates `(x, y)` to RGB values, and the trained weights become an implicit representation of the image.

This is a learning reproduction, not original research.

## Results

| Setting | Activation / Initialization | PSNR |
|---|---|---|
| Proposed | sine + matched initialization | **59.75 dB** |
| Control A | ReLU | 23.85 dB |
| Control B | sine, default (unmatched) initialization | 24.42 dB |

The controlled experiment shows the gain comes mainly from the **matched initialization**, not from the sine activation alone. This is the attribution result behind the numbers above.

## Files

| File | What it is |
|---|---|
| `siren_step_by_step.py` | Guided walkthrough (dm01–dm07) with intermediate shape prints; matched sine initialization **enabled** |
| `siren_image_fitting.py` | Compact training script with PSNR evaluation and result figures; initialization **commented out** on purpose — it is the unmatched-init control group |
| `data/siren_data/` | Training progress, loss curves and final comparison figures for both settings |
| `model/` | Saved checkpoints for the ablation runs |
| `data/alba-photo-027.png` | Target image (square, resized to 256×256 or 512×512 at runtime) |

Naming note: figures prefixed `siren_` are the matched-init run; figures prefixed `siren未配准_` (unmatched) are the control group with initialization disabled.

## Run

```bash
pip install -r requirements.txt

python siren_step_by_step.py     # walkthrough, matched init
python siren_image_fitting.py    # compact version, unmatched-init control
```

Both scripts expect a square target image (set `TARGET_IMAGE` at the top) and write figures to `./data/siren_data/`.

## Reference

Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020).
*Implicit Neural Representations with Periodic Activation Functions.* CVPR.

## License

MIT — for learning and reference.
