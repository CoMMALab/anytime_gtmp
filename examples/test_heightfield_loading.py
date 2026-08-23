import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import vamp


WORLD = 20.0
Z = 1.0


def build_synthetic_png(out_path: Path, size: int = 256) -> Path:
    """Black blobs on a white background — easy to eyeball collisions against."""
    arr = np.full((size, size), 255, dtype=np.uint8)
    # A square in the upper-left, a square in the lower-right, a vertical strip
    # in the middle. Coordinates use PIL convention (row 0 = top).
    arr[20:60, 30:90] = 0          # near top-left of image
    arr[size - 70:size - 20, size - 80:size - 30] = 0  # near bottom-right
    arr[size // 2 - 5:size // 2 + 5, 100:size - 100] = 0  # horizontal bar
    arr[100:size - 100, size // 2 - 5:size // 2 + 5] = 0  # vertical bar
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(arr).save(out_path)
    return out_path


def load_heightfield(original_png_path: Path, radius: float):
    """Build a vamp env from an original PNG where black=wall.

    Returns (env, original_arr) so callers can run direct 2D checks against
    the same pixel buffer that defined the obstacles.
    """
    img = PILImage.open(original_png_path).convert('L')
    arr = np.array(img)            # original convention: black (≤128) = wall
    vamp_arr = 255 - arr           # invert: dark-in-original becomes bright,
                                   # which vamp lifts up as a tall obstacle.
    tmp = tempfile.mktemp(suffix='.png')
    PILImage.fromarray(vamp_arr).save(tmp)

    W, H = img.size
    env = vamp.Environment()
    hf = vamp.png_to_heightfield(
        tmp,
        (WORLD / 2.0, WORLD / 2.0, 0.0),
        (WORLD / W, WORLD / H, Z),
    )
    env.add_heightfield(hf)

    vamp.sphere.set_lows([0.0, 0.0, 0.0])
    vamp.sphere.set_highs([WORLD, WORLD, Z])
    vamp.sphere.set_radius(radius)
    return env, arr


def direct_2d_collision(x, y, arr_orig, radius_world):
    """Return True iff a 2D disk of radius_world centered at world (x, y)
    overlaps any black pixel (value ≤ 128) in arr_orig.

    World→pixel convention (matches vamp.png_to_heightfield, determined
    empirically — see top of this file's docstring):
        col = (1 - x / WORLD) * W        (X axis is FLIPPED in vamp's image)
        row = (y / WORLD) * H            (PIL row 0 at top → world y = 0)

    With radius_world == 0 the function degrades to a single-pixel lookup.
    """
    H, W = arr_orig.shape
    cx_p = (1.0 - x / WORLD) * W
    cy_p = y / WORLD * H
    rx_p = radius_world / WORLD * W
    ry_p = radius_world / WORLD * H

    if rx_p < 0.5 and ry_p < 0.5:
        col = int(np.clip(round(cx_p), 0, W - 1))
        row = int(np.clip(round(cy_p), 0, H - 1))
        return bool(arr_orig[row, col] <= 128)

    c0 = max(0, int(np.floor(cx_p - rx_p)))
    c1 = min(W, int(np.ceil(cx_p + rx_p)) + 1)
    r0 = max(0, int(np.floor(cy_p - ry_p)))
    r1 = min(H, int(np.ceil(cy_p + ry_p)) + 1)
    if c1 <= c0 or r1 <= r0:
        return False

    cols = np.arange(c0, c1)
    rows = np.arange(r0, r1)
    cc, rr = np.meshgrid(cols, rows, indexing='xy')
    dx = (cc - cx_p) / rx_p
    dy = (rr - cy_p) / ry_p
    in_disk = (dx * dx + dy * dy) <= 1.0
    sub = arr_orig[r0:r1, c0:c1]
    return bool(np.any(in_disk & (sub <= 128)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--png',
        default=None,
        help='PNG path to test. If omitted, a synthetic test PNG is generated.',
    )
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--z', type=float, default=0.5,
                        help='Sphere center z (must be < Z=1.0 to overlap obstacle columns).')
    parser.add_argument('--grid', type=int, default=80,
                        help='Grid resolution for the comparison map.')
    parser.add_argument('--output',
                        default='/home/scoumar/Work/anytime_gtmp/gallery/_heightfield_loading_test.png')
    args = parser.parse_args()

    if args.png is None:
        png_path = Path(tempfile.gettempdir()) / "vamp_heightfield_test.png"
        build_synthetic_png(png_path)
        print(f"Using synthetic PNG: {png_path}")
    else:
        png_path = Path(args.png)
        if not png_path.exists():
            raise FileNotFoundError(png_path)
        print(f"Using PNG: {png_path}")

    env, arr_orig = load_heightfield(png_path, radius=args.radius)
    H, W = arr_orig.shape
    print(f"PNG size: {W}×{H}  |  world {WORLD}×{WORLD}  |  radius={args.radius}")

    n = args.grid
    xs = np.linspace(0.0, WORLD, n)
    ys = np.linspace(0.0, WORLD, n)

    vamp_coll = np.zeros((n, n), dtype=bool)
    direct_coll = np.zeros((n, n), dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            cfg = np.array([x, y, args.z], dtype=np.float32)
            vamp_coll[j, i] = not vamp.sphere.validate(cfg, env)
            direct_coll[j, i] = direct_2d_collision(x, y, arr_orig, args.radius)

    agreement = float((vamp_coll == direct_coll).mean())
    only_vamp = float((vamp_coll & ~direct_coll).mean())
    only_direct = float((direct_coll & ~vamp_coll).mean())
    print(
        f"Agreement: {agreement:.2%}  "
        f"(vamp-only collisions: {only_vamp:.2%}, direct-only: {only_direct:.2%})"
    )
    print(f"vamp collision fraction: {vamp_coll.mean():.2%}")
    print(f"direct collision fraction: {direct_coll.mean():.2%}")

    # Display the original PNG re-oriented so the axes match vamp's world frame:
    #   row 0 → world y = 0 (bottom), col 0 → world x = WORLD (right).
    # That requires flipping X (np.fliplr) and using origin='lower'.
    arr_display = np.fliplr(arr_orig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 11))

    axs[0, 0].imshow(arr_display, extent=[0, WORLD, 0, WORLD],
                     origin='lower', cmap='gray', aspect='equal')
    axs[0, 0].set_title('Original PNG, vamp orientation\n(black = wall by convention)')

    # vamp.validate at the test grid.
    axs[0, 1].imshow((~vamp_coll).astype(float),
                     extent=[0, WORLD, 0, WORLD], origin='lower',
                     cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[0, 1].set_title(
        f'vamp.sphere.validate (white = free)\n'
        f'collision fraction = {vamp_coll.mean():.2%}'
    )

    # Direct 2D disk overlap check on original PNG.
    axs[1, 0].imshow((~direct_coll).astype(float),
                     extent=[0, WORLD, 0, WORLD], origin='lower',
                     cmap='gray', aspect='equal', vmin=0, vmax=1)
    axs[1, 0].set_title(
        f'Direct 2D disk-overlaps-black (white = free)\n'
        f'collision fraction = {direct_coll.mean():.2%}'
    )

    # Disagreement map: red = only vamp says collision, blue = only direct.
    disagree = np.zeros((n, n, 3), dtype=float)
    disagree[..., 0] = (vamp_coll & ~direct_coll).astype(float)   # red
    disagree[..., 2] = (direct_coll & ~vamp_coll).astype(float)   # blue
    disagree += (vamp_coll & direct_coll)[..., None] * 0.4        # both → gray
    axs[1, 1].imshow(disagree, extent=[0, WORLD, 0, WORLD],
                     origin='lower', aspect='equal')
    axs[1, 1].set_title(
        f'Disagreement (red = vamp-only, blue = direct-only)\n'
        f'agreement = {agreement:.2%}'
    )

    for ax in axs.ravel():
        ax.set_xlim(0, WORLD)
        ax.set_ylim(0, WORLD)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks([0, 5, 10, 15, 20])

    fig.suptitle(
        f'Heightfield loading sanity check  ({png_path.name})',
        fontsize=13,
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
