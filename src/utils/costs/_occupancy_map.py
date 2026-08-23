import tempfile
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import jax.numpy as jnp
import vamp


def get_data_path():
    return Path(__file__).parent.parent.parent.parent / "data"


def build_environment(map_file: str = 'intel.npy', z: float = 1.0, radius: float = 1):
    npy_path = get_data_path() / 'real_map' / map_file
    data = np.load(npy_path)
    if data.ndim == 3:
        data = data[:, :, 0]
    # .npy convention: data[x, y] in [0.5, 1.0]; GTMP treats data >= 0.9 as free.
    # VAMP png_to_heightfield: pixel 255 = tall terrain (wall), 0 = flat (free).
    # PNG layout: rows → world y, cols → world x (flipped). So transpose + flip axis=1.
    H, W = data.shape[:2]
    wall_mask = data < 0.9
    png_layout = np.flip(wall_mask.T, axis=1)
    png_array = (png_layout * 255).astype(np.uint8)

    tmp_png = tempfile.mktemp(suffix='.png')
    PILImage.fromarray(png_array).save(tmp_png)

    env = vamp.Environment()
    hf = vamp.png_to_heightfield(
        tmp_png,
        (10.0, 10.0, 0.0),      # center of [0,20]x[0,20] world
        (20.0 / W, 20.0 / H, z),
    )
    env.add_heightfield(hf)

    vamp.sphere.set_lows([0, 0, 0])
    vamp.sphere.set_highs([20, 20, z])
    vamp.sphere.set_radius(radius)

    bounds = jnp.array([[0., 20.], [0., 20.], [0., z]])
    return env, bounds, npy_path
