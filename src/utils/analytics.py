import numpy as np

def entropy_path(paths: np.ndarray) -> np.ndarray:
    frechet_matrix = np.linalg.norm(paths[:, None, :, :] - paths[None, :, :, :], axis=-1).sum(axis=-1)
    # normalize
    frechet_matrix = frechet_matrix / frechet_matrix.sum()
    # compute entropy
    entropy = -np.sum(frechet_matrix * np.log(frechet_matrix + 1e-12))
    return entropy