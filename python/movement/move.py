import numpy as np
from numpy.typing import NDArray


def get_random_orthogonal_vector(
    vector
):
    """
    Return a random unit vector orthogonal to `vector`.

    The direction is uniformly random over the unit sphere in the
    subspace orthogonal to `vector`.
    """
    if vector.ndim != 1 or len(vector) < 2:
        raise ValueError("vector must be a 1D vector with at least 2 elements.")

    # Work in float64 for numerical stability.
    v = vector.astype(np.float64)

    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("Cannot find an orthogonal vector to the zero vector.")

    # Generate a random vector.
    random_vector = np.random.standard_normal(len(vector))

    # Remove the component parallel to v.
    random_vector -= (
        np.dot(random_vector, v) / np.dot(v, v)
    ) * v

    # Normalise.
    norm = np.linalg.norm(random_vector)

    return (random_vector / norm).astype(np.float32)


def move_on_unit_sphere(
    x: NDArray[np.float32],
    v: NDArray[np.float32],
    d: float,
) -> NDArray[np.float32]:
    """
    x: Initial vector. Should be l2 normed.
    v: The direction to move the vector.
    d: The angle to move the vector on the hypersphere.
    """
    tangent = v - np.dot(v, x) * x
    tangent_norm = np.linalg.norm(tangent)

    if tangent_norm <= 1e-12:
        raise ValueError("v has no nonzero component tangent to the sphere at x")

    tangent_unit = tangent / tangent_norm

    result = np.cos(d) * x + np.sin(d) * tangent_unit
    return result.astype(np.float32, copy=False)

def move_example():
    """
    An example of how to move an example normed vector using the move_on_unit_sphere function.
    :return: nothing
    """
    for i in range(10):
        x = np.array([0.0, 0.0, 1.0])                # The vector to be moved
        v = np.random.normal(0,1,3)   # The directional vector. This is NOT sensitive to magnitude.
        d = 0.1                                      # The desired angle between x and m, see below.
        m = move_on_unit_sphere(x, v, d)             # Generate the moved vector

        # Verify properties
        print(f"Moved vector: {m}")
        print(f"Angle between x and m (should be 0.1):  {np.arccos(np.dot(x,m)): .3f}")

