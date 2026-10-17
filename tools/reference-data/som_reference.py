"""Self-organizing map reference: MiniSom configured with SOM_PAK's schedules.

SOM_PAK (Kohonen, Hynninen, Kangas & Laaksonen 1996, the reference package from Kohonen's laboratory) decays the
learning rate linearly to zero and the neighborhood radius linearly to one over the training length T, with a Gaussian
kernel exp(-d^2 / 2r^2) of the Euclidean lattice distance. MiniSom reproduces that on a rectangular grid with
decay_function='linear_decay_to_zero', sigma_decay_function='linear_decay_to_one' and neighborhood_function='gaussian'.
(MiniSom's hexagonal lattice shifts different rows than SOM_PAK's, and its bubble is a box, so those two are checked
against SOM_PAK's formulas in the C# tests instead.)

A 4x5 map (4 rows of 5) of 3-dimensional vectors, initial learning rate 0.5 and radius 2.5, is trained for 40 steps
in sequential order over 12 inputs, from given initial weights, in float64. The fixture records the inputs, the initial
codebook, the winner of every step, and the final codebook (row-major neurons).
Default: verify; --write: regenerate tests/AiDotNet.Tests/UnitTests/NeuralNetworks/ReferenceData/som_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/UnitTests/NeuralNetworks/ReferenceData/som_reference.json"


def compute():
    import numpy as np
    from minisom import MiniSom
    rows, columns, dim, steps = 4, 5, 3, 40
    rng = np.random.default_rng(11)
    data = rng.normal(size=(12, dim))
    initial = rng.uniform(-1, 1, size=(rows, columns, dim))
    som = MiniSom(rows, columns, dim, sigma=2.5, learning_rate=0.5, decay_function="linear_decay_to_zero",
                  sigma_decay_function="linear_decay_to_one", neighborhood_function="gaussian",
                  topology="rectangular", activation_distance="euclidean")
    som._weights = initial.copy()
    winners = []
    for t in range(steps):
        x = data[t % len(data)]
        w = som.winner(x)
        winners.append(int(w[0] * columns + w[1]))
        som.update(x, w, t, steps)
    return {"rows": rows, "columns": columns, "dim": dim, "steps": steps, "learning_rate": 0.5, "radius": 2.5,
            "data": data.flatten().tolist(), "initial": initial.flatten().tolist(), "winners": winners,
            "final": som.get_weights().flatten().tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["winners"])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = stored["winners"] == result["winners"] and max(
        abs(a - b) for a, b in zip(stored["final"], result["final"])) < 1e-12
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
