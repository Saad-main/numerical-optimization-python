from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimization.problem import OptimizationProblem
from optimization.newton import NewtonMethod
from optimization.linesearch import ExactLineSearch
from problems.rosenbrock import rosenbrock, rosenbrock_gradient


def main():
    problem = OptimizationProblem(
        rosenbrock,
        rosenbrock_gradient,
    )

    x0 = np.array([-1.2, 1.0])

    methods = {
        "Newton, pas fixe": NewtonMethod(
            problem,
            max_iter=200,
        ),
        "Newton, recherche exacte": NewtonMethod(
            problem,
            max_iter=200,
            line_search=ExactLineSearch(),
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, method) in zip(axes, methods.items()):
        xmin = method.minimize(x0)

        print(name)
        print("Point final :", xmin)
        print("Valeur finale :", problem.value(xmin))
        print("Norme du gradient :", np.linalg.norm(problem.grad(xmin)))
        print("Convergence :", method.converged)
        print("Nombre d'itérations :", method.n_iter)
        print()

        trajectory = np.array(method.history)

        # Inclure tous les points visités et le minimum dans le graphique.
        points = np.vstack([trajectory, [1.0, 1.0]])
        lower = points.min(axis=0) - 0.5
        upper = points.max(axis=0) + 0.5

        x_values = np.linspace(lower[0], upper[0], 400)
        y_values = np.linspace(lower[1], upper[1], 400)

        X, Y = np.meshgrid(x_values, y_values)
        Z = rosenbrock([X, Y])

        ax.contour(
            X, Y, Z,
            levels=np.logspace(-2, 5, 20),
            cmap="viridis",
        )

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "o-",
            color="red",
            markersize=3,
            label="Itérations",
        )

        ax.plot(
            1, 1,
            "*",
            color="black",
            markersize=12,
            label="Minimum",
        )

        ax.set_title(name)
        ax.set_xlabel("x₀")
        ax.set_ylabel("x₁")
        ax.legend()

    fig.tight_layout()

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / "rosenbrock_newton.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
