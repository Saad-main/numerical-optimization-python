from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimization.problem import OptimizationProblem
from optimization.linesearch import InexactLineSearch
from problems.rosenbrock import rosenbrock, rosenbrock_gradient


def main():
    problem = OptimizationProblem(
        rosenbrock,
        rosenbrock_gradient,
    )

    x = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])

    line_search = InexactLineSearch(
        rho=0.01,
        sigma=0.1,
        tau1=9.0,
        tau2=0.1,
        tau3=0.5,
        alpha1=0.1,
        f_lower=0.0,
    )

    def phi(alpha):
        return problem.value(x + alpha * direction)

    def dphi(alpha):
        return np.dot(
            problem.grad(x + alpha * direction),
            direction,
        )

    alpha = line_search.search(problem, x, direction)

    phi0 = phi(0.0)
    slope0 = dphi(0.0)

    sufficient_decrease = (
        phi(alpha) <= phi0 + line_search.rho * alpha * slope0
    )

    strong_curvature = (
        abs(dphi(alpha)) <= -line_search.sigma * slope0
    )

    print("Pas accepté :", alpha)
    print("Nouveau point :", x + alpha * direction)
    print("Diminution suffisante :", sufficient_decrease)
    print("Courbure forte :", strong_curvature)

    # Grille utilisée pour visualiser les fonctions.
    alpha_max = max(0.3, 1.5 * alpha)
    alphas = np.linspace(0.0, alpha_max, 600)

    values = np.array([phi(a) for a in alphas])
    slopes = np.array([dphi(a) for a in alphas])

    armijo_line = phi0 + line_search.rho * alphas * slope0
    curvature_limit = -line_search.sigma * slope0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Condition de diminution suffisante.
    axes[0].plot(alphas, values, label="φ(α)")
    axes[0].plot(
        alphas,
        armijo_line,
        "--",
        label="φ(0) + ρ α φ′(0)",
    )

    axes[0].fill_between(
        alphas,
        values,
        armijo_line,
        where=values <= armijo_line,
        color="green",
        alpha=0.15,
        label="Diminution suffisante",
    )

    axes[0].plot(alpha, phi(alpha), "ro", label="Pas accepté")
    axes[0].set_title("Diminution suffisante")
    axes[0].set_ylabel("Valeur de φ")

    # Condition de courbure forte.
    axes[1].plot(alphas, slopes, label="φ′(α)")
    axes[1].axhspan(
        -curvature_limit,
        curvature_limit,
        color="green",
        alpha=0.15,
        label="Courbure forte satisfaite",
    )

    axes[1].axhline(curvature_limit, color="green", linestyle="--")
    axes[1].axhline(-curvature_limit, color="green", linestyle="--")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)

    axes[1].plot(alpha, dphi(alpha), "ro", label="Pas accepté")
    axes[1].set_title("Courbure forte")
    axes[1].set_ylabel("Pente φ′")

    for ax in axes:
        ax.axvline(alpha, color="red", linestyle=":", linewidth=1)
        ax.set_xlabel("Pas α")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / "line_search_conditions.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
