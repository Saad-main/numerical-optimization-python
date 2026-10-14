from pathlib import Path
from time import perf_counter
import csv

import numpy as np
from scipy.optimize import fmin_bfgs

from optimization.problem import OptimizationProblem
from optimization.quasi_newton import BFGSMethod
from optimization.linesearch import InexactLineSearch
from problems.chebyquad_problem import chebyquad, gradchebyquad


def main():
    tol = 1e-6
    max_iter = 2000

    problem = OptimizationProblem(chebyquad, gradchebyquad)

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "results" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for n in [4, 8, 11]:
        # Même point initial pour les deux méthodes.
        x0 = np.linspace(0.0, 1.0, n)

        method = BFGSMethod(
            problem,
            tol=tol,
            max_iter=max_iter,
            line_search=InexactLineSearch(alpha1=1.0),
        )

        start = perf_counter()
        x_ours = method.minimize(x0.copy())
        time_ours = perf_counter() - start

        # Conserver les itérés de SciPy pour compter les déplacements.
        scipy_history = []

        def record_iteration(xk):
            scipy_history.append(xk.copy())

        start = perf_counter()

        scipy_result = fmin_bfgs(
            chebyquad,
            x0.copy(),
            fprime=gradchebyquad,
            gtol=tol,
            norm=2,
            maxiter=max_iter,
            c1=0.01,
            c2=0.1,
            full_output=True,
            disp=False,
            callback=record_iteration,
        )

        time_scipy = perf_counter() - start

        (
            x_scipy,
            f_scipy,
            g_scipy,
            H_scipy,
            function_calls,
            gradient_calls,
            warnflag,
        ) = scipy_result

        results = [
            (
                "Notre BFGS",
                x_ours,
                problem.value(x_ours),
                np.linalg.norm(problem.grad(x_ours)),
                method.n_iter,
                method.converged,
                time_ours,
            ),
            (
                "SciPy BFGS",
                x_scipy,
                f_scipy,
                np.linalg.norm(g_scipy),
                len(scipy_history),
                warnflag == 0,
                time_scipy,
            ),
        ]

        print(f"\nDimension n = {n}")

        for name, points, value, grad_norm, iterations, success, duration in results:
            print(f"\n{name}")
            print("Points triés :", np.sort(points))
            print("Valeur finale :", value)
            print("Norme du gradient :", grad_norm)
            print("Itérations :", iterations)
            print("Convergence :", success)
            print("Temps :", duration)
            print("Points dans [0, 1] :", np.all((points >= 0) & (points <= 1)))

            rows.append({
                "n": n,
                "method": name,
                "objective": value,
                "gradient_norm": grad_norm,
                "iterations": iterations,
                "converged": success,
                "time_seconds": duration,
            })

        print("\nCode de statut SciPy :", warnflag)

        np.savetxt(
            output_dir / f"chebyquad_points_n{n}.csv",
            np.column_stack([
                np.sort(x_ours),
                np.sort(x_scipy),
            ]),
            delimiter=",",
            header="our_bfgs,scipy_bfgs",
            comments="",
        )

    with (output_dir / "chebyquad_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
