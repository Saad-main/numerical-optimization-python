from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optimization.problem import OptimizationProblem
from optimization.quasi_newton import BFGSMethod
from optimization.linesearch import InexactLineSearch
from problems.rosenbrock import rosenbrock, rosenbrock_gradient


class RecordedBFGSMethod(BFGSMethod):
    def initialize_matrix(self, x):
        super().initialize_matrix(x)
        self.matrix_history = []

    def compute_direction(self, x, gradient):
        # Enregistrer ensemble le point et la matrice qui y est utilisé
        self.matrix_history.append((x.copy(), self.H.copy()))

        return super().compute_direction(x, gradient)


def main():
    problem = OptimizationProblem(rosenbrock,rosenbrock_gradient)

    method = RecordedBFGSMethod(problem,tol=1e-6,max_iter=2000,line_search=InexactLineSearch(alpha1=1.0))

    xmin = method.minimize(np.array([-1.2, 1.0]))

    rows = []

    for k, (x, H) in enumerate(method.matrix_history):
        G = problem.hessian(x)
        identity = np.eye(len(x))

        condition_number = np.linalg.cond(G)
        relative_error = np.nan

      
        if np.isfinite(condition_number) and condition_number < 1e12:
            try:
                G_inverse = np.linalg.solve(G, identity)

                relative_error = (np.linalg.norm(H-G_inverse, ord="fro")/np.linalg.norm(G_inverse,ord="fro"))
            except np.linalg.LinAlgError:
                pass

        inverse_residual=np.linalg.norm(G@H-identity,ord="fro")

        rows.append([k,relative_error,inverse_residual,condition_number,np.linalg.norm(problem.grad(x))])

    data = np.array(rows)

    print("Point final :", xmin)
    print("Convergence :", method.converged)
    print("Nombre d'itérations :", method.n_iter)

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "results" / "data"
    figure_dir = project_dir / "results" / "figures"

    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(data_dir/"bfgs_hessian_quality.csv",data,delimiter=",",header="iteration,relative_error,inverse_residual,condition_number,gradient_norm",comments="")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].semilogy(data[:, 0],np.maximum(data[:, 1], 1e-16),"o-")
    axes[0].set_title("Erreur relative sur l'inverse")
    axes[0].set_ylabel("Erreur relative")

    axes[1].semilogy(data[:, 0],np.maximum(data[:, 2], 1e-16),"o-")
    axes[1].set_title("Résidu ||G H - I||")
    axes[1].set_ylabel("Norme de Frobenius")

    for ax in axes:
        ax.set_xlabel("Itération k")
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(figure_dir / "bfgs_hessian_quality.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
