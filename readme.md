# Project 3 — Numerical Optimization in Python

Project for NUMN21/FMNN25: Advanced Numerical Algorithms in Python.

Object-oriented implementation of Newton and quasi-Newton methods,
with numerical derivatives and exact and inexact line searches.

## Implemented methods

- Classical Newton method with a numerical, symmetrized Hessian
- Good Broyden with the Sherman–Morrison inverse update
- Bad Broyden
- Symmetric Rank One (SR1)
- Davidon–Fletcher–Powell (DFP)
- Broyden–Fletcher–Goldfarb–Shanno (BFGS)

The optimization problem accepts an objective function and an optional
gradient. When the gradient is omitted, it is approximated using
central finite differences.

The exact line search uses bracketing and golden-section search.
The inexact line search follows Fletcher's bracketing and sectioning
algorithms, including the corrected sufficient-decrease test.
Trial steps are selected using midpoints of the permitted intervals,
rather than cubic interpolation.

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Experiments

Run these commands from the project root.

Compare Newton with a fixed step and an exact line search on Rosenbrock:

```bash
python3 -m experiments.rosenbrock_experiment
```

Test the inexact line search independently using Fletcher's
Rosenbrock example and visualize its acceptance conditions:

```bash
python3 -m experiments.line_search_experiment
```

Compare our BFGS implementation with scipy.optimize.fmin_bfgs
on Chebyquad for dimensions 4, 8 and 11:

```bash
python3 -m experiments.chebyquad_experiment
```

Study the BFGS inverse-Hessian approximation on Rosenbrock:

```bash
python3 -m experiments.bfgs_hessian_experiment
```

Figures are saved in results/figures.
Numerical results are saved in results/data.

## Project organization

- optimization/: problem class, numerical derivatives, optimization
  methods and line searches
- problems/: Rosenbrock and the supplied Chebyquad problem
- experiments/: executable numerical experiments
- tests/: reserved for numerical verification
- results/: generated figures, data and logs

## Results and limitations

For Chebyquad, both BFGS implementations satisfy the gradient tolerance
for dimensions 4, 8 and 11. The final objective values are approximately
zero for dimensions 4 and 8, and 0.00279976 for dimension 11.

On Rosenbrock, the recorded BFGS inverse-Hessian approximation improves
globally, although its error does not decrease at every iteration.

Convergence is assessed using the gradient norm. It does not establish
global optimality. The methods are unconstrained and do not enforce
Chebyquad points to remain in [0, 1].

Good Broyden, Bad Broyden and SR1 do not necessarily produce descent
directions. A line search can reject such a direction.

main.py and run_experiments.py are reserved for a future command-line
interface and subprocess-based experiment launcher.

## Course material

The Chebyquad objective and gradient were supplied by the course:
Claus Führer, 2016. Their original attribution is retained.
The dot and linspace imports were moved from SciPy to NumPy
for compatibility.
