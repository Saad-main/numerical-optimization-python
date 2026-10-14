from .numerical_derivatives import (numerical_gradient,numerical_hessian)

class OptimizationProblem:
    def __init__(self,objective,gradient=None):
        self.objective= objective
        self.gradient= gradient
    def value(self,x):
        return self.objective(x)
    def grad(self,x):
        if self.gradient is None :
           return numerical_gradient(self.objective,x)
        return self.gradient(x)
    def hessian(self,x):
        return numerical_hessian(self.grad,x)
