import numpy as np 
from .base_method import OptimizationMethod

class NewtonMethod(OptimizationMethod):
      def compute_direction(self,x,gradient):
          hessian=self.problem.hessian(x)
          return np.linalg.solve(hessian,-gradient)
