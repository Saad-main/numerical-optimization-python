import numpy as np 

class OptimizationMethod:
      def __init__(self,problem,tol=1e-6,max_iter=100,line_search=None):
          if tol<=0:
             raise ValueError("tol doit etre strictement positif.")
          if max_iter<1:
             raise ValueError("max_iter doit etre superieur ou egal a 1")
          self.problem=problem
          self.tol=tol
          self.max_iter=max_iter
          self.line_search=line_search
      def initialize_matrix(self,x):
          pass
      def compute_direction(self,x,gradient):
          raise NotImplementedError("chaque methode doit definir sa methode")
      def compute_step(self,x,direction):
          if self.line_search is None:
             return 1.0
          return self.line_search.search(self.problem,x,direction)
      def update_matrix(self,x,x_new,gradient,gradient_new):
          pass
      def minimize(self,x0):
          x=np.array(x0,dtype=float,copy=True)
          if x.ndim !=1 or x.size==0:
             raise ValueError("x0 doit etre un vecteur non vide")
          self.history=[x.copy()]
          self.converged= False
          self.n_iter=0
          gradient=np.asarray(self.problem.grad(x),dtype=float)
          if np.linalg.norm(gradient) <=self.tol :
             self.converged= True
             return x
          self.initialize_matrix(x)
          for k in range(self.max_iter):
              direction = self.compute_direction(x,gradient)
              alpha= self.compute_step(x,direction)
              x_new= x+alpha*direction
              gradient_new= np.asarray(self.problem.grad(x_new),dtype=float)
              self.history.append(x_new.copy())
              self.n_iter=k+1
              if np.linalg.norm(gradient_new)<=self.tol:
                 self.converged=True
                 x=x_new
                 break
              self.update_matrix(x,x_new,gradient,gradient_new)
              x=x_new
              gradient=gradient_new
          return x  
