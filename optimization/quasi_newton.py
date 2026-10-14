import numpy as np

from .newton import NewtonMethod


class GoodBroydenMethod(NewtonMethod):
    def initialize_matrix(self, x):
        n=len(x)

        self.G=np.eye(n)
        self.H=np.eye(n)

    def compute_direction(self,x,gradient):
        return -self.H@gradient

    def update_matrix(self,x,x_new,gradient,gradient_new):
        delta=x_new-x
        gamma=gradient_new-gradient

        delta_squared=np.dot(delta, delta)

        if delta_squared==0:
            return

        H_gamma=self.H@gamma
        delta_H=delta@self.H

        denominator = np.dot(delta, H_gamma)
        scale = np.linalg.norm(delta)*np.linalg.norm(H_gamma)

    
        if abs(denominator) <= 1e-12*scale:
            return

        G_new=self.G+np.outer(gamma-self.G@delta,delta)/delta_squared

        H_new=self.H+np.outer(delta-H_gamma,delta_H)/denominator

        self.G=G_new
        self.H=H_new

class BadBroydenMethod(NewtonMethod):
    def initialize_matrix(self,x):
        self.H = np.eye(len(x))

    def compute_direction(self,x,gradient):
        return -self.H@gradient

    def update_matrix(self,x,x_new,gradient,gradient_new):
        delta=x_new-x
        gamma=gradient_new-gradient

        gamma_dot_gamma=np.dot(gamma, gamma)

        if gamma_dot_gamma==0:
            return

        H_gamma=self.H@gamma
        inverse_secant_error=delta-H_gamma

        correction=np.outer(inverse_secant_error,gamma)/gamma_dot_gamma

        self.H=self.H+correction 

class SymmetricBroydenMethod(NewtonMethod):
    def initialize_matrix(self,x):
        self.H = np.eye(len(x))

    def compute_direction(self,x,gradient):
        return -self.H @ gradient

    def update_matrix(self,x,x_new,gradient,gradient_new):
        delta=x_new-x
        gamma=gradient_new-gradient

        v=delta-self.H@gamma

        denominator=np.dot(v,gamma)
        scale=np.linalg.norm(v)*np.linalg.norm(gamma)

        if abs(denominator)<=1e-12*scale:
            return

        correction=np.outer(v, v)/denominator
        self.H=self.H+correction

class DFPMethod(NewtonMethod):
    def initialize_matrix(self, x):
        self.H = np.eye(len(x))

    def compute_direction(self,x,gradient):
        return -self.H@gradient

    def update_matrix(self,x,x_new,gradient,gradient_new):
        delta=x_new-x
        gamma=gradient_new-gradient

        H_gamma=self.H@gamma

        delta_dot_gamma=np.dot(delta,gamma)
        gamma_dot_H_gamma=np.dot(gamma,H_gamma)

        scale1=np.linalg.norm(delta)*np.linalg.norm(gamma)
        scale2=np.linalg.norm(gamma)*np.linalg.norm(H_gamma)

        if delta_dot_gamma<=1e-12*scale1:
            return

        if gamma_dot_H_gamma<=1e-12*scale2:
            return

        correction_delta=(np.outer(delta,delta)/delta_dot_gamma)

        correction_gradient=(np.outer(H_gamma, H_gamma)/gamma_dot_H_gamma)

        self.H=self.H+correction_delta-correction_gradient

class BFGSMethod(NewtonMethod):
    def initialize_matrix(self,x):
        self.H = np.eye(len(x))

    def compute_direction(self,x,gradient):
        return -self.H@gradient

    def update_matrix(self,x,x_new,gradient,gradient_new):
        delta=x_new-x
        gamma=gradient_new-gradient

        delta_dot_gamma=np.dot(delta,gamma)
        scale = np.linalg.norm(delta)*np.linalg.norm(gamma)

 
        if delta_dot_gamma<=1e-12*scale:
            return

        beta=1.0/delta_dot_gamma
        identity=np.eye(len(delta))

        left=identity-beta*np.outer(delta,gamma)
        right=identity-beta*np.outer(gamma,delta)

        H_new=(left@self.H@right+beta*np.outer(delta, delta))

        self.H=0.5*(H_new+H_new.T)
