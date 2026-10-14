import numpy as np

def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 +(1-x[0])**2
def rosenbrock_gradient(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)+2*(x[0]-1),200*(x[1]-x[0]**2)])

