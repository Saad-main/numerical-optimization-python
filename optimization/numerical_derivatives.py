import numpy as np

def numerical_gradient(objective,x,h=1e-5):
    x=np.asarray(x,dtype=float)
    if x.ndim != 1:
       raise ValueError("x doit etre un vecteur a une seule dimesion")
    if h<0:
       raise ValueError("h doit etre strictement positif")
    gradient=np.zeros_like(x)
    for i in range(len(x)):
        x_plus=x.copy()
        x_minus=x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        gradient[i]= (objective(x_plus)-objective(x_minus))/(2*h)
    return gradient

def numerical_hessian(gradient,x,h=1e-4):
    x=np.asarray(x,dtype=float)
    if x.ndim != 1 :
       raise ValueError("x doit etre une seule dimension")
    if h<=0:
       raise ValueError("h doit etre strictement positif")
    n=len(x)
    hessian=np.zeros((n,n))
    for j in range(n):
        x_plus=x.copy()
        x_minus=x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        g_plus = np.asarray(gradient(x_plus),dtype=float)
        g_minus = np.asarray(gradient(x_minus),dtype=float)
        hessian[:,j]=(g_plus-g_minus)/(2*h)
    return 0.5*(hessian+hessian.T)
