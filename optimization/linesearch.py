import numpy as np 
class ExactLineSearch:
      def __init__(self,tol=1e-8,max_iter=200):
          self.tol=tol
          self.max_iter=max_iter
      def search(self,problem,x,direction):
          def phi(alpha):
              return problem.value(x+alpha*direction)
          slope=np.dot(problem.grad(x),direction)
          if slope>=0:
             raise ValueError("la direction n est pas une direction de descente")
          a=0.0
          b=1.0
          fa=phi(a)
          fb=phi(b)
          for _ in range(self.max_iter):
              if fb<fa:
                 break
              b*=0.5
              fb=phi(b)
          else:
              raise RunTimeError("Impossible detrouver un pas decroissant")
          c=2.0* b
          fc=phi(c)
          for _ in range(self.max_iter):
              if fc> fb:
                 break
              a,b = b,c
              fb=fc
              c*=2.0
              fc=phi(c)
          else:
              raise RunTimeError("Impossible d encadrer un minimum")
          ratio= (np.sqrt(5.0)-1.0)/2.0
          left =a
          right=c
          u=right - ratio*(right-left)
          v= left+ ratio*(right-left)
          fu=phi(u)
          fv=phi(v)
          for _ in range(self.max_iter):
              if right-left<=self.tol:
                 return 0.5*(left+right)
              if fu<fv:
                 right=v
                 v,fv= u,fu
                 u=right -ratio*(right-left)
                 fu=phi(u)
              else:
                  left=u
                  u,fu= v,fv
                  v=left+ratio*(right-left)
                  fv=phi(v)
          raise RunTimeError("La recherche n a pas atteint la tolerance")

class InexactLineSearch:
    def __init__(self,rho=0.01,sigma=0.1,tau1=9.0,tau2=0.1,tau3=0.5,alpha1=0.1,f_lower=0.0,max_iter=200,):
        if not 0 <rho<sigma<1:
            raise ValueError("Il faut 0 < rho < sigma < 1.")

        if tau1 <= 1:
            raise ValueError("tau1 doit être supérieur à 1.")

        if not 0 <tau2<tau3<= 0.5:
            raise ValueError("Il faut 0 < tau2 < tau3 <= 0.5.")

        if alpha1<=0 or max_iter< 1:
            raise ValueError("alpha1 et max_iter doivent être positifs.")

        self.rho=rho
        self.sigma=sigma
        self.tau1=tau1
        self.tau2=tau2
        self.tau3=tau3
        self.alpha1=alpha1
        self.f_lower=f_lower
        self.max_iter=max_iter

    def search(self,problem,x,direction):
        def phi(alpha):
            return problem.value(x+alpha*direction)

        def dphi(alpha):
            return np.dot(problem.grad(x + alpha * direction),direction,)

        phi0 = phi(0.0)
        slope0 = dphi(0.0)

        if slope0 >= 0:
            raise ValueError("La direction doit être une direction de descente.")

        if phi0<=self.f_lower:
            return 0.0

        mu=(self.f_lower-phi0)/(self.rho*slope0)

        alpha_previous = 0.0
        phi_previous = phi0
        alpha = min(self.alpha1, mu)

        # Bracketing
        for _ in range(self.max_iter):
            phi_alpha = phi(alpha)

            if phi_alpha<=self.f_lower:
                return alpha

            if (phi_alpha > phi0 + self.rho * alpha * slope0 or phi_alpha >= phi_previous):
                a=alpha_previous
                b=alpha
                break

            slope_alpha=dphi(alpha)

            if abs(slope_alpha)<=-self.sigma*slope0:
                return alpha

            if slope_alpha>= 0:
                a=alpha
                b=alpha_previous
                break

            lower=2.0*alpha-alpha_previous

            if mu<=lower:
                alpha_next=mu
            else:
                upper=min(mu,alpha+self.tau1*(alpha - alpha_previous),)
                alpha_next = 0.5 * (lower + upper)

            alpha_previous = alpha
            phi_previous = phi_alpha
            alpha = alpha_next
        else:
            raise RuntimeError("Le bracketing n'a pas abouti.")

        phi_a=phi(a)

        # Sectionning
        for _ in range(self.max_iter):
            endpoint1=a+self.tau2*(b - a)
            endpoint2=b-self.tau3*(b - a)
            alpha=0.5*(endpoint1+endpoint2)

            if alpha == a or alpha == b:
               raise RuntimeError("L'intervalle ne peut plus être réduit.")

            phi_alpha=phi(alpha)

            if phi_alpha<=self.f_lower:
                return alpha

            if (phi_alpha > phi0 + self.rho*alpha*slope0 or phi_alpha>=phi_a):
                b=alpha
            else:
                slope_alpha=dphi(alpha)

                if abs(slope_alpha)<=-self.sigma*slope0:
                    return alpha

                if (b-a)*slope_alpha>=0:
                    b=a

                a=alpha
                phi_a=phi_alpha

        raise RuntimeError("Le sectioning n'a pas abouti.")
