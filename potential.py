import numpy as np

class Potential():

    def energy_and_forces(self, deltaR, k, D, rEq):
        v = self.morse(deltaR, k, D, rEq)
        f = self.forces(deltaR, k, D, rEq)
        return v, f

    def morse(self, deltaR, k, D, rEq):
        #Using a morse potential, given one distance and its constants, computes the energy
        a=np.sqrt(k/(2.0*D))
        v = 1.0 - np.exp(-a*(deltaR - rEq))
        v = D * (v**2)
        v = v - D
        return v

    def forces(self, deltaR, k, D, rEq):
        #compute the forces by aproximating the derivative with the slope of two points.
        deltaTotal=0.0001
        v1 = self.morse(deltaR+deltaTotal, k, D, rEq)
        v2 = self.morse(deltaR-deltaTotal, k, D, rEq)
        f1d=(v2-v1) / ((deltaR-deltaTotal)-(deltaR+deltaTotal))
        f = np.array([[0.,0.,f1d],[0.,0.,-f1d]])
        return f



    #def morse(self, deltaR, k, D, rEq):
    #    a=np.sqrt(k/(2.0*D))
    #    v = 1.0 - np.exp(-a*(deltaR - rEq))
    #    v = D * (v**2)
    #    v = v - D
    #    forces=1000
    #    return v, forces
