import numpy as np
from pyscf import gto, dft, scf, grad
import ehrenfest_force
from rt_scf import RT_SCF
from rt_nuclei import NUC

'''
Real-time SCF + Ehrenfest
'''

class RT_Ehrenfest(RT_SCF):
    def __init__(self, mf, timestep, max_time, filename=None, prop=None, frequency=1, orth=None, chkfile=None, verbose=3, Ne_step=10, N_step=10):

        super().__init__(mf, timestep, max_time, filename, prop, frequency, orth, chkfile, verbose)

        self.Ne_step = Ne_step
        self.N_step = N_step
        self.filename = filename

        self.den_ao = self._scf.make_rdm1(mo_coeff = self._scf.mo_coeff)
        if self.den_ao.dtype != np.complex128:
            self.den_ao = self.den_ao.astype(np.complex128)
        self.nuc = NUC(self._scf.mol)

        self.current_time = 0

        self.update_mol()
        # Reminder to check if forces should be updated again after excite()
        self.nuc.force = ehrenfest_force.get_force(self)

    def update_time(self):
        current_N = np.mod(int(self.current_time / self.timestep), self.Ne_step * self.N_step)
        current_Ne = np.mod(current_N, self.Ne_step)
        if current_N == 0: # k = 0, j = 0
            self.nuc.get_vel(0.5 * self.N_step * self.Ne_step * self.timestep)
            self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()
        elif current_Ne == 0: # k = 0, j != 0
            self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()
        elif current_N == self.N_step * self.Ne_step - 1: # k = Ne_step-1, j = N_step-1
            self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()
            self.nuc.get_vel(0.5 * self.N_step * self.Ne_step * self.timestep)
        elif current_Ne == self.Ne_step - 1: # k = Ne_step-1, j != N_step-1
            self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()
        if np.mod(int(self.current_time / self.timestep), self.N_step * self.Ne_step) == 0:
            self.nuc.get_vel(-0.5 * self.N_step * self.Ne_step * self.timestep)
            self.nuc.force = ehrenfest_force.get_force(self)
            self.nuc.get_vel(-0.5 * self.N_step * self.Ne_step * self.timestep)
        
        self.current_time += self.timestep

    def update_mol(self):
        mo_coeff = self._scf.mo_coeff
        if self._scf.istype('RKS'): xc = self._scf.xc; self._scf = scf.RKS(self.nuc.get_mol()); self._scf.xc = xc
        elif self._scf.istype('RHF'): self._scf = scf.RHF(self.nuc.get_mol())
        elif self._scf.istype('UKS'): xc = self._scf.xc; self._scf = dft.UKS(self.nuc.get_mol()); self._scf.xc = xc
        elif self._scf.istype('UHF'): self._scf = scf.UHF(self.nuc.get_mol())
        elif self._scf.istype('GKS'): xc = self._scf.xc; self._scf = scf.GKS(self.nuc.get_mol()); self._scf.xc = xc
        elif self._scf.istype('GHF'): self._scf = scf.GHF(self.nuc.get_mol())
        self._scf.mo_coeff = mo_coeff
        self._scf.mo_occ = self.occ
        self.ovlp = self._scf.get_ovlp()
        self.evals, self.evecs = np.linalg.eigh(self.ovlp)
        #self.orth = np.matmul(self.evecs, np.diag(np.power(self.evals, -0.5)))
        self.orth = self._get_orth(self.ovlp) #np.linalg.multi_dot([self.evecs, np.diag(np.power(self.evals, -0.5)), self.evecs.T])
        #self.orth = scf.addons.canonical_orth_(self.ovlp)

    def update_grad(self):
        if self._scf.istype('RKS'): self._grad = self._scf.apply(grad.RKS)
        elif self._scf.istype('RHF'): self._grad = self._scf.apply(grad.RHF)
        elif self._scf.istype('UKS'): self._grad = self._scf.apply(grad.UKS)
        elif self._scf.istype('UHF'): self._grad = self._scf.apply(grad.UHF)
        elif self._scf.istype('GKS'): self._grad = self._scf.apply(grad.GKS)
        elif self._scf.istype('GHF'): self._grad = self._scf.apply(grad.GHF)

    # Additinal term arising from the moving nuclei in the classical path approximation to be added to the fock matrix
    # Reminder to turn the complex conserving potential term Omega into one of the rtscf potential classes
    def get_omega(self):
        mol = self._scf.mol
        Rdot = self.nuc.vel
        X = self.orth
        Xinv = inv(X)
        dS = -mol.intor('int1e_ipovlp', comp=3)
    
        Omega = np.zeros(X.shape, dtype = complex)
        RdSX = np.zeros(X.shape)
        aoslices = mol.aoslice_by_atom()
        for i in range(mol.natm):
            p0, p1 = aoslices[i,2:]
            RdSX += np.einsum('x,xij,ik->jk', Rdot[i], dS[:,p0:p1,:], X[p0:p1,:])
        Omega += np.matmul(RdSX, Xinv)
        return Omega


