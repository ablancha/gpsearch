import numpy as np


class PRE:

    def __init__(self, tf, nsteps, u_init,
                 alp=0.01, ome=2*np.pi, lam=0.1, bet=0.1):
        self.alp = alp
        self.ome = ome
        self.lam = lam
        self.bet = bet
        self.tf = tf
        self.nsteps = nsteps
        self.u_init = u_init

    def RHS(self, u, t):
        x1, x2, x3 = u
        f1 = self.alp*x1 + self.ome*x2 + self.alp*x1**2 \
             + 2*self.ome*x1*x2 + x3**2
        f2 = -self.ome*x1 + self.alp*x2 - self.ome*x1**2 \
             + 2*self.alp*x1*x2
        f3 = -self.lam*x3 - (self.lam+self.bet)*x1*x3
        f = [f1, f2, f3]
        return f

    def solve(self):
        time = np.linspace(0, self.tf, self.nsteps+1)
        solver = ODESolver(self.RHS)
        solver.set_ics(self.u_init)
        u, t = solver.solve(time)
        return u, t


class ODESolver:

    def __init__(self, f):
        self.f = lambda u, t: np.asarray(f(u, t), float)

    def set_ics(self, U0):
        U0 = np.asarray(U0)
        self.neq = U0.size
        self.U0 = U0

    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        K1 = dt*f(u[k], t[k])
        K2 = dt*f(u[k] + 0.5*K1, t[k] + 0.5*dt)
        K3 = dt*f(u[k] + 0.5*K2, t[k] + 0.5*dt)
        K4 = dt*f(u[k] + K3, t[k] + dt)
        u_new = u[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new

    def solve(self, time):
        self.t = np.asarray(time)
        n = self.t.size
        self.u = np.zeros((n,self.neq))
        self.u[0] = self.U0
        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()
        return self.u[:k+2], self.t[:k+2]

