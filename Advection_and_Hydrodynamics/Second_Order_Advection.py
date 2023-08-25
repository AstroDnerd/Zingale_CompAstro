import numpy as np
import matplotlib.pyplot as plt

class FiniteDifferenceGrid_1D:
	''' A 1D Finite Difference Grid with nx zones, ng ghost zones and lying between [xmin,xmax]'''

	def __init__(self, nx, ng = 1, xmin = 0, xmax = 1):
		self.nx = nx
		self.ng = ng
		self.xmin = xmin
		self.xmax = xmax

		#Lower and Upper indices for start and end of grid minus the ghost zones
		self.LowerIndex = ng
		self.UpperIndex = ng+nx-1

		#Physical locations of the grid points
		self.dx = (xmax-xmin)/(nx-1)
		self.x = xmin + (np.arange(nx+2*ng)-ng)*self.dx

		#values in the grid points
		self.ainit = np.zeros(nx+2*ng, dtype = np.float64)
		self.a = np.zeros(nx+2*ng, dtype = np.float64)

	def EmptyGridArray(self):
		return np.zeros(nx+2*self.ng, dtype = np.float64)

	def SetPeriodicBoundaryConditions(self):
		self.a[self.LowerIndex-1] = self.a[self.UpperIndex-1]
		self.a[self.UpperIndex+1] = self.a[self.LowerIndex+1]


	def Plot(self):
		fig = plt.figure()
		plt.plot(self.x[self.LowerIndex:self.UpperIndex+1], self.ainit[self.LowerIndex:self.UpperIndex+1], label = "Initial Conditions")
		plt.plot(self.x[self.LowerIndex:self.UpperIndex+1], self.a[self.LowerIndex:self.UpperIndex+1], label = "FTCS Solution")
		plt.legend()
		return fig


def FTCSadvection(nx, u, C, ng = 1, xmin = 0, xmax = 1, num_timeperiods = 1, init_condition = None):
	# takes the inputs needed to construct a grid element and the drid timestep parameters
	FTCS_grid = FiniteDifferenceGrid_1D(nx, xmin = xmin, xmax = xmax)

	#time related parameters
	dt = C*FTCS_grid.dx/u
	t = 0
	tmax = num_timeperiods*(xmax-xmin)/u

	#initial condition
	init_condition(FTCS_grid)

	FTCS_grid.ainit[:] = FTCS_grid.a[:]

	#do the loop
	updated_grid = FTCS_grid.EmptyGridArray()

	while t<tmax:
		if t+dt >tmax:
			dt = tmax-t
			C = u*dt/FTCS_grid.dx

		FTCS_grid.SetPeriodicBoundaryConditions()

		for i in range(FTCS_grid.LowerIndex,FTCS_grid.UpperIndex+1):
			updated_grid[i] = FTCS_grid.a[i] - (C/2)*(FTCS_grid.a[i+1]-FTCS_grid.a[i-1])

		FTCS_grid.a[:] = updated_grid[:]

		t+=dt

	return FTCS_grid


def Upwindadvection(nx, u, C, ng = 1, xmin = 0, xmax = 1, num_timeperiods = 1, init_condition = None):
	# takes the inputs needed to construct a grid element and the drid timestep parameters
	Upwind_grid = FiniteDifferenceGrid_1D(nx, xmin = xmin, xmax = xmax)

	#time related parameters
	dt = C*Upwind_grid.dx/u
	t = 0
	tmax = num_timeperiods*(xmax-xmin)/u

	#initial condition
	init_condition(Upwind_grid)

	Upwind_grid.ainit[:] = Upwind_grid.a[:]

	#do the loop
	updated_grid = Upwind_grid.EmptyGridArray()

	while t<tmax:
		if t+dt >tmax:
			dt = tmax-t
			C = u*dt/Upwind_grid.dx

		Upwind_grid.SetPeriodicBoundaryConditions()

		for i in range(Upwind_grid.LowerIndex,Upwind_grid.UpperIndex+1):
			updated_grid[i] = Upwind_grid.a[i] - C*(Upwind_grid.a[i]-Upwind_grid.a[i-1])

		Upwind_grid.a[:] = updated_grid[:]

		t+=dt

	return Upwind_grid

#initial Condition
def tophat(g):
    g.a[:] = 0.0
    g.a[np.logical_and(g.x >= 1./3, g.x <= 2./3.)] = 1.0

nx = 128
u = 1.0
C = 0.5

g = Upwindadvection(nx, u, C, init_condition=tophat, num_timeperiods = 1)
fig = g.Plot()
plt.savefig('Upwind_tophat_nx128_u1_Cpoint5_numperiods1.png')

