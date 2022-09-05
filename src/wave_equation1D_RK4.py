import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from petsc4py import PETSc
from dolfinx import fem, mesh, io

# https://jorgensd.github.io/dolfinx-tutorial/chapter2/heat_equation.html
# https://fenicsproject.discourse.group/t/getting-inf-inf-inf-in-this-coupled-displacement-pressure-problem-implemented-with-mixedelements/9183
# https://docs.fenicsproject.org/dolfinx/v0.5.0/python/demos/demo_cahn-hilliard.html
# https://fenics-solid-tutorial.readthedocs.io/en/latest/2DPlaneStrain/2D_Elasticity.html
# https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html

rho = 1.2
c = 343
fmax = 1000
sigma0 = 0.2
ppw = 8

CFL = 0.1

# Define temporal parameters
t = 0 # Start time
T = 20.0/c # Final time
dx = c/fmax/ppw
dt = CFL*dx/c
num_steps = int(np.ceil(T/dt))

# Define mesh
xminmax = [-2,2]
nx = int(np.ceil((xminmax[1]-xminmax[0])/dx))
domain = mesh.create_interval(MPI.COMM_WORLD, nx, xminmax)

xdmf = io.XDMFFile(domain.comm, "wave_equation.xdmf", "w")
xdmf.write_mesh(domain)

P_v = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
P_p = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
element = ufl.MixedElement([P_v, P_p])
ME = fem.FunctionSpace(domain, element)

def initial_condition1D(x, x0=0.0):
    return np.exp(-((x[0]-x0)/sigma0)**2)

# Variational problem
du = ufl.TrialFunction(ME)
u = fem.Function(ME)
u.x.array[:] = 0
v,p = u.split()
p.name, v.name = "pressure", "velocity"
p.interpolate(initial_condition1D)

phi = ufl.TestFunction(ME)
phi_v,phi_p = ufl.split(phi)

xdmf.write_function(p, t)
xdmf.write_function(v, t)

# du is change for a timestep defined as the mixed element with dv = du[0] and dp = du[1]
a_p = du[1]*phi_p * ufl.dx # p_t
a_v = du[0]*phi_v * ufl.dx # v_t
a_ = a_v + a_p

# u is the field defined as the mixed element with v = u[0] and p = u[1]
L_p = 1/rho * ufl.grad(u.sub(1))[0] * phi_p * ufl.dx     # change in velocity p_t
L_v = -rho*c**2 * u.sub(0) * ufl.grad(phi_v)[0] * ufl.dx # change in pressure v_t
L_ = L_p + L_v
  
k2, k3, k4 = ufl.TrialFunction(ME), ufl.TrialFunction(ME), ufl.TrialFunction(ME)
u1, u2, u3 = fem.Function(ME), fem.Function(ME), fem.Function(ME)
  
a_k1 = fem.form(a_)
a_k2 = fem.form(ufl.replace(a_, {du: k2}))
a_k3 = fem.form(ufl.replace(a_, {du: k3}))
a_k4 = fem.form(ufl.replace(a_, {du: k4}))
  
L_k1 = fem.form(L_)
L_k2 = fem.form(ufl.replace(L_, {u: u1}))
L_k3 = fem.form(ufl.replace(L_, {u: u2}))
L_k4 = fem.form(ufl.replace(L_, {u: u3}))
  
A1 = fem.petsc.assemble_matrix(a_k1); A1.assemble()
A2 = fem.petsc.assemble_matrix(a_k2); A2.assemble()
A3 = fem.petsc.assemble_matrix(a_k3); A3.assemble()
A4 = fem.petsc.assemble_matrix(a_k4); A4.assemble()
  
b_k1 = fem.petsc.create_vector(L_k1)
b_k2 = fem.petsc.create_vector(L_k2)
b_k3 = fem.petsc.create_vector(L_k3)
b_k4 = fem.petsc.create_vector(L_k4)
  
# SOLVER - Using petsc4py to create a linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
  
du = fem.Function(ME) # displacement container for holding temp solutions
  
# Updating the solution and right hand side per time step
for i in range(num_steps):
    t += dt
    
    # Solve for k1    
    with b_k1.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b_k1, L_k1) # Update the right hand side reusing the initial vector
    solver.setOperators(A1); solver.solve(b_k1, du.vector) # Solve linear problem
    k1 = du.x.array.copy()

    u1.x.array[:] = (u.x.array + dt/2*du.x.array).copy() # u1 used for calculating k2
    with b_k2.localForm() as loc_b:
        loc_b.set(0)    
    fem.petsc.assemble_vector(b_k2, L_k2)
    solver.setOperators(A2); solver.solve(b_k2, du.vector)
    k2 = du.x.array.copy()

    u2.x.array[:] = (u.x.array + dt/2*du.x.array).copy() # u2 used for calculating k3
    with b_k3.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b_k3, L_k3)
    solver.setOperators(A3); solver.solve(b_k3, du.vector)
    k3 = du.x.array.copy()

    u3.x.array[:] = (u.x.array + dt*du.x.array).copy() # u3 used for calculating k4
    with b_k4.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b_k4, L_k4)
    solver.setOperators(A4); solver.solve(b_k4, du.vector)    
    k4 = du.x.array.copy()
  
    # Update u for time n+1
    u.x.array[:] = (u.x.array + dt/6*(k1 + 2*k2 + 2*k3 + k4)).copy()

    if i % 25 == 0:
        # Write solutions to file 
        xdmf.write_function(p, t) # p and v are pointers to u
        xdmf.write_function(v, t) # p and v are pointers to u

xdmf.close()

