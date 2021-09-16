### 3D finite viscoelastic model for stress relaxation
from dolfin import *
import matplotlib.pyplot as plt
import os
import dolfin as dlf
import numpy as np
import math

### Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

### traction
class Traction(UserExpression):
    def __init__(self):
        super().__init__(self)
        self.t = 0.0
    def eval(self, values, x):
        values[0] = 0*self.t
        values[1] = 0.0
        values[2] = 0.0
    def value_shape(self):
        return (3,)

### mesh generation
def geometry_3d():
    mesh = UnitCubeMesh(6, 6, 6)
    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    x0 = AutoSubDomain(lambda x: near(x[0], 0))
    x1 = AutoSubDomain(lambda x: near(x[0], 1))
    y0 = AutoSubDomain(lambda x: near(x[1], 0))
    z0 = AutoSubDomain(lambda x: near(x[2], 0))
    x0.mark(boundary_parts, 1)
    y0.mark(boundary_parts, 2)
    z0.mark(boundary_parts, 3)
    x1.mark(boundary_parts, 4)
    return boundary_parts

### stress calculation
def CalculatePKStress(u,pressure,Be,ce,coe):
    I = Identity(V.mesh().geometry().dim())  # Identity tensor
    F = I + grad(u)          # Deformation gradient
    invF = inv(F)
    J = det(F)
    B = F*F.T
    S = 2*ce*B+2*coe*Be-pressure*I # Cauchy stress tensor
    T = J*invF*S*invF.T            # 2nd PK stress tensor
    return T, (J-1)


### Gradient of Be
def CalcGradBe(u,u_old,Be_old,dt,coe,eta):
    I = Identity(V.mesh().geometry().dim())
    F_old = I + grad(u_old)
    F = I + grad(u)
    L = ((F-F_old)/Constant(dt))*inv(F)
    slope =L*Be_old+Be_old*L.T-(2.0/eta)*Be_old*2*coe*dev(Be_old)
    return slope


### Create mesh and define function space
facet_function = geometry_3d()
mesh = facet_function.mesh()
gdim = mesh.geometry().dim()
dx = Measure("dx")
ds = Measure("ds", subdomain_data=facet_function, subdomain_id=4)
print('Number of nodes: ',mesh.num_vertices())
print('Number of cells: ',mesh.num_cells())
dx = dx(degree=4)
ds = ds(degree=4)


### Time stepping parameters
Tload = 1
Tend = 5
dt = 0.1
Nsteps = int(Tend/dt)
time = np.linspace(0, Tend, Nsteps+1)
eps_max = 0.01 # max applied strain during stress relaxation
eps_dot = eps_max/Tload  # strain rate to be applied during the loading phase

### material parameters
ce, coe, eta = 2, 2, 13.34 # 3D viscoelastic model parameter
E1, E2, eta_sls = 12, 12, 20 # Equivalent SLS model parameters
tau_sigma, tau_eps = (eta_sls/E1)*(1+E1/E2), eta_sls/E2 # SLS model time constants


### Create function space
element_s = FiniteElement("CG", mesh.ufl_cell(), 1)
element_v = VectorElement("CG", mesh.ufl_cell(), 1)
element_t = TensorElement("DG", mesh.ufl_cell(), 0)
mixed_element = MixedElement([element_v, element_s,element_t])
V = FunctionSpace(mesh, mixed_element)


### Define test and trial functions
dupbe = TrialFunction(V)
_u, _p, _Be = TestFunctions(V)

_u_p_be = Function(V)
u, p, Be = split(_u_p_be)
_u_p_be_old = Function(V)
u_old, p_old, Be_old = split(_u_p_be_old)


### initialize variables
# u_initial= interpolate(Constant((0.0,0.0,0.0)), V.sub(0).collapse())
# p_initial= interpolate(Constant(2*(ce+coe)), V.sub(1).collapse())
# Be_initial= interpolate(Constant(((1.0,0.0, 0.0),(0.0,1.0, 0.0),(0.0,0.0,1.0))),V.sub(2).collapse())
# assign(_u_p_be, [u_initial,p_initial,Be_initial])
# assign(_u_p_be_old, [u_initial,p_initial,Be_initial])

I = Identity(V.mesh().geometry().dim())
WF = TensorFunctionSpace(mesh, "DG", degree=0)
Be_initial = project(I, WF)
assign(_u_p_be.sub(2),Be_initial)
assign(_u_p_be_old.sub(2),Be_initial)

### Create tensor function spaces for stress and stretch for result plotting
defGrad = Function(WF, name='F')
PK_stress = Function(WF, name='PK1')

### initialize traction class
h = Traction() # Traction force on the boundary

### Define Dirichlet boundary
bc0 = DirichletBC(V.sub(0).sub(0), Constant(0.), facet_function, 1)
bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), facet_function, 2)
bc2 = DirichletBC(V.sub(0).sub(2), Constant(0.), facet_function, 3)
tDirBC = Expression(('eps_d*time_'),eps_d = eps_dot, time_=0.0 , degree=0)
bc3 = DirichletBC(V.sub(0).sub(0), tDirBC, facet_function, 4)
bcs = [bc0,bc1,bc2,bc3]


### weak forms
I = Identity(V.mesh().geometry().dim())
F_cur = I + grad(u)
pkstrs, hydpress =  CalculatePKStress(u,p,Be,ce,coe)
F1 = inner(dot(F_cur,pkstrs), grad(_u))*dx - dot(h, _u)*ds  # PK2 weak form
F2 = hydpress*_p*dx                                         # hydrostatic pressure weak form
F3 = inner((Be - Be_old),_Be)*dx \
        - Constant(dt)*inner(CalcGradBe(u,u_old,Be_old,dt,coe,eta),_Be)*dx  # Evolution equation weak form
F = F1+F2+F3
J = derivative(F, _u_p_be,dupbe)


### Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, _u_p_be, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'


### Save solution in VTK format
file_results = XDMFFile("./Results/TestUniaxialLoading/Uniaxial.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

stretch_vec = np.zeros((Nsteps+1,1))
stress_vec =  np.zeros((Nsteps+1,1))
stress_ana =  np.zeros((Nsteps+1,1))

for i in range(len(time)):
    tt = time[i]
    if (i%10)==0:
        print(i, 'time: ', tt)

    ### update time-dependent variables
    h.t = tt
    tDirBC.time_ = tt if tt<=Tload else Tload # stress relaxation

    ### solve
    solver.solve()

    ### Extract solution components
    u_print, p_print, Be_print = _u_p_be.split()
    u_print.rename("u", "displacement")
    p_print.rename("p", "pressure")
    Be_print.rename("Be", "Be")

    ### update old variables
    _u_p_be_old.assign(_u_p_be)


    ### Save DefGrad to file
    point = (0.5,0.5,0.5) # point at which Defgrad is required
    DF = I + grad(u_print)
    defGrad.assign(project(DF, WF))
    stretch_vec[i] = defGrad(point)[0]

    # Save Stress to file
    PK_s,tempPress = CalculatePKStress(u_print,p_print,Be_print,ce,coe)
    PK_1 = DF*PK_s # this is 1st PK stress
    PK_stress.assign(project(PK_1, WF))
    stress_vec[i] = PK_stress(point)[0]

    # save xdmf file
    file_results.write(u_print, tt)
    file_results.write(defGrad, tt)
    file_results.write(PK_stress, tt)


### analytical solution for SLS model (Stress relaxation)
for i in range(len(time)):
    if time[i] <= Tload:
        stress_ana[i] = E1*eps_dot*(tau_sigma-tau_eps)*(1-np.exp(-time[i]/tau_eps)) \
                            +E1*(stretch_vec[i]-1)
    else:
        eps0 = eps_dot*Tload
        sigma_t1 = E1*eps_dot*(tau_sigma-tau_eps)*(1-np.exp(-Tload/tau_eps)) \
                            +E1*((1+eps0)-1)
        stress_ana[i] = (sigma_t1-E1*eps0)*np.exp(Tload/tau_eps)*np.exp(-time[i]/tau_eps)\
                            +E1*eps0

### plot results
f = plt.figure(figsize=(12,6))
plt.plot(time, stress_vec,'r-',label='Fenics')
plt.plot(time, stress_ana,'k.',label='Analytical')
plt.xlabel('time')
plt.ylabel('stress')
plt.legend()
plt.show()
