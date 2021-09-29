# Import modules
from __future__ import division
from dolfin import *

import argparse
import math
import os
import shutil
import sympy
import sys

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Parameters for DOLFIN and SOLVER
# ----------------------------------------------------------------------------
set_log_level(LogLevel.INFO)  # log level
# set_log_level(LogLevel.WARNING)
# set some dolfin specific parameters
info(parameters, False)
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"]=4
# parameters["mesh_partitioner"] = "SCOTCH"
# parameters["allow_extrapolation"] = True

# -----------------------------------------------------------------------------
# parameters of the solvers
# snes_solver_parameters = {"nonlinear_solver": "snes",
#                           "snes_solver": {"linear_solver": "lu",
#                                           'absolute_tolerance':1e-8,
#                                           'relative_tolerance':1e-8,
#                                           "maximum_iterations": 20,
#                                           "report": True,
#                                           "error_on_nonconvergence": True}}

# parameters of the solvers
snes_solver_parameters   = {"nonlinear_solver": "snes",
                         "snes_solver": {"linear_solver": "mumps",
                                         "method" : "newtontr",
                                         "line_search": "cp",
                                         "preconditioner" : "hypre_amg",
                                         "maximum_iterations": 200,
                                         "absolute_tolerance": 1e-8,
                                         "relative_tolerance": 1e-7,
                                         "solution_tolerance": 1e-7,
                                         "report": True,
                                         "error_on_nonconvergence": False}}

# -----------------------------------------------------------------------------
# # set the user parameters
# parameters.parse()
# userpar = Parameters("user")
# # this section of code may be modified as needed for the new deformation regime
# userpar.add("t_initial", 0)
# userpar.add("t_final", 5)
# userpar.add("delta_t", 0.1)
# userpar.add("u_min",0.)
# userpar.add("u_max",0.1)
# userpar.parse()

# deformation parameters
t_A = 0
t_B = 5
t_C = 10
dt = 0.05

numsteps_A2B = int((t_B-t_A)/dt)
numsteps_B2C = int((t_C-t_B)/dt)+1

u_min = 0.
u_max = 1

time_A2B = np.linspace(t_A, t_B, numsteps_A2B, endpoint=False)
time_B2C = np.linspace(t_B, t_C, numsteps_B2C, endpoint=True)
time = np.concatenate((time_A2B, time_B2C))

u_A2B = np.linspace(u_min, u_max, numsteps_A2B, endpoint=False)
u_B2C = np.linspace(u_max, u_min, numsteps_B2C, endpoint=True)
u_list = np.concatenate((u_A2B, u_B2C))
lambda_list = 1+u_list # for a reference unit square domain

# Material parameters
nu = Constant(100.0) # number of Kuhn segments in the chains for the network
kappa = Constant(100.0) # normalized bulk modulus for penalty formulation
G = Constant(1.0) # normalized shear modulus for penalty formulation
k_d = Constant(0.007) # 0.2 # abs(L) = 0.02
k_a = k_d
C_tot = Constant(2.0)
C_att_0 = (k_a/(k_a+k_d))*C_tot # C_tot
C_det_0 = C_tot-C_att_0


# # Define mesh and mixed function space
# nx, ny, nz = 5, 5, 5
# nx_string = "%s" % nx
# ny_string = "%s" % ny
# nz_string = "%s" % nz

# modelname = "tnt-model-penalty-incomp-weak-form"
# meshtype = "UnitCubeMesh"
# meshname = modelname+"-"+meshtype+"-"+nx_string+"-"+ny_string+"-"+nz_string+".xdmf"
# savedir = modelname+"/"+meshtype+"-"+nx_string+"-"+ny_string+"-"+nz_string+"/"

# mesh = UnitCubeMesh(nx, ny, nz)
# geo_mesh = XDMFFile(MPI.comm_world, savedir+meshname)
# geo_mesh.write(mesh)
# mesh.init()

# center_point = (0.5,0.5,0.5)

# Geometry paramaters
L, H, N = 1.0, 1.0, 30
L_string = "%.1f" % L
H_string = "%.1f" % H
N_string = "%s" % N
hsize = float(L/N)

modelname = "tnt-model-penalty-incomp-weak-form"
meshtype = "RectangleMesh"
meshname = meshtype+"-"+L_string+"-"+H_string+"-"+N_string+".xdmf"
savedir = modelname+"/"+meshtype+"-"+L_string+"-"+H_string+"-"+N_string+"/"

mesh = RectangleMesh(Point(0., 0.), Point(L, H), int(N), int(float(H/hsize)), "right/left")
geo_mesh = XDMFFile(MPI.comm_world, savedir+meshname)
geo_mesh.write(mesh)
mesh.init()
ndim = mesh.geometry().dim()  # get number of space dimensions
if MPI.rank(MPI.comm_world) == 0:
    print ("the dimension of mesh: {0:2d}".format(ndim))

center_point = (float(L/2),float(H/2))

# Arrays to store state parameters during deformation
sigma_G_11_center_point_array = np.zeros((numsteps_A2B+numsteps_B2C,1))
F_11_center_point_array       = np.zeros((numsteps_A2B+numsteps_B2C,1))
L_prior_11_center_point_array = np.zeros((numsteps_A2B+numsteps_B2C,1))
mu_11_center_point_array      = np.zeros((numsteps_A2B+numsteps_B2C,1))

sigma_G_22_center_point_array = np.zeros((numsteps_A2B+numsteps_B2C,1))
F_22_center_point_array       = np.zeros((numsteps_A2B+numsteps_B2C,1))
L_prior_22_center_point_array = np.zeros((numsteps_A2B+numsteps_B2C,1))
mu_22_center_point_array      = np.zeros((numsteps_A2B+numsteps_B2C,1))

# sigma_G_33_center_point_array = np.zeros((numsteps_A2B+numsteps_B2C,1))
# F_33_center_point_array       = np.zeros((numsteps_A2B+numsteps_B2C,1))
# L_prior_33_center_point_array = np.zeros((numsteps_A2B+numsteps_B2C,1))
# mu_33_center_point_array      = np.zeros((numsteps_A2B+numsteps_B2C,1))

Psi_center_point_array = np.zeros((numsteps_A2B+numsteps_B2C,1))

V_u =  VectorFunctionSpace(mesh, "Lagrange", 2)

# Define trial and test functions and
# unknown solution u = displacement
du = TrialFunction(V_u) # Incremental displacement
v_u = TestFunction(V_u) # Test function
u = Function(V_u) # Solution u
dim = len(u)
I = Identity(dim)

V_scalar = FunctionSpace(mesh, "Lagrange", 1)
V_tensor = TensorFunctionSpace(mesh, "DG", degree=0)

# u_prior = 0 at reference configuration
# u_prior = project(Constant((0.0, 0.0, 0.0)), V_u)
u_prior = project(Constant((0.0, 0.0)), V_u)

# k_attach = 0 at reference configuration
k_attach = project(k_a, V_scalar)

# k_detach = 0 at reference configuration
k_detach = project(k_d, V_scalar)

# # k_detach = 0 at reference configuration
# k_detach_prior = project(Constant(0.0), V_scalar)

# mu = I at reference configuration
mu_prior = project(I, V_tensor)

# C_total = 1 at reference configuration
C_total = project(C_tot, V_scalar)

# C_attach = 1 at reference configuration
C_attach_prior = project(C_att_0, V_scalar)
C_attach_refconfig = project(C_att_0, V_scalar)

# C_detach = 0 at reference configuration
C_detach_prior = project(C_det_0, V_scalar)

# # Apply the boundary conditions to the unit cube mesh
# def right_face(x, on_boundary):
#     return near(x[0], 1, DOLFIN_EPS)

# def left_face(x, on_boundary):
#     return near(x[0], 0, DOLFIN_EPS)

# def front_face(x, on_boundary):
#     return near(x[1], 0, DOLFIN_EPS)

# def bottom_face(x, on_boundary):
#     return near(x[2], 0, DOLFIN_EPS)

# class point_constrained_x_y_z(SubDomain):
#     def inside(self, x, on_boundary):
#         return near(x[0], 0, DOLFIN_EPS) and near(x[1], 0, DOLFIN_EPS) and near(x[2], 0, DOLFIN_EPS)

# point_x_0_y_0_z_0 = point_constrained_x_y_z()

# deformation = Expression("u_x", u_x=0.0, degree=0)

# bc_I = DirichletBC(V_u, Constant((0.0, 0.0, 0.0)), point_x_0_y_0_z_0, method="pointwise")
# bc_II = DirichletBC(V_u.sub(0), Constant(0.0), left_face)
# bc_III = DirichletBC(V_u.sub(1), Constant(0.0), front_face)
# bc_IV = DirichletBC(V_u.sub(2), Constant(0.0), bottom_face)
# bc_V = DirichletBC(V_u.sub(0), deformation, right_face)

# bcs = [bc_I, bc_II, bc_III, bc_IV, bc_V]

# Apply the boundary conditions to the unit cube mesh
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0., DOLFIN_EPS)

def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], L, DOLFIN_EPS)

def origin_point(x, on_boundary):
    return  near(x[0], 0.0, DOLFIN_EPS) and near(x[1], 0.0, DOLFIN_EPS)

# def left_pinpoints(x, on_boundary):
#     return near(x[0], 0, DOLFIN_EPS) and near(x[1], 0, DOLFIN_EPS)

deformation = Expression("u_x", u_x=0.0, degree=0)

# bc_I = DirichletBC(V_u.sub(0), Constant(0.0), left_boundary)
# bc_II = DirichletBC(V_u.sub(1), Constant(0.0), left_pinpoints, method='pointwise')
# bc_III = DirichletBC(V_u.sub(0), deformation, right_boundary)

bc_I = DirichletBC(V_u.sub(0), Constant(0.0), left_boundary)
bc_II = DirichletBC(V_u.sub(1), Constant(0.0), origin_point, method='pointwise')
bc_III = DirichletBC(V_u.sub(0), deformation, right_boundary)

bcs = [bc_I, bc_II, bc_III]

# B = Constant((0.0, 0.0, 0.0)) # Constant((0.0, -0.5, 0.0))  # Body force per unit volume
# T = Constant((0.0, 0.0, 0.0)) # Constant((0.1, 0.0, 0.0))  # Traction force on the boundary

B = Constant((0.0, 0.0))
T = Constant((0.0, 0.0))

# Update kinematics, concentrations, and conformation tensor
F_prior = I + grad(u_prior) # Deformation gradient from prior time step
F = I + grad(u) # Deformation gradient
F_inv = inv(F)
J = det(F) # Volume ratio

L_prior = ((F-F_prior)/Constant(dt))*inv(F_prior) # velocity gradient tensor
# L_prior = ((F-F_prior)/Constant(dt))*inv(F) # velocity gradient tensor
I_L_prior = tr(L_prior)
D_prior = (L_prior + L_prior.T)/Constant(2.0)

C_attach_dot = k_attach*C_detach_prior - k_detach*C_attach_prior
mu_dot = k_attach*C_detach_prior/C_attach_prior*I - k_detach*mu_prior - (C_attach_dot/C_attach_prior-I_L_prior)*mu_prior + D_prior*mu_prior + mu_prior*D_prior

C_attach = C_attach_dot*Constant(dt) + C_attach_prior
C_detach = C_total-C_attach

mu = mu_dot*Constant(dt) + mu_prior

I_mu = tr(mu)
lmbda_c = sqrt(I_mu/3.0)
lmbda_c__sqrt_nu = lmbda_c/sqrt(nu)

def Psi_e(mu):
    return 0.5*C_attach/C_attach_refconfig*tr(mu-I) + 0.5*kappa*(J-1)**2

# Update rates
# k_detach = k_detach_prior
# k_rupture = k_rupture_prior

# Calculate P = 1st PK stress
P_G = J*C_attach/C_attach_refconfig*(mu-I)*F_inv.T + (J-1)*J*kappa/G*F_inv.T # penalty method for incompressibility

# Specify the quadrature degree for efficiency
WF = (inner(P_G, grad(v_u)))*dx(metadata={"quadrature_degree": 4}) - dot(B, v_u)*dx - dot(T, v_u)*ds

# Gateaux derivative
J_0 = derivative(WF, u, du)

# Solve variational problem
varproblem = NonlinearVariationalProblem(WF, u, bcs, J=J_0)
solver = NonlinearVariationalSolver(varproblem)
solver.parameters.update(snes_solver_parameters)
info(solver.parameters, False)

# Save results to an .xdmf file since we have multiple fields
file_results = XDMFFile(MPI.comm_world, savedir+"/results.xdmf")

# Solve with Newton solver for each displacement value using the previous
# solution as a starting point
for t_ind, t in enumerate(time):
    # Update the displacement value
    u_def = u_list[t_ind]
    deformation.u_x = u_def
    # Solve the nonlinear problem (using Newton solver)
    (iter, converged) = solver.solve()

    # Calculate Cauchy stress
    sigma_G = P_G/J*F.T

    # Calculate the network free energy
    Psi = Psi_e(mu)

    # Perform necessary projections onto function spaces
    mu_val = project(mu, V_tensor)
    P_G_val = project(P_G, V_tensor)
    sigma_G_val = project(sigma_G, V_tensor)
    Psi_val = project(Psi, V_scalar)
    C_attach_val = project(C_attach, V_scalar)
    C_detach_val = project(C_detach, V_scalar)
    lmbda_c_val = project(lmbda_c, V_scalar)
    lmbda_c__sqrt_nu_val = project(lmbda_c__sqrt_nu, V_scalar)
    # k_detach_val = project(k_detach, V_scalar)
    # k_rupture_val = project(k_rupture, V_scalar)
    F_val = project(F, V_tensor)
    L_prior_val = project(L_prior, V_tensor)

    # Points lie along center of domain, which experiences no rotation during uniaxial tensile stretch. Therefore, the components of the deformation gradient are the stretch components
    sigma_G_11_center_point_array[t_ind] = sigma_G_val(center_point)[0]
    F_11_center_point_array[t_ind]       = F_val(center_point)[0]
    L_prior_11_center_point_array[t_ind] = L_prior_val(center_point)[0]
    mu_11_center_point_array[t_ind]      = mu_val(center_point)[0]

    sigma_G_22_center_point_array[t_ind] = sigma_G_val(center_point)[3]
    F_22_center_point_array[t_ind]       = F_val(center_point)[3]
    L_prior_22_center_point_array[t_ind] = L_prior_val(center_point)[3]
    mu_22_center_point_array[t_ind]      = mu_val(center_point)[3]

    Psi_center_point_array[t_ind]      = Psi_val(center_point)

    # sigma_G_22_center_point_array[t_ind] = sigma_G_val(center_point)[4]
    # F_22_center_point_array[t_ind]       = F_val(center_point)[4]
    # L_prior_22_center_point_array[t_ind] = L_prior_val(center_point)[4]
    # mu_22_center_point_array[t_ind]      = mu_val(center_point)[4]

    # sigma_G_33_center_point_array[t_ind] = sigma_G_val(center_point)[8]
    # F_33_center_point_array[t_ind]       = F_val(center_point)[8]
    # L_prior_33_center_point_array[t_ind] = L_prior_val(center_point)[8]
    # mu_33_center_point_array[t_ind]      = mu_val(center_point)[8]

    # Update parameters
    u_prior.assign(u)
    mu_prior.assign(mu_val)
    C_attach_prior.assign(C_attach_val)
    C_detach_prior.assign(C_detach_val)
    # k_detach_prior.assign(k_detach_val)
    # k_rupture_prior.assign(k_rupture_val)

    # Note that this is now a deep copy not a shallow copy
    u.rename("Displacement", "u")
    mu_val.rename("Conformation tensor", "mu_val")
    P_G_val.rename("Normalized 1st PK stress", "P_G_val")
    Psi_val.rename("Network free energy", "Psi_val")
    sigma_G_val.rename("Normalized Cauchy stress", "sigma_G_val")
    C_attach_val.rename("Attached chain concentration", "C_attach_val")
    C_detach_val.rename("Detached chain concentration", "C_detach_val")
    lmbda_c_val.rename("Chain stretch", "lmbda_c_val")
    lmbda_c__sqrt_nu_val.rename("Normalized chain stretch", "lmbda_c__sqrt_nu_val")
    # k_attach_val.rename("Rate of attachment", "k_attach_val")
    # k_detach_val.rename("Rate of detachment", "k_detach_val")
    # k_rupture_val.rename("Rate of rupture", "k_rupture_val")
    F_val.rename("Deformation gradient", "F_val")
    L_prior_val.rename("Velocity gradient", "L_prior_val")

    file_results.parameters["rewrite_function_mesh"] = False
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u,t)
    file_results.write(mu_val,t)
    file_results.write(P_G_val,t)
    file_results.write(Psi_val,t)
    file_results.write(sigma_G_val,t)
    file_results.write(C_attach_val,t)
    file_results.write(C_detach_val,t)
    file_results.write(lmbda_c_val,t)
    file_results.write(lmbda_c__sqrt_nu_val,t)
    # file_results.write(k_attach_val,t)
    # file_results.write(k_detach_val,t)
    # file_results.write(k_rupture_val,t)
    file_results.write(F_val,t)
    file_results.write(L_prior_val,t)

plt.rcParams['axes.linewidth'] = 1.0 #set the value globally
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['text.usetex'] = True

plt.rcParams['ytick.right']= True
plt.rcParams['ytick.direction']="in"
plt.rcParams['xtick.top']=True
plt.rcParams['xtick.direction']="in"

plt.rcParams["xtick.minor.visible"] = True

plt.figure(0)
plt.plot(time, u_list, linestyle='-', color='black', alpha=1, linewidth=2.5)
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$u_1$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"u_1_vs_t.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(1)
plt1data1, = plt.plot(time, sigma_G_11_center_point_array, linestyle='-', color='red', label=r'$\frac{\sigma_{11}}{G}$', alpha=1, linewidth=2.5)
plt1data2, = plt.plot(time, sigma_G_22_center_point_array, linestyle='--', color='red', label=r'$\frac{\sigma_{22}}{G}$', alpha=1, linewidth=2.5)
# plt1data3, = plt.plot(time, sigma_G_33_center_point_array, linestyle=':', color='red', label=r'$\frac{\sigma_{33}}{G}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt1data1, plt1data2, plt1data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt1data1, plt1data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$\frac{\sigma}{G}$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"sigma_G_vs_t.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(2)
plt2data1, = plt.plot(time, F_11_center_point_array, linestyle='-', color='red', label=r'$F_{11}$', alpha=1, linewidth=2.5)
plt2data2, = plt.plot(time, F_22_center_point_array, linestyle='--', color='red', label=r'$F_{22}$', alpha=1, linewidth=2.5)
# plt2data3, = plt.plot(time, F_33_center_point_array, linestyle=':', color='red', label=r'$F_{33}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt2data1, plt2data2, plt2data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt2data1, plt2data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$F$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"F_vs_t.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(3)
plt3data1, = plt.plot(time, L_prior_11_center_point_array, linestyle='-', color='red', label=r'$L_{11}$', alpha=1, linewidth=2.5)
plt3data2, = plt.plot(time, L_prior_22_center_point_array, linestyle='--', color='red', label=r'$L_{22}$', alpha=1, linewidth=2.5)
# plt3data3, = plt.plot(time, L_prior_33_center_point_array, linestyle=':', color='red', label=r'$L_{33}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt3data1, plt3data2, plt3data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt3data1, plt3data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$L$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"L_prior_vs_t.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(4)
plt4data1, = plt.plot(time, mu_11_center_point_array, linestyle='-', color='red', label=r'$\mu_{11}$', alpha=1, linewidth=2.5)
plt4data2, = plt.plot(time, mu_22_center_point_array, linestyle='--', color='red', label=r'$\mu_{22}$', alpha=1, linewidth=2.5)
# plt4data3, = plt.plot(time, mu_33_center_point_array, linestyle=':', color='red', label=r'$\mu_{33}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt4data1, plt4data2, plt4data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt4data1, plt4data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$\mu$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"mu_vs_t.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(5)
plt.plot(time, Psi_center_point_array, linestyle='-', color='black', alpha=1, linewidth=2.5)
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$\Psi$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"Psi_vs_t.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(6)
plt.plot(time, lambda_list, linestyle='-', color='black', alpha=1, linewidth=2.5)
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$\lambda_1$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"lambda_1_vs_t.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(7)
plt1data1, = plt.plot(lambda_list, sigma_G_11_center_point_array, linestyle='-', color='red', label=r'$\frac{\sigma_{11}}{G}$', alpha=1, linewidth=2.5)
plt1data2, = plt.plot(lambda_list, sigma_G_22_center_point_array, linestyle='--', color='red', label=r'$\frac{\sigma_{22}}{G}$', alpha=1, linewidth=2.5)
# plt1data3, = plt.plot(lambda_list, sigma_G_33_center_point_array, linestyle=':', color='red', label=r'$\frac{\sigma_{33}}{G}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt1data1, plt1data2, plt1data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt1data1, plt1data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$\lambda_1$', fontsize=30)
plt.ylabel(r'$\frac{\sigma}{G}$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"sigma_G_vs_lambda_1.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(8)
plt2data1, = plt.plot(lambda_list, F_11_center_point_array, linestyle='-', color='red', label=r'$F_{11}$', alpha=1, linewidth=2.5)
plt2data2, = plt.plot(lambda_list, F_22_center_point_array, linestyle='--', color='red', label=r'$F_{22}$', alpha=1, linewidth=2.5)
# plt2data3, = plt.plot(lambda_list, F_33_center_point_array, linestyle=':', color='red', label=r'$F_{33}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt2data1, plt2data2, plt2data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt2data1, plt2data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$\lambda_1$', fontsize=30)
plt.ylabel(r'$F$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"F_vs_lambda_1.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(9)
plt3data1, = plt.plot(lambda_list, L_prior_11_center_point_array, linestyle='-', color='red', label=r'$L_{11}$', alpha=1, linewidth=2.5)
plt3data2, = plt.plot(lambda_list, L_prior_22_center_point_array, linestyle='--', color='red', label=r'$L_{22}$', alpha=1, linewidth=2.5)
# plt3data3, = plt.plot(lambda_list, L_prior_33_center_point_array, linestyle=':', color='red', label=r'$L_{33}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt3data1, plt3data2, plt3data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt3data1, plt3data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$\lambda_1$', fontsize=30)
plt.ylabel(r'$L$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"L_prior_vs_lambda_1.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(10)
plt4data1, = plt.plot(lambda_list, mu_11_center_point_array, linestyle='-', color='red', label=r'$\mu_{11}$', alpha=1, linewidth=2.5)
plt4data2, = plt.plot(lambda_list, mu_22_center_point_array, linestyle='--', color='red', label=r'$\mu_{22}$', alpha=1, linewidth=2.5)
# plt4data3, = plt.plot(lambda_list, mu_33_center_point_array, linestyle=':', color='red', label=r'$\mu_{33}$', alpha=1, linewidth=2.5)
# plt.legend(handles=[plt4data1, plt4data2, plt4data3], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.legend(handles=[plt4data1, plt4data2], loc='upper right', fontsize=14, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(r'$\lambda_1$', fontsize=30)
plt.ylabel(r'$\mu$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"mu_vs_lambda_1.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()

plt.figure(11)
plt.plot(lambda_list, Psi_center_point_array, linestyle='-', color='black', alpha=1, linewidth=2.5)
plt.xlabel(r'$\lambda_1$', fontsize=30)
plt.ylabel(r'$\Psi$', fontsize=30)
plt.tight_layout()
plt.savefig(savedir+"Psi_vs_lambda_1.pdf", transparent=True)
# plt.savefig(savedir+"N_8_energetic_lmbda_b_crit_13162.eps", format='eps', dpi=1000, transparent=True)
plt.close()