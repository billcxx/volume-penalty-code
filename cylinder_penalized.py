import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
import dedalus.public as de
import matplotlib.pyplot as plt
import os
from scipy.special import erf
import time
import logging
root = logging.root
for h in root.handlers: h.setLevel("INFO")
logger = logging.getLogger(__name__)

# Simulation parameters
Re = 100
eta = 1e-2
epsilon = np.sqrt(eta/Re)
delta = 3.8017192844*epsilon
l = 0
N = [256,64,64,128]
boundaries = sorted(set([.1,1-l-delta,1-l+delta,4]))
iterations, wall_time = 1000+1, 1*60*60
dt = 5e-3
print_freq = 10
sim_name = 'cylinder-penalized'
data_dir = os.path.join('runs',sim_name)
if rank==0 and not os.path.isdir(data_dir): os.makedirs(data_dir)

# Create problem bases and domain
thetabasis = de.Fourier('theta', N[0], interval=(-np.pi,np.pi), dealias=3/2)
rbases = [de.Chebyshev(f'r{i}',N[i+1],interval=boundaries[i:i+2], dealias=3/2) for i in range(len(boundaries)-1)]
rbasis = de.Compound('r',rbases)
domain = de.Domain([thetabasis,rbasis], grid_dtype=np.float64)
theta, r = domain.grids(domain.dealias)
theta2,rr = np.meshgrid(theta,r,indexing='ij')

# Boundary condition functions
from dedalus.core.operators import GeneralFunction
# Define GeneralFunction subclass for time dependent boundary conditions
class ConstantFunction(GeneralFunction):
    def __init__(self, domain, layout, func, args=[], kw={}, out=None,):
        super().__init__(domain, layout, func, args=[], kw={}, out=None,)

    def meta_constant(self, axis):
        return True

def normalized_mask(x): return np.piecewise(x, [x<=-1,(x>-1)&(x<1),x>=1],
                                           [lambda x:1,
                                            lambda x:(1-erf(np.sqrt(np.pi)*x/np.sqrt(1-x**2)))/2,
                                            lambda x:0])
def bc_func(solver): return normalized_mask(1-solver.sim_time/2)
def oscillation_func(solver): return np.sin(2*solver.sim_time/np.pi)

bc = ConstantFunction(domain, layout='g', func=bc_func)
oscillation = ConstantFunction(domain, layout='g', func=oscillation_func)

# Volume penalty mask
gamma = domain.new_field(scales=domain.dealias)
gamma['g'] = normalized_mask((rr-1)/delta)

disk = de.IVP(domain, variables=['u','v','p','q'], ncc_cutoff=1e-10)
disk.meta[:]['r']['dirichlet'] = True

# Parameters
params = [boundaries[0],boundaries[-1],Re,bc,np.pi] + N + [eta, epsilon, delta, l, gamma]
param_names = ['R0','R1','Re','bc','PI','Ntheta']+[f'Nr{i}' for i in range(len(N)-1)] +['eta', 'epsilon', 'delta', 'l', 'gamma']
if len(params)==len(param_names):
    for param, param_name in zip(params, param_names): 
        disk.parameters[param_name] = param

disk.substitutions['pr'] = "p - 0.5*(u*u+v*v)"
disk.substitutions['qr'] = "q/r"
disk.substitutions['c'] = "cos(theta)"
disk.substitutions['s'] = "sin(theta)"
disk.substitutions['fpr'] = "-pr"
disk.substitutions['fptheta'] = "0"
disk.substitutions['fvr'] = "0"
disk.substitutions['fvtheta'] = "(dtheta(u) + r*dr(v) - v)/(Re*r)"
disk.substitutions['phi'] = "(PI/2)*(1-cos(2*t/PI))"
disk.substitutions['omega'] = "sin(2*t/PI)"
disk.substitutions['alpha'] = "(2/PI)*cos(2*t/PI)"

disk.add_equation("dr(r*u) + dtheta(v) = 0")
disk.add_equation("r*r*dt(u) + (1/Re)*dtheta(q) +       r*r*dr(p) =  r*v*q - (r*r*gamma/eta)*u")
disk.add_equation("r*r*dt(v) - (1/Re)*(r*dr(q) - q) + r*dtheta(p) = -r*u*q - (r*r*gamma/eta)*(v-r*omega)")
disk.add_equation("q - dr(r*v) + dtheta(u) = 0")

# Boundary conditions                                                       
disk.add_bc("left(u)  = 0")
disk.add_bc("left(v)  = omega*R0")
disk.add_bc("right(u) = bc*cos(theta)", condition="(ntheta != 0)")
disk.add_bc("right(v) =-bc*sin(theta)")
disk.add_bc("right(p) = 0", condition="(ntheta == 0)")

# Build timestepper and solver
ts = de.timesteppers.SBDF3
solver = disk.build_solver(ts)
solver.stop_sim_time, solver.stop_wall_time, solver.stop_iteration = np.inf, wall_time, iterations

# Initialize variables
bc.original_args = bc.args = [solver]
oscillation.original_args = oscillation.args = [solver]
u, v, p, q = (solver.state[name] for name in disk.variables)
for field in [u,v,p,q]: 
    field.set_scales(domain.dealias)
    field['g'] = 0

# Save state variables
analysis = solver.evaluator.add_file_handler('{}/data-{}'.format(data_dir,sim_name), iter=10, max_writes=200,mode='overwrite')
for task in disk.variables: analysis.add_task(task)
analysis.add_task("omega")
analysis.add_task("phi")
analysis.add_task("alpha")
analysis.add_task("bc")

# Save force calcs
forces = solver.evaluator.add_file_handler('{}/force-{}'.format(data_dir,sim_name), 
                                           iter=1, max_writes=iterations,mode='overwrite')
forces.add_task("integ(interp((c*fpr-s*fptheta)*r,r='left'),'theta')",name='Fpx')
forces.add_task("integ(interp((c*fvr-s*fvtheta)*r,r='left'),'theta')",name='Fvx')
forces.add_task("integ(interp((s*fpr+c*fptheta)*r,r='left'),'theta')",name='Fpy')
forces.add_task("integ(interp((s*fvr+c*fvtheta)*r,r='left'),'theta')",name='Fvy')
forces.add_task("integ(interp(fvtheta*r*r,r='left'),'theta')",name='Tv')
forces.add_task("omega")
forces.add_task("phi")
forces.add_task("alpha")

# Save simulation parameters   
parameters = solver.evaluator.add_file_handler('{}/parameters-{}'.format(data_dir,sim_name), iter=np.inf, max_writes=np.inf,mode='overwrite')
for param_name in param_names: parameters.add_task(param_name)

# Run the simulation
start_time = time.time()

while solver.ok:
    solver.step(dt)
    if solver.iteration % print_freq == 0:
        logger.info('It:{:0>5d}, Time:{:.2f}, Max u:{:.2f}'.format(solver.iteration, (time.time()-start_time)/60,u['g'][N[0]//4,-1]))
