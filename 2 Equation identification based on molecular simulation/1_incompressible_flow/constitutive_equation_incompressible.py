# %% import modules
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import time
import scipy.io as scio

# For reproduction
s = 0
random.seed(s)
np.random.seed(s)

import SITE
from simplification import p_symbol, linker_add, protected_div_symbol


# %% define functions for tensor operators 
def tensor_add(*args):
    args = [arg for arg in args]
    sum = 0
    for arg in args:
        sum += arg
    return sum

def tensor_sub(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        return a - b
    else:
        return False

def tensor_inner_product(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        result = np.empty_like(a)
        for i in range(a.shape[0]):
            result[i] = np.matmul(a[i], b[i])
        return result
    else:
        return False
        
# define a protected division to avoid dividing by zero
def protected_div(x1, x2):
    if isinstance(x2, np.ndarray):
        abs_x2 = np.maximum(x2,-x2)
        if (abs_x2 < 1e-10).any():
            return 1
        return x1 / x2
    else:
        if abs(x2) < 1e-10:
            return 1
        return x1 / x2

# a placeholder function
def p_(tensor):
    pass


# %%
data = scio.loadmat('data/gradients.mat')
u_x = data['u_x']
u_y = data['u_y']
v_x = data['v_x']
v_y = data['v_y']
x_data = data['x']  
y_data = data['y'] 

target_data = np.loadtxt('data/TecGrid-Cavity_Kn0.005_nloop50000000.dat', delimiter=' ', skiprows = 3)
p_xx = target_data[:,7:8]
p_xy = target_data[:,10:11]
p_yx = p_xy
p_yy = target_data[:,8:9]

n_rho = target_data[:,2:3]
k = 1.38065e-23
T = target_data[:,6:7]
p = n_rho * k * T

x_range1 = (0.2, 0.8)
y_range1 = (0.2, 0.8)
x_indices = np.where((x_data >= x_range1[0]) & (x_data <= x_range1[1]))[0]
y_indices = np.where((y_data >= y_range1[0]) & (y_data <= y_range1[1]))[0]

indices0 = np.intersect1d(x_indices, y_indices)

u_x = u_x[indices0]
u_y = u_y[indices0]
v_x = v_x[indices0]
v_y = v_y[indices0]

p_xx = p_xx[indices0]
p_xy = p_xy[indices0]
p_yx = p_yx[indices0]
p_yy = p_yy[indices0]

p = p[indices0]

# subsample
indices = np.random.choice(u_x.shape[0], 200, replace=False)

u_x = u_x[indices]
u_y = u_y[indices]
v_x = v_x[indices]
v_y = v_y[indices]

p_xx = p_xx[indices]
p_xy = p_xy[indices]
p_yx = p_yx[indices]
p_yy = p_yy[indices]

p = p[indices]

u_i__j = np.stack((np.stack((u_x, u_y), axis=1), np.stack((v_x, v_y), axis=1)), axis=2)
u_j__i = np.stack((np.stack((u_x, v_x), axis=1), np.stack((u_y, v_y), axis=1)), axis=2)
u_i__j = u_i__j.reshape((len(u_x), 2, 2))
u_j__i = u_j__i.reshape((len(u_x), 2, 2))

S_ij = 0.5 * (u_i__j + u_j__i)
Omega_ij = 0.5 * (u_i__j - u_j__i)

miu = 2.117e-5 # Viscosity coefficient, as the temperature is not high, we can use a constant value

D_kk = u_x + v_y

shape = (len(u_x),1)
ones = np.ones(shape, dtype = np.float64)
zeros = np.zeros(shape, dtype = np.float64)
delta_ij = np.stack((np.stack((ones, zeros), axis=1), np.stack((zeros, ones), axis=1)), axis=2)
delta_ij = delta_ij.reshape((len(u_x), 2, 2))                         

sigma_ij = np.stack((np.stack((p_xx, p_xy), axis=1), np.stack((p_yx, p_yy), axis=1)), axis=2)
sigma_ij = sigma_ij.reshape((len(p_xx), 2, 2))

Y = sigma_ij

dict_of_variables = {'S_ij':S_ij,
                     'delta_ij':delta_ij,
                     'D_kk':D_kk,
                     'miu':miu,
                     'p':p}

symbolic_function_map = {
        'tensor_add': linker_add,
        'tensor_sub': operator.sub,
        'tensor_inner_product': operator.mul,
        'p_': p_symbol,
        operator.add.__name__: linker_add,
        operator.sub.__name__: operator.sub,
        operator.mul.__name__: operator.mul,
        'protected_div': operator.truediv,
    }

dict_of_operators = {'tensor_add':tensor_add,
                     'tensor_sub':tensor_sub,
                     'tensor_inner_product':tensor_inner_product,
                     operator.add.__name__: operator.add,
                     operator.sub.__name__: operator.sub,
                     operator.mul.__name__: operator.mul,
                     'protected_div':protected_div,
                     'linker_add':linker_add
                     }

dict_of_dimension = {'S_ij':[0,0,-1],
                     'delta_ij':[0,0,0],
                     'D_kk':[0,0,-1],
                     'miu':[1,-1,-1],
                     'p':[1,-1,-2]} 

NUM_UNITS = 3  # length of dimension list

target_dimension = [1,-1,-2]

host_pset = gep.PrimitiveSet('Host', input_names=['S_ij','delta_ij']) # you can add "Omega_ij" as input_names if you want to test its effect
host_pset.add_function(tensor_add, 2)
host_pset.add_function(tensor_sub, 2)
host_pset.add_function(tensor_inner_product, 2)
host_pset.add_function(p_, 1)

plasmid_pset = gep.PrimitiveSet('Plasmid', input_names=['p','D_kk'])
plasmid_pset.add_symbol_terminal('miu', miu)
plasmid_pset.add_function(operator.add, 2)
plasmid_pset.add_function(operator.sub, 2)
plasmid_pset.add_function(operator.mul, 2)
plasmid_pset.add_function(protected_div, 2)

# %% Create the individual and population
# Define the indiviudal class, a subclass of gep.Chromosome
creator.create("FitnessMax", base.Fitness, weights=(-1,))  # weights=(-1,)/weights=(1,) means to minimize/maximize the objective (fitness).
creator.create("Host_Individual", gep.Chromosome, fitness=creator.FitnessMax, plasmid=[])
creator.create("Plasmid_Individual", gep.Chromosome) 

# Register the individual and population creation operations
h_host = 5            # head length
h_plasmid = 10
n_genes_host = 4  # number of genes in a chromosome for hosts
n_genes_plasmid = 1      # number of genes in a chromosome for plasmids

toolbox = gep.Toolbox()

toolbox.register('host_gene_gen', gep.Gene, pset=host_pset, head_length=h_host)
toolbox.register('host_individual', creator.Host_Individual, gene_gen=toolbox.host_gene_gen, n_genes=n_genes_host, linker=tensor_add)
toolbox.register("host_population", tools.initRepeat, list, toolbox.host_individual)

creator.create("tinder", gep.Chromosome, plasmid=[])
toolbox.register('tinder_individual', creator.tinder, gene_gen=toolbox.host_gene_gen, n_genes=1, linker=tensor_add)

toolbox.register('plasmid_gene_gen', gep.Gene, 
                 pset=plasmid_pset, head_length=h_plasmid)
toolbox.register('plasmid_individual', creator.Plasmid_Individual, 
                 gene_gen=toolbox.plasmid_gene_gen, 
                 n_genes=n_genes_plasmid, 
                 linker=linker_add)
toolbox.register("plasmid_population", SITE.plasmid_generate, 
                 toolbox.plasmid_individual)
     
toolbox.register('compile', SITE.my_compile, 
                 dict_of_operators = dict_of_operators, 
                 symbolic_function_map = symbolic_function_map, 
                 dict_of_variables = dict_of_variables,
                 Y = Y)
toolbox.register('dimensional_verification', SITE.dimensional_verification, 
                 dict_of_dimension = dict_of_dimension, 
                 num_units = NUM_UNITS, 
                 target_dimension = target_dimension)
toolbox.register('evaluate', SITE.evaluate, 
                 tb = toolbox, 
                 dict_of_operators = dict_of_operators,
                 dict_of_variables = dict_of_variables, 
                 Y = Y)

# %% Register genetic operators
toolbox.register('select', tools.selTournament, tournsize=200) # Selection operator
# 1. general operators for host population
toolbox.register('mut_uniform', SITE.mutate_uniform, host_pset = host_pset, 
                 func = toolbox.plasmid_individual, ind_pb=0.2, pb=1)
toolbox.register('mut_invert', SITE.invert, pb=0.2)
toolbox.register('mut_is_transpose', SITE.is_transpose, pb=0.2)
toolbox.register('mut_ris_transpose', SITE.ris_transpose, pb=0.2)
toolbox.register('mut_gene_transpose', SITE.gene_transpose, pb=0.2)
toolbox.register('cx_1p', SITE.crossover_one_point, pb=0.2)
toolbox.register('cx_2p', SITE.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', SITE.crossover_gene, pb=0.2)
# 2. general operators for plasmid population
toolbox.register('mut_uniform_plasmid', gep.mutate_uniform, pset = plasmid_pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_plasmid', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose_plasmid', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose_plasmid', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose_plasmid', gep.gene_transpose, pb=0.1)

# %% Statistics to be inspected
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
# stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# %% Launch evolution

# Define size of population and number of generations
n_pop = 1600             # Number of individuals in a host_population
n_gen = 2000            # Maximum Generation
tol = 1e-6               # Threshold to terminate the evolution
output_type = 'incompressible'     # Name of the problem

host_pop = toolbox.host_population(n=n_pop) 
plasmid_pop = toolbox.plasmid_population(host_pop)
for ind_host, ind_plasmid in zip(host_pop, plasmid_pop):
    ind_host.plasmid = ind_plasmid 

# Only record the best three individuals ever found in all generations
champs = 3 
hof = tools.HallOfFame(champs)   


# %%
# Evolve
start_time = time.time()
pop, log = SITE.gep_simple(host_pop, plasmid_pop, toolbox, n_generations=n_gen, n_elites=1, n_alien_inds=100,
                          stats=stats, hall_of_fame=hof, verbose=True,tolerance = tol,GEP_type = output_type)
print(time.time() - start_time)

