
# %% import modules
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import time

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
# target equation: T_ij = epsilon_0 * (E[i] * E[j] - 0.5 * E2 * delta_ij) + (1 / mu_0) * (B[i] * B[j] - 0.5 * B2 * delta_ij)

# constants
epsilon_0 = 8.854e-12  # F/m(C/V*m)
mu_0 = 4 * np.pi * 1e-7  # H/m(N/A^2)

# field parameters
E0 = 1e6  # (V/m)
B0 = 0.001    # (T)
k = np.pi   # (1/m)

# Define the electric and magnetic field functions
def E_field(x, y, z):
    return np.array([
        E0 * np.sin(k * x),
        E0 * np.cos(k * y),
        E0 * np.sin(k * z)
    ])

def B_field(x, y, z):
    return np.array([
        B0 * np.cos(k * x),
        B0 * np.sin(k * y),
        B0 * np.cos(k * z)
    ])

def maxwell_stress_tensor(x, y, z):
    E = E_field(x, y, z)
    B = B_field(x, y, z)
    E2 = np.dot(E, E)
    B2 = np.dot(B, B)
    T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            delta = 1 if i == j else 0
            T[i, j] = (
                epsilon_0 * (E[i] * E[j] - 0.5 * E2 * delta) +
                (1 / mu_0) * (B[i] * B[j] - 0.5 * B2 * delta)
            )
    return T

n_points = 150
points = np.random.uniform(low=-1, high=1, size=(n_points, 3))

E_iE_j = np.zeros((n_points, 3, 3))
B_iB_j = np.zeros((n_points, 3, 3))
delta_ij = np.zeros((n_points, 3, 3))
E2 = np.zeros((n_points,1))
B2 = np.zeros((n_points,1))

for idx, (x, y, z) in enumerate(points):
    E = E_field(x, y, z)
    B = B_field(x, y, z)
    E_iE_j[idx] = np.outer(E, E)
    B_iB_j[idx] = np.outer(B, B)
    E2[idx] = np.dot(E, E)
    B2[idx] = np.dot(B, B)
    delta_ij[idx] = np.eye(3)  # Kronecker delta, identity matrix

T_ij = epsilon_0*(E_iE_j - 0.5 * E2[:,None] * delta_ij) + (1 / mu_0) * (B_iB_j - 0.5 * B2[:,None] * delta_ij)
Y = T_ij

dict_of_variables = {'E_iE_j':E_iE_j,
                     'B_iB_j':B_iB_j,
                     'delta_ij':delta_ij,
                     'mu_0':mu_0,
                     'epsilon_0':epsilon_0,
                     'E2':E2,
                     'B2':B2,}

symbolic_function_map = {
        'tensor_add': linker_add,
        'tensor_sub': operator.sub,
        'tensor_inner_product': operator.mul,
        'p_': p_symbol,
        operator.add.__name__: operator.add,
        operator.sub.__name__: operator.sub,
        operator.mul.__name__: operator.mul,
        'protected_div': protected_div_symbol,
    }

dict_of_operators = {'tensor_add':tensor_add,
                    'tensor_sub':tensor_sub,
                    'tensor_inner_product':tensor_inner_product,
                    operator.add.__name__: operator.add,
                    operator.sub.__name__: operator.sub,
                    operator.mul.__name__: operator.mul,
                    'protected_div':protected_div,
                    'linker_add':linker_add}

# [M,L,T,I]
dict_of_dimension = {'E_iE_j':[2,2,-6,-2],
                     'B_iB_j':[2,0,-4,-2],
                     'delta_ij':[0,0,0,0],
                     'mu_0':[1,1,-2,-2],
                     'epsilon_0':[-1,-3,4,2],
                     'E2':[2,2,-6,-2],
                     'B2':[2,0,-4,-2],} 

NUM_UNITS = 4  # length of dimension list
target_dimension = [1,-1,-2,0]


# %% Creating the primitives set
# Define the operators
host_pset = gep.PrimitiveSet('Host', input_names=['E_iE_j','B_iB_j','delta_ij'])
host_pset.add_function(tensor_add, 2)
host_pset.add_function(tensor_sub, 2)
host_pset.add_function(tensor_inner_product, 2)
host_pset.add_function(p_, 1)

plasmid_pset = gep.PrimitiveSet('Plasmid', input_names=['E2', 'B2'])
plasmid_pset.add_symbol_terminal('epsilon_0', epsilon_0)
plasmid_pset.add_symbol_terminal('mu_0', mu_0)
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
output_type = 'maxwell_only_tlr'     # Name of the problem

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



# %%
