# coding=utf-8
# =============================================================================
# Symbolic identification of tensor equations in multidimensional physical fields
# Tianyi Chen, Hao Yang, Wenjun Ma, and Jun Zhang
# 2025. Beihang University
# This moudule implements the SITE algorithm, which is a modified version of GEP. SITE can be used for symbolic identification of tensor equations.
# =============================================================================
    
import copy
import numpy as np
import random
import os
import deap
import warnings
import geppy as gep
import time
from simplification import simplify
from scipy.linalg import qr
import gc

_DEBUG = False
without_LINEAR = False # If you want to use linear regression to scale the expression, choose 'False'.


class ExpressionTree:
    """
    A modified class used for SITE. The original code is from geppy.core.entity.ExpressionTree.
    """
    def __init__(self, root):
        self._root = root

    @property
    def root(self):
        """
        Get the root node of this expression tree.
        """
        return self._root

    class Node:
        def __init__(self, name, index = None):
            self._children = []
            self._name = name
            self._index = index

        @property
        def children(self):
            return self._children

        @property
        def name(self):
            return self._name

        @property
        def index(self):
            return self._index

    @classmethod
    def from_genotype(cls, genome):
        if isinstance(genome, gep.Gene):
            return cls._from_kexpression(genome.kexpression)
        elif isinstance(genome, gep.KExpression):
            return cls._from_kexpression(genome)
        elif isinstance(genome, gep.Chromosome):
            if len(genome) == 1:
                return cls._from_kexpression(genome[0].kexpression)
            sub_trees = [cls._from_kexpression(gene.kexpression, i) for i, gene in enumerate(genome)]
            # combine the sub_trees with the linking function
            root = cls.Node(genome.linker.__name__)
            root.children[:] = sub_trees
            return cls(root)
        raise TypeError('Only an argument of type KExpression, Gene, and Chromosome is acceptable. The provided '
                        'genome type is {}.'.format(type(genome)))

    @classmethod
    def _from_kexpression(cls, expr, index = None):
        """
        Create an expression tree from a K-expression.
        :param expr: a K-expression
        :return: :class:`ExpressionTree`, an expression tree
        """
        if len(expr) == 0:
            return None
        # first build a node for each primitive
        if index == None:
            nodes = [cls.Node(p.name, '[0]['+str(i)+']') for i, p in enumerate(expr)]
        else:
            nodes = [cls.Node(p.name, '['+str(index)+']['+str(i)+']') for i, p in enumerate(expr)]
        # connect each node to its children if any
        i = 0
        j = 0
        while i < len(nodes):
            for _ in range(expr[i].arity):
                j += 1
                nodes[i].children.append(nodes[j])
            i += 1
        return cls(nodes[0])


def plasmid_generate(func, host_pop):
    '''
    Generate a plasmid population based on the host population.
    :param func: The plasmid individual function
    :param host_pop: The host population
    '''
    plasmid_pop = []
    p_times_lis = _count_p_functions(host_pop)
    for i in range(len(p_times_lis)):
        plasmid_pop.append([])
        for j in range(len(p_times_lis[i])):
            plasmid_pop[i].append([])
            for k in range(p_times_lis[i][j]):
                plasmid_pop[i][j].append(func())
    return plasmid_pop

def _count_p_functions(host_pop):
    ''' 
    Count the number of 'p_' functions in each gene of each individual in the host population.
    '''
    p_counts_pop = []
    for ind in host_pop:
        p_counts_ind =[]
        for gene in ind:
            p_num = [symbol.name for symbol in gene].count('p_')
            p_counts_ind.append(p_num)
        p_counts_pop.append(p_counts_ind)
    return p_counts_pop

def extract_expressed_plasmid(host_ind):
    '''
    Extract the plasmid from the host individual and return a list of plasmids.
    '''
    plasmid_lis = host_ind.plasmid
    et = ExpressionTree.from_genotype(host_ind)
    def preorderTraversal(tree):
        stack = [tree.root]
        result = []
        while len(stack) != 0:
            node = stack.pop()
            if node.name == 'p_':
                result.append(node.index)
            if node.children is not None:
                for child in node.children:
                    if isinstance(child, ExpressionTree):
                        stack.append(child.root)
                    else:
                        stack.append(child)
        return result[::-1]    
    def renumber(q):
        parsed = [s[1:-1].split('][') for s in q]
        parsed = [[eval(i) for i in lis] for lis in parsed]
        d = {}
        for x, y in parsed:
            if x not in d:
                d[x] = []
            d[x].append(y)
        for x in d:
            sorted_unique_values = sorted(d[x])
            d[x] = {v: i for i, v in enumerate(sorted_unique_values)}
        for i in parsed:
            i[1] = d[i[0]][i[1]]
        result = ['['+str(i)+']['+str(j)+']' for i, j in parsed]
        return result        
    index_lis = preorderTraversal(et)  
    result = []
    for index in renumber(index_lis):
        result.append(eval('plasmid_lis'+str(index)))

    return result

def tlr(Individual, dict_of_operators, dict_of_variables, Y):
    '''
    A core function that implements the Tensor Linear Regression (TLR) technique.
    '''
    plasmid_library = [plasmid for plasmid in extract_expressed_plasmid(Individual)]
    
    def p_(tensor, plasmid_library= plasmid_library):
        if isinstance(tensor, np.ndarray):
            scalar = eval(str(plasmid_library.pop(0)),create_var)
            if isinstance(scalar, np.ndarray):
                return scalar[:, None] * tensor
            return tensor * scalar
        else:
            raise ValueError('inout of p_ is not a instance of ndarray')
        
    create_var = locals()
    create_var.update(dict_of_operators)
    create_var.update(dict_of_variables)

    if without_LINEAR:
        yp = eval(str(Individual), create_var)
        return yp
    
    n, d, _ = Y.shape
    X = []
    non_zero_index = []
    for i, term in enumerate(Individual):
        x = eval(str(term), create_var)
        if x.any() != np.zeros(Y.shape).any(): #If the term is totally 0-tensor, do not consider it into regression.
            X.append(x)
            non_zero_index.append(i)
    if len(X) == 0:
        return None, None, None
    m = len(X)

    Xlis = [[] for _ in range(d**2)]
    ylis = [[] for _ in range(d**2)]
    for i in range(d):
        for j in range(d):
            Xlis[i*d+j] = np.array([term[:, i, j] for term in X]).T
    for i in range(d):
        for j in range(d):
            ylis[i*d+j] = Y[:, i, j]
    A = np.zeros((m, m))
    b = np.zeros((m, 1))
    for i in range(d):
        for j in range(d):
            denom = np.dot(ylis[i*d+j].T, ylis[i*d+j])
            if denom < 1e-12:
                continue
            A += np.dot(Xlis[i*d+j].T, Xlis[i*d+j]) / denom
            b += np.dot(Xlis[i*d+j].T, ylis[i*d+j])[:, None] / denom
    eeps = 1e-6
    A_new = A.copy()
    
    Q, R, P = qr(A_new, mode='economic', pivoting=True)
    diag_R = np.abs(np.diag(R))
    tol = np.max(diag_R) * eeps
    true_rank = np.sum(diag_R > tol)
    P = P[:true_rank]
    independent_indices = P[::-1]
    if len(P) < m:
        X_pru = [X[i] for i in independent_indices]
        Xlis = [[] for _ in range(d**2)]
        for i in range(d):
            for j in range(d):
                Xlis[i*d+j] = np.array([term[:, i, j] for term in X_pru]).T
        A_new = np.zeros((len(P), len(P)))
        b = np.zeros((len(P), 1))
        for i in range(d):
            for j in range(d):
                denom = np.dot(ylis[i*d+j].T, ylis[i*d+j])
                if denom < 1e-12:
                    continue
                A_new += np.dot(Xlis[i*d+j].T, Xlis[i*d+j]) / denom
                b += np.dot(Xlis[i*d+j].T, ylis[i*d+j])[:, None] / denom
        W = np.linalg.solve(A_new, b)
        w_full = np.zeros((len(X), 1))
        for idx, val in zip(independent_indices, W):
            w_full[idx] = val
        W = w_full
    else:
        W = np.linalg.solve(A, b)

    yp = np.zeros((n, d, d))
    for i in range(d):
        for j in range(d):
            yp[:, i, j] = np.dot(W.T, np.stack(X)[:,:,i,j])
            
    return np.real(W), non_zero_index, yp

def my_compile(Individual, dict_of_operators, symbolic_function_map, dict_of_variables, Y):
    # Compile the expression to human-readable format.
    plasmid_library = []
    plasmidind_lib = extract_expressed_plasmid(Individual)
    for plasmid_ind in plasmidind_lib:
        scalar_expression = simplify(plasmid_ind, plasmid_library = None, symbolic_function_map = symbolic_function_map)
        plasmid_library.append(scalar_expression)

    if without_LINEAR:
        tensor_expression = simplify(Individual, plasmid_library=plasmid_library, symbolic_function_map=symbolic_function_map)
        return str(tensor_expression)

    w, non_zero_index, _ = tlr(Individual, dict_of_operators, dict_of_variables, Y)
    if not isinstance(w, np.ndarray):
        return 'All terms are zero!'

    final_terms = []
    for i in range(len(Individual)):
        if i not in non_zero_index:
            term_str = str(simplify(Individual[i], plasmid_library=plasmid_library, symbolic_function_map=symbolic_function_map))
            continue
        term_str = str(simplify(Individual[i], plasmid_library=plasmid_library, symbolic_function_map=symbolic_function_map))
        final_term = str(w[non_zero_index.index(i)]) +' * [' + term_str + ']'
        final_terms.append(final_term)
    tensor_expression = ''
    for i, final_term in enumerate(final_terms):
        tensor_expression = (tensor_expression + final_term + ' + ') if i != len(final_terms)-1 else (tensor_expression + final_term)
    return str(tensor_expression)

def dimensional_verification(individual, dict_of_dimension, num_units, target_dimension):
    # Dimensional homogeneity checking function.
    plasmid_library = [str(plasmid).replace('\t','').replace('\n','').replace('linker_add','add')
                       .replace('add','dim_add').replace('sub','dim_sub')
                       .replace('mul','dim_mul').replace('protected_div','dim_div').replace('square','dim_square').replace('pow','dim_pow')
                       for plasmid in extract_expressed_plasmid(individual)]

    def dim_add(*args):
        if any(isinstance(arg, bool) for arg in args):
            return False
        args = [arg for arg in args]
        for i in range(len(args)):
            if isinstance(args[i],int) or isinstance(args[i],float):
                args[i] = [0] * num_units    #[0,0,0,...]
                
        if all(arg == args[0] for arg in args):
            return args[0]
        return False

    def dim_sub(a,b):
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        if isinstance(a,int) or isinstance(a,float):
            a = [0] * num_units
        if isinstance(b,int) or isinstance(b,float):
            b = [0] * num_units

        if a == b:
            return a
        else:
            return False

    def dim_mul(a,b):
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        if isinstance(a,int) or isinstance(a,float):
            a = [0] * num_units
        if isinstance(b,int) or isinstance(b,float):
            b = [0] * num_units
            
        return [x + y for x, y in zip(a, b)]

    def dim_div(a,b):
        if isinstance(a,bool) or isinstance(b,bool) or b == 0:
            return False
        if isinstance(a,int) or isinstance(a,float):
            a = [0] * num_units
        if isinstance(b,int) or isinstance(b,float):
            b =[0] * num_units

        return [x - y for x, y in zip(a, b)]

    def dim_p_(tensor, plasmid_library = plasmid_library):
        if isinstance(tensor, bool):
            return False
        scalor_expr = plasmid_library.pop(0)
        scalor = eval(scalor_expr, create_var)

        return dim_mul(tensor, scalor)
    
    def dim_square(a):
        if isinstance(a,bool):
            return False
        if isinstance(a,int) or isinstance(a,float):
            a = [0] * num_units
        return [2*x for x in a]
    
    def dim_pow(a,b):
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        if b != [0] * num_units or a != [0] * num_units:
            return False
        return [0] * num_units
    
    create_var = locals()
    create_var.update(dict_of_dimension)

    individual_expr = str(individual).replace('\t','').replace('\n','').replace('tensor_add','dim_add').replace('tensor_sub','dim_sub').replace('tensor_inner_product','dim_mul').replace('p_','dim_p_')
    dimension_of_DDEq = eval(individual_expr)
    if dimension_of_DDEq == target_dimension:
        return True
    else:
        return False

def loss_func(Yp, Y):
    total_loss = 0
    valid_count = 0  # Counter for valid (i,j) components
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            yp = Yp[:, i, j].reshape(-1, 1)
            y = Y[:, i, j].reshape(-1, 1)
            l2_norm_delta_y = np.sqrt(np.sum((yp - y) ** 2))
            l2_norm_y = np.sqrt(np.sum((y) ** 2))
            if l2_norm_y < 1e-12:
                # If the true value is zero, penalize the prediction magnitude directly
                loss_ij = np.sqrt(np.sum(yp ** 2))
            else:
                # Otherwise, use the relative L2 error
                loss_ij = l2_norm_delta_y / l2_norm_y
            total_loss += loss_ij
            valid_count += 1
    if valid_count == 0:
        raise ValueError("No valid components to compute loss!")
    fitness = total_loss / valid_count
    return fitness

def evaluate(individual, tb, dict_of_operators, dict_of_variables, Y):
    """
    First check whether the individuals satisfy dimensional homogeneity.
    If it is not dimensional homogeneous, we would identify it as an invalid individual and directly assign a significant loss to it. Otherwise, we evaluate its loss.
    """
    validity = tb.dimensional_verification(individual)
    if not validity:
        return 1000,
    else:
        if without_LINEAR:
            Yp = tlr(individual, dict_of_operators, dict_of_variables, Y)
        else:
            Yp = tlr(individual, dict_of_operators, dict_of_variables, Y)[2]
        if isinstance(Yp, np.ndarray):
            return loss_func(Yp, Y),
        else: #Yp = None
            return 1000,

# new genetic operators designed for SITE
def _choose_subsequence(seq, min_length=1, max_length=-1):
    if max_length <= 0:
        max_length = len(seq)
    length = random.randint(min_length, max_length)
    start = random.randint(0, len(seq) - length)
    return start, start + length
    
def _choose_function(pset):
    return random.choice(pset.functions)

def _choose_a_terminal(terminals):
    terminal = random.choice(terminals)
    if isinstance(terminal, gep.EphemeralTerminal):  # an Ephemeral
        terminal = copy.deepcopy(terminal)  # generate a new one
        terminal.update_value()
    return terminal

def _choose_terminal(pset):
    return _choose_a_terminal(pset.terminals) 

def mutate_uniform(host_individual, host_pset, func, ind_pb='2p'):
    plasmid_lis_lis = host_individual.plasmid
    if isinstance(ind_pb, str):
        assert ind_pb.endswith('p'), "ind_pb must end with 'p' if given in a string form"
        length = host_individual[0].head_length + host_individual[0].tail_length
        ind_pb = float(ind_pb.rstrip('p')) / (len(host_individual) * length)
    for gene, plasmid_lis in zip(host_individual, plasmid_lis_lis):
        # mutate the gene with the associated pset
        # head: any symbol can be changed into a function or a terminal
        for i in range(gene.head_length):
            if random.random() < ind_pb:
                if gene[i].name == 'p_' : # p mutate
                    if random.random() < 0.5:  # to a function
                        new_function = _choose_function(host_pset)
                        gene[i] = new_function
                        if new_function.name != 'p_': # p to else function
                            plasmid_lis.pop(0)
                    else:                      # to a terminal
                        gene[i] = _choose_terminal(host_pset)
                        plasmid_lis.pop(0) # p to terminal
                else: # not p mutate
                    if random.random() < 0.5:  # to a function
                        new_function = _choose_function(host_pset)
                        gene[i] = new_function
                        if new_function.name == 'p_': # else to p
                            plasmid_lis.insert(0, func())
                    else:                      # to a terminal
                        gene[i] = _choose_terminal(host_pset)
        # tail: only change to another terminal
        for i in range(gene.head_length, gene.head_length + gene.tail_length):
            if random.random() < ind_pb:
                gene[i] = _choose_terminal(host_pset)
    return host_individual,

def invert(individual):
    """
    A gene is randomly chosen, and afterwards a subsequence within this gene's head domain is randomly selected
    and inverted.
    """
    if individual.head_length < 2:
        return individual,
    location = random.choice(range(len(individual)))
    gene = individual[location]
    plasmid_lis = individual.plasmid[location]
    start, end = _choose_subsequence(gene.head, 2, gene.head_length)
    subsequence = gene[start: end]
    n_i = 0
    n_j = 0
    for i in range(end+1):
        if gene[i].name == 'p_':
            n_i += 1
    for j in range(len(subsequence)):
        if subsequence[j].name == 'p_':
            n_j += 1
    gene[start: end] = reversed(gene[start: end])
    plasmid_lis[(n_i - n_j):n_i] = reversed(plasmid_lis[(n_i - n_j):n_i])
    if _DEBUG:
        print('invert [{}: {}]'.format(start, end))
    return individual,

def _choose_donor_donee(individual):
    i1, i2 = random.choices(range(len(individual)), k=2)  # with replacement
    return individual[i1], individual[i2], i1, i2

def _choose_subsequence_indices(i, j, min_length=1, max_length=-1):
    """
    Choose a subsequence from [i, j] (both included) and return the subsequence boundaries [a, b] (both included).
    """
    if max_length <= 0:
        max_length = j - i + 1
    length = random.randint(min_length, max_length)
    start = random.randint(i, j - length + 1)
    return start, start + length - 1

def is_transpose(individual):
    """
    Perform IS transposition
    """
    # Donor is the gene who give out a segment, while donee is the gene who take in a segment. Donor may be the same as donee.
    donor, donee, i1, i2 = _choose_donor_donee(individual) 
    donor_plasmid_lis = individual.plasmid[i1]
    donee_plasmid_lis = individual.plasmid[i2]
    a, b = _choose_subsequence_indices(0, donor.head_length + donor.tail_length - 1, max_length=donee.head_length - 1)
    is_start, is_end = a, b + 1
    is_ = donor[is_start: is_end]
    n_i = 0
    n_j = 0
    for i in donor[: is_end]:
        if i.name == 'p_':
            n_i += 1
    for j in is_:
        if j.name == 'p_':
            n_j += 1
    is_plasmid = donor_plasmid_lis[(n_i - n_j):n_i]
    insertion_pos = random.randint(1, donee.head_length - len(is_))
    n_k = 0
    for k in donee[:insertion_pos]:
        if k.name == 'p_':
            n_k += 1
    n_l = 0
    for l in donee[insertion_pos: insertion_pos + donee.head_length - insertion_pos - len(is_)]:
        if l.name == 'p_':
            n_l += 1
    donee_plasmid_lis[:] = donee_plasmid_lis[:n_k] + is_plasmid + donee_plasmid_lis[n_k : n_k+n_l]
    donee[:] = donee[:insertion_pos] + is_ + \
               donee[insertion_pos: insertion_pos + donee.head_length - insertion_pos - len(is_)] + \
               donee[donee.head_length:]
    if _DEBUG:
        print('IS transpose: g{}[{}:{}] -> g{}[{}:]'.format(i1, is_start, is_end, i2, insertion_pos))
    return individual,

def ris_transpose(individual):
    """
    Perform RIS transposition
    """
    n_trial = 0
    while n_trial <= 2 * len(individual):
        donor, donee, i1, i2 = _choose_donor_donee(individual)
        # choose a function node randomly to start RIS
        function_indices = [i for i, p in enumerate(donor.head) if isinstance(p, gep.Function)]
        if not function_indices:  # no functions in this donor, try another
            n_trial += 1
            continue
        donor_plasmid_lis = individual.plasmid[i1]
        donee_plasmid_lis = individual.plasmid[i2]
        ris_start = random.choice(function_indices)
        # determine the length randomly
        length = random.randint(2, min(donee.head_length, donor.head_length + donor.tail_length - ris_start))
        # insert ris at the root of donee
        ris = donor[ris_start: ris_start + length]
        n_i = 0
        n_j = 0
        for i in donor[: ris_start + length]:
            if i.name == 'p_':
                n_i += 1
        for j in ris:
            if j.name == 'p_':
                n_j += 1
        ris_plasmid = donor_plasmid_lis[(n_i - n_j):n_i]
        n_k = 0
        for k in donee[0: donee.head_length - length]:
            if k.name == 'p_':
                n_k += 1
        donee_plasmid_lis[:] = ris_plasmid + donee_plasmid_lis[: n_k]
        donee[:] = ris + donee[0: donee.head_length - length] + donee[donee.head_length:]
        if _DEBUG:
            print('RIS transpose: g{}[{}:{}] -> g{}[0:]'.format(i1, ris_start, ris_start + length, i2))
        return individual,
    return individual,

def gene_transpose(individual):
    """
    Perform gene transposition
    """
    if len(individual) <= 1:
        return individual,
    source = random.randint(1, len(individual) - 1)
    individual[0], individual[source] = individual[source], individual[0]
    individual.plasmid[0], individual.plasmid[source] = individual.plasmid[source], individual.plasmid[0]
    if _DEBUG:
        print('Gene transpose: g0 <-> g{}'.format(source))
    return individual,

def crossover_one_point(ind1, ind2):
    """
    Execute one-point recombination of two host individuals.
    """
    assert len(ind1) == len(ind2)
    # the gene containing the recombination point, and the point index in the gene
    which_gene = random.randint(0, len(ind1) - 1)
    which_point = random.randint(0, len(ind1[which_gene]) - 1)
    plasmid_lis1 = ind1.plasmid
    plasmid_lis2 = ind2.plasmid
    # exchange the upstream materials
    plasmid_lis1[:which_gene], plasmid_lis2[:which_gene] = plasmid_lis2[:which_gene], plasmid_lis1[:which_gene]
    n_i = 0
    n_j = 0
    for i in ind1[which_gene][:which_point + 1]:
        if i.name == 'p_':
            n_i += 1
    for j in ind2[which_gene][:which_point + 1]:
        if j.name == 'p_':
            n_j += 1
    plasmid_lis1[which_gene][:n_i], plasmid_lis2[which_gene][:n_j] = plasmid_lis2[which_gene][:n_j], plasmid_lis1[which_gene][:n_i]

    ind1[:which_gene], ind2[:which_gene] = ind2[:which_gene], ind1[:which_gene]
    ind1[which_gene][:which_point + 1], ind2[which_gene][:which_point + 1] = \
        ind2[which_gene][:which_point + 1], ind1[which_gene][:which_point + 1]
    if _DEBUG:
        print('cxOnePoint: g{}[{}]'.format(which_gene, which_point))
    return ind1, ind2


def crossover_two_point(ind1, ind2):
    """
    Execute two-point recombination of two individuals.
    """
    assert len(ind1) == len(ind2)
    plasmid_lis1 = ind1.plasmid
    plasmid_lis2 = ind2.plasmid
    # the two genes containing the two recombination points
    g1, g2 = random.choices(range(len(ind1)), k=2)  # with replacement, thus g1 may be equal to g2
    if g2 < g1:
        g1, g2 = g2, g1
    # the two points in g1 and g2
    p1 = random.randint(0, len(ind1[g1]) - 1)
    p2 = random.randint(0, len(ind1[g2]) - 1)
    
    # change the materials between g1->p1 and g2->p2: first exchange entire genes, then change partial genes at g1, g2
    if g1 == g2:
        if p1 > p2:
            p1, p2 = p2, p1
        n_i ,n_j, n_k, n_l = 0, 0, 0, 0
        for i in ind1[g1][: p2+1]:
            if i.name == 'p_':
                n_i += 1
        for j in ind1[g1][p1: p2+1]:
            if j.name == 'p_':
                n_j += 1
        for k in ind2[g2][: p2+1]:
            if k.name == 'p_':
                n_k += 1
        for l in ind2[g2][p1: p2+1]:
            if l.name == 'p_':
                n_l += 1
        plasmid_lis1[g1][n_i-n_j:n_i], plasmid_lis2[g2][n_k-n_l:n_k] = plasmid_lis2[g2][n_k-n_l:n_k], plasmid_lis1[g1][n_i-n_j:n_i]
        ind1[g1][p1: p2+1], ind2[g2][p1: p2+1] = ind2[g2][p1: p2+1], ind1[g1][p1: p2+1]
    else:
        n_i ,n_j, n_k, n_l = 0, 0, 0, 0
        for i in ind1[g1][:p1]:
            if i.name == 'p_':
                n_i += 1
        for j in ind2[g1][:p1]:
            if j.name == 'p_':
                n_j += 1
        for k in ind1[g2][: p2+1]:
            if k.name == 'p_':
                n_k += 1
        for l in ind2[g2][: p2+1]:
            if l.name == 'p_':
                n_l += 1
        plasmid_lis1[g1 + 1: g2], plasmid_lis2[g1 + 1: g2] = plasmid_lis2[g1 + 1: g2], plasmid_lis1[g1 + 1: g2]
        plasmid_lis1[g1][n_i:], plasmid_lis2[g1][n_j:] = plasmid_lis2[g1][n_j:], plasmid_lis1[g1][n_i:]
        plasmid_lis1[g2][:n_k], plasmid_lis2[g2][:n_l] = plasmid_lis2[g2][:n_l], plasmid_lis1[g2][:n_k]
        ind1[g1 + 1: g2], ind2[g1 + 1: g2] = ind2[g1 + 1: g2], ind1[g1 + 1: g2]
        ind1[g1][p1:], ind2[g1][p1:] = ind2[g1][p1:], ind1[g1][p1:]
        ind1[g2][:p2 + 1], ind2[g2][:p2 + 1] = ind2[g2][:p2 + 1], ind1[g2][:p2 + 1]
    if _DEBUG:
        print('cxTwoPoint: g{}[{}], g{}[{}]'.format(g1, p1, g2, p2))
    return ind1, ind2

def crossover_gene(ind1, ind2):
    """
    Entire genes are exchanged between two parent chromosomes.
    """
    assert len(ind1) == len(ind2)
    pos1, pos2 = random.choices(range(len(ind1)), k=2)
    ind1[pos1], ind2[pos2] = ind2[pos2], ind1[pos1]
    ind1.plasmid[pos1], ind2.plasmid[pos2] = ind2.plasmid[pos2], ind1.plasmid[pos1]
    if _DEBUG:
        print('cxGene: ind1[{}] <--> ind2[{}]'.format(pos1, pos2))
    return ind1, ind2

def _validate_basic_toolbox(tb):
    """
    Validate the operators in the toolbox *tb* according to our conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a 'select' operator."
    # whether the ops in .pbs are all registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Operators must start with 'mut' or 'cx' except selection."
        assert hasattr(tb, op), "Probability for a operator called '{}' is specified, but this operator is not " \
                                "registered in the toolbox.".format(op)
    # whether all the mut_ and cx_ operators have their probabilities assigned in .pbs
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('{0} is registered, but its probability is NOT assigned in Toolbox.pbs. '
                          'By default, the probability is ZERO and the operator {0} will NOT be applied.'.format(op),
                          category=UserWarning)


def _apply_modification_host(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])

            del population[i].fitness.values
    return population

def _apply_modification_plasmid(population, operator, pb):
    for i in range(len(population)):
        if len(population[i].plasmid) > 0:
            for plasmid_lis in population[i].plasmid:
                if type(population[i]) != deap.creator.Host_Individual:
                    print(type(population[i]))
                if len(plasmid_lis) > 0:
                    for plasmid_ind in plasmid_lis:
                        if random.random() < pb:
                            plasmid_ind, = operator(plasmid_ind)
                            del population[i].fitness.values
    return population

def _apply_crossover_host(op, population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population


def gep_simple(host_population, plasmid_population, toolbox, n_generations=100, n_elites=1, n_alien_inds =1,
               stats=None, hall_of_fame=None, verbose=__debug__,tolerance = 1e-10,GEP_type = ''):
    """
    The main function of GEP algorithm. Also, this reflects the main evolutionary loop of the SITE framework.
    """
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    start_time = time.time()
    
    is_exists = os.path.exists('output')
    if not is_exists:
        os.mkdir('output')
    
    best_fitness = 1000
    tinder_lib = set()

    for gen in range(n_generations + 1):
        # First, generation for hosts
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in host_population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(host_population)
        record = stats.compile(host_population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)

        if verbose:
            print(logbook.stream)

        if gen == n_generations:
            break

        # selection with elitism
        elites = deap.tools.selBest(host_population, k=n_elites)
        if n_elites > 0:
            if gen >= 0 and gen % 10 == 0:  
                tinder = []
                def dh_gene_generate():
                    while True:
                        ind = toolbox.tinder_individual()
                        gene = ind[0]
                        p_num = [symbol.name for symbol in gene].count('p_')
                        ind.plasmid = [[toolbox.plasmid_individual() for _ in range(p_num)]]
                        validity = toolbox.dimensional_verification(ind)
                        if validity and (tuple(gene) not in tinder_lib):
                            tinder_lib.add(tuple(gene))
                            return gene, ind.plasmid[0]
                        
                dh_ind = toolbox.host_individual()
                dh_ind.plasmid = []
                while len(tinder) < len(dh_ind):
                    dh_gene_zip = dh_gene_generate()
                    dh_gene = dh_gene_zip[0]
                    dh_gene_plasmid = dh_gene_zip[1]
                    tinder.append((dh_gene,dh_gene_plasmid))
                
                for num, gene in enumerate(tinder):
                    dh_ind[num] = gene[0]
                    dh_ind.plasmid.append(gene[1])
                dh_ind.fitness.values = toolbox.evaluate(dh_ind)
                tinder_ind = dh_ind
                ailen_inds = [tinder_ind] * n_alien_inds
        else:
            ailen_inds = []
        offspring = toolbox.select(host_population, len(host_population) - n_alien_inds - n_elites)

        # output the real-time result
        if gen > 0 and gen % 1 == 0:
            elites_IR = elites[0]
            simplified_best = toolbox.compile(elites_IR)
            if elites_IR.fitness.values[0] < best_fitness:
                best_fitness = elites_IR.fitness.values[0]
                elapsed = time.time() - start_time
                time_str = '%.2f' % (elapsed) 
                if elites_IR.fitness.values[0] != 1000:  
                    key= f'In generation {gen}, with CPU running {time_str}s, \nOur No.1 best prediction is:'
                    with open(f'output/{GEP_type}_equation.dat', "a") as f:
                        f.write('\n'+ key+ str(simplified_best)+ '\n'+f'with loss = {elites_IR.fitness.values[0]}'+'\n')
                else:
                    key= f'In generation {gen}, with CPU running {time_str}s, \nOur No.1 best prediction 1 is:'
                    with open(f'output/{GEP_type}_equation.dat', "a") as f:
                        f.write('\n'+ key+ str(simplified_best)+ '\n'+f'which is invalid!'+'\n' )

        # Termination criterion of error tolerance
        if gen > 0 and gen % 100 == 0:
            error_min = elites[0].fitness.values[0]
            if error_min < tolerance:
                break

        offspring = [toolbox.clone(ind) for ind in offspring]
        gc.collect()
        # mutation for host
        for op in toolbox.pbs:
            if op.startswith('mut') and (not op.endswith('plasmid')):
                offspring = _apply_modification_host(offspring, getattr(toolbox, op), toolbox.pbs[op])
        
        # crossover for host
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover_host(op, offspring, getattr(toolbox, op), toolbox.pbs[op])
        
        # mutation for plasmid
        for op in toolbox.pbs:
            if op.startswith('mut') and op.endswith('plasmid'):
                offspring = _apply_modification_plasmid(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # %% end of a total generation
        # replace the current host population with the offsprings and update plasmid population
        host_population = elites + offspring + ailen_inds
        for ind_host, ind_plasmid in zip(host_population, plasmid_population):
            ind_plasmid = ind_host.plasmid
    

    return host_population, logbook
