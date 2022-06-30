from __future__ import print_function
from typing_extensions import Self

import torch
import numpy as np
import subprocess
from z3 import *
from collections import defaultdict

counterexample_pos = []
counterexample_imp = []
counterexmaple_neg = []

def ges_fn(data):
    for i in range(len(data)):
        if data[i] >= 0:
            data[i] = ges(data[i])
        else:
            data[i] = les(data[i])
    return(data)

def les_fn(data):
    for i in range(len(data)):
        if data[i] >= 0:
            data[i] = les(data[i])
        else:
            data[i] = ges(data[i])
    return(data)

def pbqu(d, c_i, eps):
    d = ((c_i) ** 2) / ((d - eps) ** 2 + (c_i) ** 2)
    return d

def ges(d):
    return pbqu(d, 5, 0)

def les(d):
    return pbqu(d, 0.5, 0)

def gt(data):
    return pbqu(data, 0.5, 5, 0)

def lt(data):
    return pbqu(data, 0.5, 5, 0)

def gaussian (data, k):
    #assumes the form 2^-(kx^2) where k is an approximation for std dev
    #applies it uniformly across the whole matrix
    #instead of AOC = 1 we scale to have gaus(0) = 1
    data = - 0.5*((data/k) ** 2)
    data = data.exp()
    return data

CLN_OPS = {
            '=': gaussian,
            '>': gt,
            '<': lt,
            '>=': ges,
            '<=': les,
          }

class IneqConstraint(torch.nn.Module):
    def __init__(self, coeffs, var_names, op_str):
        super(IneqConstraint, self).__init__()
        coeffs = defaultdict(lambda:0, coeffs)
        coeff_w = [coeffs[var] for var in var_names]
        self.weight = torch.tensor(coeff_w, dtype=torch.float).reshape(1, -1)

        self.b = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.c1 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.c2 = torch.nn.Parameter(torch.tensor([5.0], requires_grad=True))
        self.eps = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

        self.op_str = op_str
        self.op = CLN_OPS[op_str]

    def forward(self, x, xn):
        with torch.no_grad():
            self.B.clamp_(min=0.1)
            self.eps.clamp_(min=0.5)

        weight, b = self.weight, self.b
        out = torch.nn.functional.linear(x, weight).squeeze() - b
        activation = self.op(out, self.B, self.eps)
        return activation

    def get(self, search_type):
        return [self] if isinstance(self, search_type) else []


def pwgaussian (data,k, alpha):
    # computes gaussian as above but with an asymmetric tail
    # should model ge
    powerl = -0.5*((data/k)**2)
    powerr = -0.5*((alpha* data/k)**2)
    left = powerl.exp()
    right = alpha*powerr.exp() + 1 - alpha 
    return left * data.lt(0).float() + right * data.ge(0).float()

def triangle (data,std):
    slope = -1/torch.abs(std)
    return (  torch.abs(data)*(slope) + 1  ).clamp(min=0) ## triangle


def primary_loss (out):
    return (1-out.mean())*(1-out.mean()) # if not batched, then mean is trivial


def log_loss(out):
    return torch.mean( -torch.log(out) )


def lin_weight_sum (linear_weights):
    return torch.abs(1 - torch.abs(linear_weights).sum())

def colwise_op (data, func):
    if data.ndim < 2:
        raise ValueError
    
    data = data.clone().detach()
    
    numcols = data.size()[1]
    if numcols < 2:
        return data
    
    out = data[:,[0]]
    for i in range(max(numcols-1, 0)):
        out = func(out, data[:,[i+1]])
    
    return out

def t_norm(data, type="prod"):
    if type == "luk":
        return colwise_op(data, lambda c1,c2: np.maximum(0, c1+c2-1))
    elif type == "godel":
        return data.min(dim=1).values
    elif type == "prod":
        return data.cumprod(dim=1)
    else:
        raise ValueError

def t_conorm(data, type="prod"):
    if type == "luk":
        return colwise_op(data, lambda c1,c2: np.maximum(1, c1+c2))
    elif type == "godel":
        return data.max(dim=1).values
    elif type == "prod":
        return colwise_op(data, lambda c1,c2: c1+c2-c1*c2)
    else:
        raise ValueError
        

def load_consts(problem_number, const_file):
    consts = []
    with open(const_file, 'rt') as const_file:
        for line in const_file:
            number, consts_original, consts_expand = line.strip().split()
            if int(number) == problem_number:
                consts = consts_expand.split(',')
                consts = [int(const) for const in consts]
                
    return consts


def infer_single_var_bounds(df, problem_number,
                           const_file='../code2inv/our_preprocess/const.txt'):
    ges = []
    les = []
    eqs = []

    consts = load_consts(problem_number, const_file)

    return infer_single_var_bounds_consts(df, consts)


def infer_single_var_bounds_consts(df, consts):
    ges = []
    les = []
    eqs = []

    for var in df.columns:
        #print("Inside infer single varible bounds, the value of df.columns:", df.columns)
        if var in ('init', 'final'):
            continue

        max_v = max(df[var].unique())
        #print("df[var] is:",df[var])
        #print("max_v is following:", max_v)
        min_v = min(df[var].unique())
        #print("min_v is following:", min_v)
        if max_v == min_v:
            if max_v in consts:
                eqs.append( (var, max_v ))
            continue
        if max_v in consts:
            les.append( (var, max_v) )
            print("les is:", les)
        if min_v in consts:
            ges.append( (var, min_v) )
            print("ges is:", ges)
            
    return ges, les, eqs


def construct_invariant(var_names, eq_coeff, ges, les, eqs, pred_str='',
        non_loop_invariant=None):
    #additional_expr=None, additional_expr2=None):
    print("Inside construct Invariant: Starting from here::::::::::::")
    pred1, pred2 = None, None
    #print("prediction is:", pred_str)
    #pred_str contains the loop condition and symbol is < or <=
    if pred_str is not None:
        if ('<' in pred_str) or ('<=' in pred_str):
            pred = pred_str.split()
            try:
                v1 = int(pred[2])
            except ValueError:
                v1 = Real(pred[2])
            try:
                v2 = int(pred[3])
            except ValueError:
                v2 = Real(pred[3])
            
            if pred[1] == '<':
                pred1 = v1 < v2
                pred2 = v1 <= v2
            elif pred[1] == '<=':
                pred1 = v1 <= v2
                pred2 = v1 <= v2 + 1

    #print('preds', pred1, pred2)
    #reals contain all the variables used in the program    
    reals = []
    for var in var_names:
        if var == '1':
            reals.append(1)
        else:
            reals.append(Real(var))

    #print("reals are:", reals)

    ands = []
    ineqs = []

    print("coefficients are:", eq_coeff)

    if eq_coeff is not None:
        eq_constraint = 0 * 0
        if (eq_coeff[0,0] != 0):
            eq_constraint = reals[0]*eq_coeff[0,0]
        for i, real in enumerate(reals[1:]):
            if ( eq_coeff[0,i+1] != 0):
                eq_constraint += eq_coeff[0, i+1] * real

        print('eq_constraint', eq_constraint)
        # print( type(eq_constraint == 0))
        #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO Change this for inequality change OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
        if isinstance(eq_constraint == 0, z3.BoolRef):
            ands += [eq_constraint >= 0]
        
    for ge in ges:
        ands.append(reals[var_names.index(ge[0])] >= ge[1])
        ineqs.append(reals[var_names.index(ge[0])] >= ge[1])

    for le in les:
        ands.append(reals[var_names.index(le[0])] <= le[1])
        ineqs.append(reals[var_names.index(ge[0])] >= ge[1])

    for eq_ in eqs:
        if eq_[0] != '1':
            ands.append(reals[var_names.index(eq_[0])] == eq_[1])

    # if additional_expr is not None:
        # ands.append(additional_expr)
    # print('extra')
    print("ges and les:\n", ges, les)
    #print("equality constraint is:", eq_)

    # print(ands)
    # for a in ands:
        # print(type(a))
    # ands = [a for a in ands if not isinstance(a, np.bool_)]
    # print(ands)
    I0 = And(*ineqs)
    I1 = And(*ands)
    I2, I3 = None, None
    if pred1 is not None and pred2 is not None:
        I2 = And(*ands, pred1)
        I3 = And(*ands, pred2)
    print('I0 is :', I0)
    print("I1 is:", I1)
    print('I2 and I3 are:', I2, I3)
    if non_loop_invariant is not None:
        # print('non loop', non_loop_invariant)
        print("non loop invarinat is following:", non_loop_invariant)
        I1 = Or(I1, non_loop_invariant)
        # print('I', I)
        if pred1 is not None and pred2 is not None:
            I2 = Or(I2, non_loop_invariant)
            I3 = Or(I3, non_loop_invariant)

    # print('I', I)
    # print('I2', I2)
    # print('I3', I3)
    Is = [I1]
    """
    if I2 is not None and I3 is not None:
        Is.extend([I2, I3])
        # return I1, I2, I3, I0
    Is.append(I0)
    """
    print("Ending Construct invariant:::::::::::::::::::")

    return Is


def smt_check(inv,name, path2result, path2smt, check='./check.sh'):
    '''
    name is the c file name: e.g. '100.c'
    '''
    inv = inv.replace('|', '')
    result_file = path2result + "/" + name + ".inv.smt" 
    with open(result_file , "w") as f:
        f.write(inv)
    #print("result_file: ", result_file)
    #print("path to smt: ", path2smt + "/" + name)
    #print("countexample is:", stdout = subprocess.PIPE)
        #print(subprocess.run([check, result_file, path2smt + "/" + name], stdout=subprocess.PIPE, timeout=3))
        #timeout used when other process don't reply
    #cmd = [check, result_file, path2smt + "/" + name]
    try:
        return subprocess.run([check, result_file, path2smt + "/" + name], stdout=subprocess.PIPE, timeout=3)
    except:
        return None


def generic_train(xs, model, lr=0.15, decay=1.0, max_iters=2000, v=False, logger=None, opt=None):

    params = [x for x in xs] 
    if opt:
        optimizer = opt
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)

    loss_hist = []
    for i in range(max_iters):
        optimizer.zero_grad()

        pred = model(xs)
#         print('  out',pred.data)
        loss = log_loss(pred)

        if v:
            print("{}: xs = {}, output = {}, loss = {}".format(i, xs, pred.data, loss.data.numpy().round(3)))
        if logger:
            logger(i, xs, pred, loss)

        loss_hist.append(loss.item())
        print("loss history is:", loss_hist)

        if torch.isnan(loss):
            print('terminating, loss is nan on iter',i)
            break

        if loss < 0.01:
            print("terminating, objective reached on iter {}, loss = {}"\
                    .format(i, loss.data.numpy().round(3)))
            print(xs)
            break

        if loss.requires_grad and not torch.isinf(loss):
            pred.retain_grad()
            loss.backward()
            if v:
                #print(' lgrad',pred.grad)
                for i, x in enumerate(xs):
                    print('  x{} grad [{}]'.format(i, x.grad.data.numpy()))

            optimizer.step()

            optimizer.param_groups[0]['lr'] *= decay

        else:
            print('no loss grad!')
            print(xs)
            break
                    
        if i == max_iters-1:
            print('terminating, did not reach objective in limit {} iters, loss = {}'.format(max_iters,
                    loss.data.numpy().round(3)))
            # print("final x = {}, loss = {:0.4f}".format(xs, loss_hist[-1]))
            print(xs)

    return xs, loss_hist