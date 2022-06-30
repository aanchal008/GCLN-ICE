from tkinter import Y
import numpy as np
import pandas as pd
from fractions import Fraction
from math import gcd, floor
from pip import main
import torch
import subprocess
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from subprocess import run
import json
import matplotlib.pyplot as plt
import csv
from statistics import mean
from z3 import *
from functools import reduce

max_denominator = 10

def smt_check(inv, name, path2result, path2smt, check='./check.sh'):
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
    p = subprocess.Popen(["check", "result_file", "path2smt" + "/" + "name"], stdin = subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print("out is following:", out)
    print("error is following:", err)
    
    try:
        return subprocess.run([check, result_file, path2smt + "/" + name], stdout=subprocess.PIPE, timeout=3)
    except:
        return None

def construct_invariant(var_names, eq_coeff):

    reals = []
    ands = []
    for var in var_names:
        if var == '1':
            reals.append(1)
        else:
            reals.append(Real(var))

    if eq_coeff is not None:
        eq_constraint = 0 * 0
        if (eq_coeff[0,0] != 0):
            eq_constraint = reals[0]*eq_coeff[0,0]
        for i, real in enumerate(reals[1:]):
            if ( eq_coeff[0,i+1] != 0):
                eq_constraint += eq_coeff[0, i+1] * real

        if isinstance(eq_constraint == 0, z3.BoolRef):
            ands += [eq_constraint == 0]
    
    print("equality is:", eq_constraint)

    Is = And(*ands)
    Inv = [Is]

    return Inv

def gaussian(data, k):
    data = - 0.5*((data/k) ** 2)
    data = data.exp()
    return data

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(outputSize, inputSize).uniform_(-1, 1))
        print("Weights assigned initally are following:", self.weight)
    
    def forward(self, x):
        with torch.no_grad():
            for weight in self.weight:
                weight = weight / torch.max(torch.abs(weight))  
        out = torch.nn.functional.linear(x, self.weight)
        return out

class CLN(torch.nn.Module):
    def __init__(self, inputSize, midSize):
        super(CLN, self).__init__()
        self.inputSize, self.midSize = inputSize, midSize
        self.or_gates = torch.nn.Parameter(torch.Tensor(midSize, inputSize // midSize).uniform_(1.0))
        self.and_gates = torch.nn.Parameter(torch.Tensor(midSize).fill_(1.0))

    def forward(self, x):
        xs = torch.chunk(x, self.midSize, dim=1)
        with torch.no_grad():
            self.or_gates.data.clamp_(0.0, 1.0)
            self.and_gates.data.clamp_(0.0, 1.0)
        mids = []
        for x_, or_gate in zip(xs, self.or_gates):
            mid = 1 - torch.prod(1 - x_ * or_gate, -1)
            mids.append(mid.view(-1, 1))
        mids_ = torch.cat(mids, 1)
        out = torch.prod(1 + self.and_gates * (mids_ - 1), -1)

        return out

def data_normalize(data):
    data = 10 * normalize(data, norm='l2', axis=1)
    return data


def gcln_model(i):
    df = pd.read_csv("500.csv")
    y = df['target']
    df_data = df.drop(columns=['init', 'final', 'target'])
    learning_rate = 0.001
    max_epoch = 4000
    loss_threshold = 1e-6
    min_std=0.1
    Is = gcln_infer_data(df_data, y, learning_rate=learning_rate, max_epoch=max_epoch,min_std=min_std, loss_threshold=loss_threshold)

    ext = ".c"

    run(['mkdir', '-p', 'tmp'])

    for I in Is:
        p = smt_check(I.sexpr(), str(i) + ext, "tmp",  "smt2")
        if p is None:
            continue
        screen_output = p.stdout.decode("utf-8")
        solved = screen_output.count('unsat') == 3
        if solved:
            break

    return True

def gcln_infer_data(df_data, y, learning_rate=0.001, max_epoch=4000,min_std=0.1, loss_threshold=1e-6):
    data = df_data.to_numpy(dtype=np.float)
    ##normalizing data using l2 norm
    data = data_normalize(data)

    # or_reg=(0.0000001, 1.00001, 0.0000001)
    or_reg=(0.001, 1.00001, 0.1)
    # and_reg=(1.000, 0.99999, 0.1)
    and_reg=(1.0, 0.99999, 0.1)

    or_reg, or_reg_decay, max_or_reg = or_reg
    and_reg, and_reg_decay, min_and_reg = and_reg

    input_size = data.shape[1]
    coeff = None

    if input_size > 1:
        valid_equality_found = False
        mid_width, out_width = 1, 1
        inputs_np = np.array(data, copy=True)
        ## mean and standard deviation array with initially assigned value of 0
        ## calculate the mean and standard deviation for each feature
        means_input, std_input = np.zeros([input_size], dtype=np.double), np.zeros([input_size], dtype=np.double)
    
        for i in range(input_size):
            means_input[i] = np.mean(data[:, i])
            std_input[i] = np.std(data[:, i])
            inputs_np[:, i] = (data[:, i] - means_input[i])
        inputs = torch.from_numpy(inputs_np).float()

        input_size = input_size
        model = linearRegression(input_size, 1)
        cln = CLN(mid_width, out_width)
        optimizer = torch.optim.Adam(list(model.parameters())+list(cln.parameters()), lr=learning_rate)

        for epoch in range(max_epoch):
            print("epoc is:", epoch)
            optimizer.zero_grad
            outputs = model(inputs).squeeze()
            print("model outputs are:", outputs)
            outputs_std = max([outputs.std().detach(), min_std])
            activation = gaussian(outputs, outputs_std)
            print("activation output is:", activation)
            final_outputs = cln(activation.reshape([-1,1]))
            print("final outputs are:", final_outputs)

            y = torch.Tensor(y)
            np_arr_pos = final_outputs.clone()
            mask_pos = (y == 1)
            pos = torch.masked_select(np_arr_pos, mask_pos)
            count_positive = torch.numel(pos)
            loss_pos = torch.sub(1, pos)
            loss_pos = torch.sum(loss_pos)
            print("positive loss is:", count_positive, loss_pos)

            np_arr_neg = final_outputs.clone()
            mask_neg = (y == 0)
            neg = torch.masked_select(np_arr_neg, mask_neg)
            count_negative = torch.numel(neg)
            loss_neg = torch.sum(neg) 
            print("negative loss is:", count_negative,loss_neg)

            np_arr_for_two = final_outputs.clone()
            mask_two = (y == 2)
            two = torch.masked_select(np_arr_for_two, mask_two)
            count_imp = torch.numel(two)

            np_arr_for_three = final_outputs.clone()
            mask_three = (y == 3)
            three = torch.masked_select(np_arr_for_three, mask_three)

            imp = torch.sub(two, three)
            imp_mask = (imp > 0)
            imp = torch.masked_select(imp, imp_mask)
            loss_imp = torch.sum(imp) 
            print("loss due to implication is:", count_imp, loss_imp)

            total_loss = (loss_pos + loss_neg + loss_imp)/(count_positive + count_negative + count_imp)

            print("total loss is:", total_loss, total_loss.grad_fn)
            print("\n")

            or_reg = min(or_reg * or_reg_decay, max_or_reg)
            and_reg = max(and_reg * and_reg_decay, min_and_reg)
            l_or_reg =  or_reg * torch.sum(torch.abs(cln.or_gates))
            l_and_reg =  -and_reg * torch.sum(torch.abs(cln.and_gates))

            loss = total_loss + l_or_reg + l_and_reg 
            
            if total_loss < loss_threshold:
                valid_equality_found = True
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cln.parameters(), 0.01)
            optimizer.step()

        coeff_ = model.weight.detach().numpy().reshape([input_size])
        
        coeff = []
        denominator = 1
        for i in range(input_size):
            a = Fraction.from_float(float(coeff_[i])).limit_denominator(max_denominator)
            coeff.append(a)
            denominator = denominator * a.denominator // gcd(denominator, a.denominator)
        coeff = np.asarray([[floor(a * denominator) for a in coeff]])
        print("coefficients are:", coeff)
        var_names = list(df_data.columns)

        Is = construct_invariant(var_names, coeff)
        print("invariant is:", Is)

    return Is

gcln_model(500)

