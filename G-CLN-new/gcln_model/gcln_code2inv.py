from tkinter import Y
import numpy as np
import pandas as pd
from fractions import Fraction
from math import gcd, floor
from pip import main
import torch
import deepinv as dinv
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from subprocess import run
import json
import matplotlib.pyplot as plt
import csv
from statistics import mean
from z3 import *
from functools import reduce
# from new_multi_lin_eq_solver import CLN

global_i = -1
global_y = []
# max_denominator = 10
max_denominator = 2


def relu(d1,d2):
    if (d1 - d2) < 0:
        return 0
    else:
        return (d1- d2)



def check_early_stopping(last_50_losses, main_loss, loss_index):
    last_50_losses[loss_index] = main_loss
    loss_index = (loss_index+1)%50
    for i in range(1,50):
        if last_50_losses[i] != last_50_losses[0]:
            return False, loss_index
    return True, loss_index

def is_operand(c):
    return c.isnumeric()

def isexpression(expr):
    if "*" in expr or "+" in expr or "-" in expr or "/" in expr:
        return True
    return False


def evaluate(exp):
    tokens = exp.split()
    stack = []
    for t in reversed(tokens):
        print(t,stack)
        if   t == '+': stack[-2:] = [stack[-1] + stack[-2]]
        elif t == '-': stack[-2:] = [stack[-1] - stack[-2]]
        elif t == '*': stack[-2:] = [stack[-1] * stack[-2]]
        elif t == '/': stack[-2:] = [stack[-1] / stack[-2]]
        else: stack.append(int(t))
    assert len(stack) == 1, 'Malformed expression'
    return stack[0]

def func(x):
    f = open("tmp_file.txt", "r")
    tmp = f.readlines()
    index = int(tmp[0].rstrip("\n"))
    y = []
    for val in tmp[1:]:
        y.append(int(val.rstrip("\n")))
    f.close()

    f = open("tmp_file.txt","w")
    f.write("0\n")
    for ele in y:
        f.write(str(ele) + "\n")
    f.close()

    if y[index] == 1:
        return x
    elif y[index] == 0:
        return 1 - x
    elif y[index] == 2:
        return 1 - x
    else:
        return 1 + x

def counterexample(counter_example, problem_number, current_dataframe, updated_dataframe, var_names):
    print("var_names :", var_names)
    #path = "../benchmarks/code2inv/traces_with_targets/" + str(problem_number) + ".csv"
    if counter_example["pre"]:
        #print("Counter examples found for pre and values for x and y are:", counter_example["pre"]['x'], counter_example["pre"]['y'])
        var_values = []
        vars = []
        variables = []
        for var in var_names:
            if isexpression(var) or var == "1":
                continue
            else:
                variables.append(var)

        for var in var_names[1:]:
            if isexpression(var):
                exp = var.lstrip('(').rstrip(')')
                #print("variables", variables)
                #print("exp in pre", exp)
                for variable in variables:
                    print(exp, str(variable), str(counter_example["pre"][str(variable)]))
                    exp = exp.replace(str(variable), str(counter_example["pre"][str(variable)]))
                #print("exp in pre", exp)
                var_values.append(evaluate(exp))
            else:
                var_values.append(int(counter_example["pre"][str(var)]))
            vars.append(str(var))
        df_value = pd.Series([1]+var_values,index=['1']+vars)
        #df_value = pd.Series([1,int(x),int(y)],index=['1','x','y'])
        current_dataframe = current_dataframe.append(df_value, ignore_index=True)
        for new_row, old_row in enumerate(range(1), start=len(current_dataframe)):
            current_dataframe.ix[new_row] = df_value

        df_value = pd.Series([1,1,1] + var_values + [1], index=['init', 'final', '1'] + vars +['target'])
        for new_row, old_row in enumerate(range(1), start=len(updated_dataframe)):
            updated_dataframe.ix[new_row] = df_value
    
    
    if counter_example["recursive"]:
        #print("Counter examples found for recursive :", counter_example["recursive"]['x'], counter_example["recursive"]['y'], counter_example["recursive"]['x!'], counter_example["recursive"]['y!'])
    
        #x = counter_example["recursive"]['x']
        #y = counter_example["recursive"]['y']
        
        #x1 = counter_example["recursive"]['x!']
        #y1 = counter_example["recursive"]['y!']

        var_values = []
        var_values_final = []
        vars = []
        variables = []
        for var in var_names:
            if isexpression(var) or var == "1":
                continue
            else:
                variables.append(var)

        for var in var_names[1:]:
            if isexpression(var):
                exp = var.lstrip('(').rstrip(')')
                for variable in variables:
                    exp = exp.replace(str(variable), str(counter_example["recursive"][str(variable)]))
                var_values.append(evaluate(exp))
            else:
                var_values.append(int(counter_example["recursive"][str(var)]))
            vars.append(str(var))

        for var in var_names[1:]:
            if isexpression(var):
                exp = var.lstrip('(').rstrip(')')
                for variable in variables:
                    exp = exp.replace(str(variable), str(counter_example["recursive"][str(variable+"!")]))
                var_values_final.append(evaluate(exp))
            else:
                var_values_final.append(int(counter_example["recursive"][str(var+"!")]))

        #df_value = pd.Series([1,int(x),int(y)],index=['1','x','y'])
        df_value = pd.Series([1]+var_values,index=['1']+vars)
        for new_row, old_row in enumerate(range(1), start=len(current_dataframe)):
            current_dataframe.ix[new_row] = df_value
        #df_value = pd.Series([1,int(x1),int(y1)], index=['1','x','y'])
        df_value = pd.Series([1]+var_values_final,index=['1']+vars)
        for new_row, old_row in enumerate(range(1), start=len(updated_dataframe)):
            current_dataframe.ix[new_row] = df_value

        #df_value = pd.Series([1,1,1,int(x),int(y),2], index=['init', 'final', '1','x','y','target'])
        df_value = pd.Series([1,1,1] + var_values + [2], index=['init', 'final', '1'] + vars +['target'])
        for new_row, old_row in enumerate(range(1), start=len(updated_dataframe)):
            updated_dataframe.ix[new_row] = df_value
        #df_value = pd.Series([1,1,1,int(x1),int(y1),2], index=['init', 'final', '1','x','y','target'])
        df_value = pd.Series([1,1,1] + var_values_final + [3], index=['init', 'final', '1'] + vars +['target'])
        for new_row, old_row in enumerate(range(1), start=len(updated_dataframe)):
            updated_dataframe.ix[new_row] = df_value
        
    if counter_example["post"]:
        #print("Counter examples found for post and values for x and y are:", counter_example["post"]['x'], counter_example["post"]['y'])
        var_values = []
        vars = []
        variables = []
        for var in var_names:
            if isexpression(var) or var == "1":
                continue
            else:
                variables.append(var)

        for var in var_names[1:]:
            if isexpression(var):
                exp = var.lstrip('(').rstrip(')')
                for variable in variables:
                    exp = exp.replace(str(variable), str(counter_example["post"][str(variable)]))
                var_values.append(evaluate(exp))
            else:
                var_values.append(int(counter_example["post"][str(var)]))
            vars.append(str(var))
        
        df_value = pd.Series([1]+var_values,index=['1']+vars)
        current_dataframe = current_dataframe.append(df_value, ignore_index=True)
        for new_row, old_row in enumerate(range(1), start=len(current_dataframe)):
            current_dataframe.ix[new_row] = df_value

        df_value = pd.Series([1,1,1] + var_values + [0], index=['init', 'final', '1'] + vars +['target'])
        for new_row, old_row in enumerate(range(1), start=len(updated_dataframe)):
            updated_dataframe.ix[new_row] = df_value

class CLN(torch.nn.Module):
    def __init__(self, inputSize, midSize):
        super(CLN, self).__init__()
        self.inputSize, self.midSize = inputSize, midSize
        #print("Input size in cln and mid size", inputSize, midSize)
        #print("Or gates:")
        self.or_gates = torch.nn.Parameter(torch.Tensor(midSize, inputSize // midSize).uniform_(1.0))
        #print("And gates:")
        self.and_gates = torch.nn.Parameter(torch.Tensor(midSize).fill_(1.0))

    def forward(self, x):
        #print("x.shape is following:", x.shape)
        xs = torch.chunk(x, self.midSize, dim=1)
        #print("xs is:", xs)
        with torch.no_grad():
            self.or_gates.data.clamp_(0.0, 1.0)
            self.and_gates.data.clamp_(0.0, 1.0)
        mids = []
        for x_, or_gate in zip(xs, self.or_gates):
            mid = 1 - torch.prod(1 - x_ * or_gate, -1)
            mids.append(mid.view(-1, 1))
        mids_ = torch.cat(mids, 1)
        # out = torch.prod(mids_, -1)
        out = torch.prod(1 + self.and_gates * (mids_ - 1), -1)
        #print("Out value is following:", out)

        return out


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(outputSize, inputSize).uniform_(-1, 1))
        #self.weight = torch.nn.Parameter(torch.Tensor(outputSize, inputSize).xavier_uniform_(-1, 1))
        #print("Input size in linear", inputSize)
        #input size here is number of variables used
        #self.weight = torch.nn.Parameter(torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True))
        print("Weights assigned initally are following:", self.weight)
    

    def forward(self, x):
        with torch.no_grad():
            for weight in self.weight:
                weight = weight / torch.max(torch.abs(weight))  
        out = torch.nn.functional.linear(x, self.weight)
        return out


def data_normalize(data):
    ## normalizing column wise
    data = 10 * normalize(data, norm='l2', axis=1)
    return data


def gcln_infer(problem_number, problem_dataframe, learning_rate=0.01, max_epoch=4000,
                 loss_threshold=1e-6, non_loop_invariant=None):
    print("Reading the CSV file data:")
    print("gcln infer called on dataset of size:", len(problem_dataframe))
    #df = pd.read_csv("../benchmarks/code2inv/traces/" + str(problem_number) + ".csv")
    df = problem_dataframe.copy()
    y = df['target']
    df_data = df.drop(columns=['init', 'final', 'target'])
    var_names = list(df_data.columns)
    print("var_names are following:", var_names)
    df_data['1'] = 1
    consts = dinv.load_consts(problem_number, '../benchmarks/code2inv/smt2/const.txt')
    print("Constants from consts file for this problem are:", consts)
    Is, coeff, var_names = gcln_infer_data(df_data, y, consts, learning_rate=learning_rate, max_epoch=max_epoch,
            loss_threshold=loss_threshold, non_loop_invariant=non_loop_invariant, pname=problem_number)

    ext = ".c"

    print("Invariant synthesized in the gncln model:", Is)

    run(['mkdir', '-p', '../benchmarks/code2inv/tmp'])

    total_checks = 0
    correct_predictions = 0
    incorrect_predictions = 0
    
    for I in Is:
        print("\n")
        print("Is Invariant being considered here is:", I)
        #print("type of I is:", type(I))

        count = len(y)
        last_2 = False

        for i in range(count):
            var_values = {}
            for var in var_names:
                var_values[var] = int(df_data.iloc[i][var])
            result = True

            #reading one expression at a time from invariant
            for j in range(I.num_args()):
                I2 = I.arg(0)
                #print("I2 expression", I2)
                exp = str(I2)
                #print("printing exp", exp)
                exp = exp.replace("\n", "")
                #replacing variable with their values
                for var in var_names:
                    exp = exp.replace(var, str(var_values[var]))
                result = result and eval(exp)

            if y[i] == 1 and result:
                correct_predictions += 1
            elif y[i] == 0 and not result:
                correct_predictions += 1
            elif y[i] == 2:
                last_2 = result
            elif y[i] == 3:
                if (last_2 and result) or not last_2:
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
            else:
                incorrect_predictions += 1


            total_checks += 1

        print("accuracy: ", correct_predictions/(correct_predictions + incorrect_predictions))
        print("total checks In loop:",total_checks, len(y))

        p = dinv.smt_check(I.sexpr(), str(problem_number) + ext, "../benchmarks/code2inv/tmp", 
                "../benchmarks/code2inv/smt2")
        print("counterexamples from pipeline are following=====================",p)
        pk = str(p.stdout).lstrip("b'").rstrip("'")
        pk_list = pk.split("\\n")
        stdout_lines = pk_list
        counter_example = { "pre" : {},
                            "recursive" : {},
                            "post" : {}
                        }
        index = ""
        for i in range(len(stdout_lines)):
            line = stdout_lines[i]
            if "pre-condition" in line:
                index = "pre"
            if "recursive condition" in line:
                index = "recursive"
            if "post-condition" in line:
                index = "post"
            if "define-fun" in line:
                strings = line.split()
                variable_name = strings[1]
                line = stdout_lines[i+1]
                strings = line.split()
                val = ""
                val = val.join(strings)
                variable_value = val.rstrip(')').lstrip('(')
                counter_example[index][variable_name] = variable_value

        print(counter_example)
        
        df_last_3 = problem_dataframe.iloc[-3:]
        print("Before counter example updation out of function:\n", df_last_3)
        
        counterexample(counter_example, problem_number, df_data, problem_dataframe, var_names)

        df_last_3 = problem_dataframe.iloc[-3:]
        print("After counter example updation out of function:\n", df_last_3)

        if p is None:
            continue
        else:
            screen_output = p.stdout.decode("utf-8")
            solved = screen_output.count('unsat') == 3
            if solved:
                #count_program_solved += 1
                #print("solved problems equal to:", count_program_solved)
                break
            #else:
            #    print("out after smt check is following:", out)
    
    acc = correct_predictions/(correct_predictions + incorrect_predictions)
    
    print("total checks:",total_checks)
    print("accuracy till now is:", acc)

    return solved, I

   
    
def gcln_infer_data(df_data, y, consts, learning_rate=0.01, max_epoch=4000,
                 loss_threshold=1e-6, non_loop_invariant=None, min_std=0.1, max_denominator = 10, pname=1):
    ##converting data type to one type using to_numpy()
    print(df_data)
    data = df_data.to_numpy(dtype=np.float) 
    """
    f = open("tmp_file.txt","w")
    f.write("0\n")
    for ele in y:
        f.write(str(ele) + "\n")
    f.close()

    print("Number of Data values considered are:", data.shape[0])
    print("\n")
    #print("Data after converting to numpy array:", data)
    ##remove duplicates using unique()
    #data = np.unique(data, axis=0)
    """
    ##normalizing data using l2 norm
    data = data_normalize(data)


    # or_reg=(0.0000001, 1.00001, 0.0000001)
    or_reg=(0.001, 1.00001, 0.1)
    # and_reg=(1.000, 0.99999, 0.1)
    and_reg=(1.0, 0.99999, 0.1)

    or_reg, or_reg_decay, max_or_reg = or_reg
    and_reg, and_reg_decay, min_and_reg = and_reg

    ges, les, eqs = dinv.infer_single_var_bounds_consts(df_data, consts)

    input_size = data.shape[1]
    print("Input Size is:", input_size)
    coeff = None
    if input_size > 1:
        valid_equality_found = False

        # data preparation
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

        # build and train the model
        input_size = input_size
        model = linearRegression(input_size, 1)
        cln = CLN(mid_width, out_width)
        optimizer = torch.optim.Adam(list(model.parameters())+list(cln.parameters()), lr=learning_rate)

        last_50_losses = list(range(-1,-51,-1))
        loss_index = 0
        for epoch in range(max_epoch):
            print("Epoch number is:", epoch)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            #print("Output from the modelis following:", outputs)
            outputs_std = max([outputs.std().detach(), min_std])
            #activation = dinv.gaussian(outputs, outputs_std)
            activation = dinv.ges_fn(outputs)
            #activation = dinv.les_fn(outputs)
            #print("activation outputs are:", activation)
            final_outputs = cln(activation.reshape([-1,1]))
            #print("final outputs are:", final_outputs)
            
            
            y = torch.Tensor(y)
            #print(type(y), y)
            np_arr_pos = final_outputs.clone()
            #print(np_arr_pos)
            mask_pos = (y == 1)
            #print("printing positive mask", mask_pos)
            pos = torch.masked_select(np_arr_pos, mask_pos)
            #print(pos)
            count_positive = torch.numel(pos)
            loss_pos = torch.sub(1, pos)
            loss_pos = torch.sum(loss_pos)
            print("positive loss is:", count_positive, loss_pos)

            np_arr_neg = final_outputs.clone()
            mask_neg = (y == 0)
            neg = torch.masked_select(np_arr_neg, mask_neg)
            count_negative = torch.numel(neg)
            #print("printing negative mask", mask_neg)
            #print(neg)
            loss_neg = torch.sum(neg) 
            print("negative loss is:", count_negative,loss_neg)

            np_arr_for_two = final_outputs.clone()
            mask_two = (y == 2)
            two = torch.masked_select(np_arr_for_two, mask_two)
            count_imp = torch.numel(two)
            #print(two)

            np_arr_for_three = final_outputs.clone()
            mask_three = (y == 3)
            three = torch.masked_select(np_arr_for_three, mask_three)
            #print(three)

            imp = torch.sub(two, three)
            #print("Before masking of negative values:",imp)
            imp_mask = (imp > 0)
            #print("after masking the negative values:", imp)
            imp = torch.masked_select(imp, imp_mask)
            #print("final results in implication are:", imp)
            loss_imp = torch.sum(imp) 
            print("loss due to implication is:", count_imp, loss_imp)

            main_loss = (loss_pos + loss_neg + loss_imp)/(count_positive + count_negative + count_imp)
            print("total loss is:", main_loss, main_loss.grad_fn)
            #print("loss when masked tensor is used:", loss_pos)
            print("\n")

            
            losses_are_same, loss_index = check_early_stopping(last_50_losses, main_loss, loss_index)
            if losses_are_same:
                print("Stopped because Last 5 outputs same")
                break
        

            or_reg = min(or_reg * or_reg_decay, max_or_reg)
            and_reg = max(and_reg * and_reg_decay, min_and_reg)
            l_or_reg =  or_reg * torch.sum(torch.abs(cln.or_gates))
            l_and_reg =  -and_reg * torch.sum(torch.abs(cln.and_gates))

            loss = main_loss + l_or_reg + l_and_reg 
            
            ##loss threshold is 1e-6
            if main_loss < loss_threshold:
                valid_equality_found = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cln.parameters(), 0.01)
            optimizer.step()
        
        coeff_ = model.weight.detach().numpy().reshape([input_size])
        scaled_coeff = np.round(coeff_/np.abs(coeff_).min())
        coeff = []
        denominator = 1
        for i in range(input_size):
            a = Fraction.from_float(float(coeff_[i])).limit_denominator(max_denominator)
            coeff.append(a)
            denominator = denominator * a.denominator // gcd(denominator, a.denominator)
        coeff = np.asarray([[floor(a * denominator) for a in coeff]])


    var_names = list(df_data.columns)
    
    # print('loading', pname)
    with open('../benchmarks/code2inv/conditions/' + str(pname) + '.json', 'r') as f:
        condition = json.load(f)
    # print(condition)

    pred = condition['predicate']

    Is = dinv.construct_invariant(var_names, coeff, ges, les, eqs, pred, non_loop_invariant)

    if scaled_coeff.max() < 50: # large coeffs cause z3 timeouts
        #print("Scaled Coefficients are:", scaled_coeff)
        scaled_Is = dinv.construct_invariant(var_names, scaled_coeff.reshape(1,-1), ges, les, eqs, pred, non_loop_invariant)
        Is.extend(scaled_Is)


    return Is, coeff, var_names
    # ext = ".c"
    # if mod:
        # ext = "_mod"
    # p = dinv.smt_check(I.sexpr(), str(problem_number) + ext, "../results", "../code2inv/smt2")
    # screen_output = p.stdout.decode("utf-8")
    # if v:
        # print(screen_output)
        # print()
    # return screen_output.count('unsat') == 3


if __name__ == '__main__':
    import sys
    problem_number = int(sys.argv[1])
    # problem_number = 1
    gcln_infer(problem_number)