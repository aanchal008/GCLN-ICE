import pandas as pd
import numpy as np
from z3 import *
from gcln import gcln_model
import time
import sys, os
from z3 import Real, And

def run_gcln():
    solved = False
    attempts = 0
    I = None
    start_time = time.time()
    while(solved == False):
        attempts += 1
        if attempts > 20:
            break
        else:
            i = 500
            solved = gcln_model(i)
    
    end_time = time.time()
    runtime = end_time - start_time

    print('solved?',solved, 'time:', runtime)
    if I is not None:
        print(I)
    return solved, runtime

if __name__=='__main__':
    if len(sys.argv) > 1: 
        print('running problem', sys.argv[1])
        problem = int(sys.argv[1])
        run_gcln()

