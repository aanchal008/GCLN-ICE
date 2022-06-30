# Checks for program number 500 which has invariant x = y
from z3 import *

x = Int('x')
y = Int('y')
x1 = Int('x1')
y1 = Int('y1')
s = Solver()
I = And(y - 2*x - 1 >= 0)
I_ = substitute(I, (x, x1), (y, y1) )
logic_program = And(y1 == 2*x + 1, x1 == x + 1)
condition = x<100
post = (y >= 2*x)
print(logic_program.sexpr())

# Check 1
s.add((x1 == 1), (y1 == 3), Not(I))
# Check 2
#s.add(I, condition, logic_program,  Not(I_))
# Check 3
#s.add(I, Not(condition), Not(post))
print(s.check())
print(s.model())