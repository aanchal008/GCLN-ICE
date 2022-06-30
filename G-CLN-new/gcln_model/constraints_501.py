# Checks for program number 501

from z3 import *

x = Int('x')
y = Int('y')
x1 = Int('x1')
y1 = Int('y1')
s = Solver()
#I = And(x>=0, x <= 1000, y >= 2, Implies(x%2 ==0, x + 4 == 2*y), Implies(x%2 !=0, x + 5 == 2*y))
#I = And(1 + 2*x + -1*y >= 0)
I = And(2 + 3*x + -2*y >= 0)
I_ = substitute(I, (x, x1), (y, y1) )
logic_program = And(Implies(x % 2 ==0, y1 == y+1), Implies(x % 2 !=0, y1 == y), x1 == x + 1)
condition = (x < 200)
pre = (y <= x)
post = (y <= 2*x)
#print(logic_program.sexpr())

# Check 1
#s.add(pre, x1 == 0, y1 == 0, Not(I_))
# Check 2
#s.add(I, condition, logic_program,  Not(I_))
# Check 3
s.add(I, Not(condition), Not(post))
print(s.check())
print(s.model())