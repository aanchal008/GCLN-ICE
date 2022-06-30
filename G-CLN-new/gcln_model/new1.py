# %%
import pandas as pd
#import torch
import matplotlib.pyplot as plt
import numpy as np
# %%
x = [-1, 0 , 1 , 2 , 3, 4, 5, 6, 7, 8, 9, 10]
#output = [0.9615384615384616, 1.0, 0.9615384615384616, 0.8620689655172413, 0.7352941176470589, 0.6097560975609756, 0.5, 0.4098360655737705, 0.33783783783783783, 0.2808988764044944, 0.2358490566037736, 0.2]

def pbqu(data, c_i, eps):
    data = ((c_i) ** 2) / ((data - eps) ** 2 + (c_i) ** 2)
    return data

def ges(data):
    return pbqu(data, 5, 0)

def les(data):
    return pbqu(data, 0.5, 0)

for i in range(len(x)):
    if x[i] >= 0:
        out = ges(x[i])
        output.append(out)
    else:
        out = les(x[i])
        output.append(out)

print(output)

# %%
plt.plot(x, output)
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
plt.show()