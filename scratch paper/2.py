import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))


x=np.arange(-50,50,0.1)
y=sigmoid(x)

plt.plot(x,y)
plt.show()