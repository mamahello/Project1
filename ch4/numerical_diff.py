import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)


def function1(x):
    return 0.01*x**2+0.1*x


x=np.arange(0.0,20.0,0.1)
y=function1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

y1=numerical_diff(function1,5)
print(y1)

y2=numerical_diff(function1,10)
print(y2)