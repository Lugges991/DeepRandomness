# Numerical Integration using Monte Carlo method
# FB - 201006137
import math
import random
import numpy as np
# define any function here!


def f(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(1 / 2) * np.square(x))


# define any xmin-xmax interval here! (xmin < xmax)
xmin = 0.0
xmax = 1

# find ymin-ymax
numSteps = 1000000  # bigger the better but slower!
ymin = f(xmin)
ymax = ymin
for i in range(numSteps):
    x = xmin + (xmax - xmin) * float(i) / numSteps
    y = f(x)
    if y < ymin:
        ymin = y
    if y > ymax:
        ymax = y

# Monte Carlo
rectArea = (xmax - xmin) * (ymax - ymin)
numPoints = 1000000  # bigger the better but slower!
ctr = 0
for j in range(numPoints):
    x = xmin + (xmax - xmin) * random.random()
    y = ymin + (ymax - ymin) * random.random()
    if math.fabs(y) <= math.fabs(f(x)):
        if f(x) > 0 and y > 0 and y <= f(x):
            ctr += 1  # area over x-axis is positive
        if f(x) < 0 and y < 0 and y >= f(x):
            ctr -= 1  # area under x-axis is negative

fnArea = rectArea * float(ctr) / numPoints
print("Numerical integration = " + str(fnArea))
