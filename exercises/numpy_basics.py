import numpy as np
import math

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
print(a * b)

x = np.array([0, 1, 2, 3, 4])

# x^2 + 2x + 1
print(x**2 + 2*x + 1)


# Crea un array de 100 puntos entre 0 y 10: (extremos incluidos)
x = np.linspace(0,10, 100)
print(x)

# Vectorización
# f(x)=sin(x)+x^2

# Con numpy
y = np.sin(x) + x**2

# Usando Python vanilla
list_y = [ math.sin(yi) + yi**2 for yi in x]

print(y[:10])
print(list_y[:10])




























