import random

data = [random.uniform(0, 10) for _ in range(100)]


def f(x):
    return x ** 2 - 3 * x + 2


def g(x):
    return x


results = [f(x) for x in data]
results2 = [g(x) for x in data]

print("Avg:", sum(results) / len(results))
print("Max:", max(results))
print("Min:", min(results))

print("")
print("Avg:", sum(results2) / len(results2))
print("Max:", max(results2))
print("Min:", min(results2))
