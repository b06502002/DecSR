multiply = lambda x, y: x*y
print(multiply(4,2))


def a(i):
    return lambda a: a*i**2
b = a(2)
print(b(9))