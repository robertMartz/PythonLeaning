import math

def circ_area(radius):
    return math.pi * radius**2

def sphere_volume(radius):
    return ( 4*math.pi*radius**3 ) / 3

def avg(numbers):
    total = 0
    for n in numbers:
        total += n
    return total/len(numbers)

def geometric_sum(end):
    total = 0
    j = 1
    while j  <= end:
        total += 1/j
        j += 1
    return total

def geometric():
    r = float(input("\nEnter radius: "))
    print(f"Circle area of radius {r}: {circ_area(r):.2f}")
    print(f"Sphere volume of radius {r}: {sphere_volume(r):.2f}")

    endpoint = int(input("\nEnter endpoint of geometric series: "))
    print(f"Geometric series sum from i=1 to i={endpoint} is : {geometric_sum(endpoint):.2f}")

def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False

def tell_parity():
    n = int(input("\nEnter number: "))
    if is_even(n):
        print("Your number is Even")
    else:
        print("Your number is Odd")

def abs_value():
    n = float(input("\nEnter any number: "))
    if n < 0:
        n = -n
    print(f"The absolute value of your number is: {n}")


def sqrt_approx(a):
    x = a

    for i in range(10):
        x = 0.5 * (x + a / x)

    return x

def sqrt():
    a = float(input("\nEnter any number: "))
    approx = sqrt_approx(a)
    real = math.sqrt(a)

    print(f"The approximate sqrt value of your number is: {approx:.9f}")
    print(f"The real sqrt value of your number is: {real:.9f}")

    error = abs(approx - real)

    print(f"Error: {error}")
    print(f"Relative percentual error: {(error/real)*100:.9f} %")