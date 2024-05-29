from py_ecc.bn128 import G1, G2, multiply, add, curve_order, eq, Z1, pairing
import numpy as np
import galois
from functools import reduce


##########################################################
#                   Initial Equations and Constraints    #
##########################################################
# The goal is to prove that we know an `x` and `z` to solve:
#    x^3 + 4x^2 - xz + 4 = 529
#
# This formula can be broken down into three constraints:
#    1) x_squared = x * x
#    2) x_cubed = x * x_squared
#    3) result - x_cubed - 4 * x_squared - 4 = -xz
##########################################################



# Function to print and log messages
def log(message):
    print(message)

##########################################################
#                  Initialize Galois Field               #
##########################################################

log("Initializing a large field, this may take a while...")
GF = galois.GF(curve_order)

##########################################################
#                  Define R1CS Matrices                  #
##########################################################

O = np.array([
    [0, 0, 1, 0, 0, 0],  # LHS of 1) x^2
    [0, 0, 0, 1, 0, 0],  # LHS of 2) x^3
    [-4, 0, -4, -1, 0, 1]  # LHS of 3) y - x^3 - 4x^2 - 4
])

L = np.array([
    [0, 1, 0, 0, 0, 0],  # LHARG of 1) x
    [0, 0, 1, 0, 0, 0],  # LHARG of 2) x^2
    [0, -1, 0, 0, 0, 0]  # LHARG of 3) -x
])

R = np.array([
    [0, 1, 0, 0, 0, 0],  # RHARG of 1) x
    [0, 1, 0, 0, 0, 0],  # RHARG of 2) x
    [0, 0, 0, 0, 1, 0]  # RHARG of 3) z
])

##########################################################
#           Test R1CS Matrices with Witness Values       #
##########################################################

x = 4
z = 3
v1 = x * x
v2 = v1 * x 
out = v2 + 4 * v1 - x * z + 4  

witness = np.array([1, x, v1, v2, z, out])
log(f"Witness: {witness}")

assert all(np.equal(np.matmul(L, witness) * np.matmul(R, witness), np.matmul(O, witness))), "R1CS constraints not satisfied"
log("R1CS constraints are satisfied.")

##########################################################
#          Convert Matrices to Galois Field              #
##########################################################

L_galois = GF((L + curve_order) % curve_order)
R_galois = GF((R + curve_order) % curve_order)
O_galois = GF((O + curve_order) % curve_order)

x = GF(4)
z = GF(3)

v1 = x * x
v2 = v1 * x  
out = v2 + GF(4) * v1 - x * z + GF(4)

witness = GF(np.array([1, x, v1, v2, z, out]))
log(f"Witness in Galois field: {witness}")

assert all(np.equal(np.matmul(L_galois, witness) * np.matmul(R_galois, witness), np.matmul(O_galois, witness))), "R1CS constraints not satisfied in Galois field"
log("R1CS constraints are satisfied in Galois field.")

##########################################################
#        Convert R1CS to QAP using Lagrange              #
##########################################################

def interpolate_column(col):
    xs = GF(np.array([1, 2, 3]))
    return galois.lagrange_poly(xs, col)

U_polys = np.apply_along_axis(interpolate_column, 0, L_galois)
V_polys = np.apply_along_axis(interpolate_column, 0, R_galois)
W_polys = np.apply_along_axis(interpolate_column, 0, O_galois)

def inner_product_polynomials_with_witness(polys, witness):
    mul_ = lambda x, y: x * y
    sum_ = lambda x, y: x + y
    return reduce(sum_, map(mul_, polys, witness))

term_1 = inner_product_polynomials_with_witness(U_polys, witness)
term_2 = inner_product_polynomials_with_witness(V_polys, witness)
term_3 = inner_product_polynomials_with_witness(W_polys, witness)

##########################################################
#            Define Polynomial t(x) and h(x)             #
##########################################################

t = galois.Poly([1, curve_order - 1], field=GF) * galois.Poly([1, curve_order - 2], field=GF) * galois.Poly([1, curve_order - 3], field=GF)

h = (term_1 * term_2 - term_3) // t

assert term_1 * term_2 == term_3 + h * t, "Division has a remainder"
log("QAP conversion and division are correct.")

##########################################################
#        Compute Inner Product for EC Points AND         #
#                    POWER OF TAU CEREMONY               #
##########################################################

def inner_product(ec_points, coeffs):
    return reduce(add, (multiply(point, int(coeff)) for point, coeff in zip(ec_points, coeffs)), Z1)

def generate_powers_of_tau(tau, degree, curve):
    return [multiply(curve, int(tau ** i)) for i in range(degree + 1)]

tau = GF(8)
powers_of_tau_term_1 = generate_powers_of_tau(tau, term_1.degree, G1)
powers_of_tau_term_2 = generate_powers_of_tau(tau, term_2.degree, G2)
powers_of_tau_term_3 = generate_powers_of_tau(tau, term_3.degree, G1)
powers_of_tau_term_4 = generate_powers_of_tau(tau, (h * t).degree, G1)

evaluate_on_ec_term_1 = inner_product(powers_of_tau_term_1, term_1.coeffs[::-1])
evaluate_on_ec_term_2 = inner_product(powers_of_tau_term_2, term_2.coeffs[::-1])
evaluate_on_ec_term_3 = inner_product(powers_of_tau_term_3, term_3.coeffs[::-1])
evaluate_on_ec_term_4 = inner_product(powers_of_tau_term_4, (h * t).coeffs[::-1])

##########################################################
#              Compute Points A, B, and C                #
##########################################################

A = evaluate_on_ec_term_1
B = evaluate_on_ec_term_2
C = add(evaluate_on_ec_term_3, evaluate_on_ec_term_4)

proof = [A, B, C]

##########################################################
#                   Verify the Proof                     #
##########################################################

result = pairing(proof[1], proof[0]) == pairing(G2, proof[2])
log(f"e(A, B) == e(C, G2[1]): {result}")
log(f"The proof is valid? {result}")
