
# Probability Utilities
# This module provides utility functions for probability calculation tasks.

# Function to calculate factorial
def factorial(n):
    final_product=1
    for i in range(n,0,-1):
        final_product *= i
    return final_product

# Function to calculate combinations
def combinations(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))