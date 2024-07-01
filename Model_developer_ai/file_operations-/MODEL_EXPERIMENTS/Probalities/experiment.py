import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from typing import Tuple

def plot_beta(a: int, b: int, n: int, h: int) -> None:
    """
    Plot the prior and posterior Beta distributions.

    Parameters:
    a (int): Parameter 'a' of the prior Beta distribution.
    b (int): Parameter 'b' of the prior Beta distribution.
    n (int): Total number of coin tosses.
    h (int): Number of heads observed.

    Returns:
    None
    """
    # Define the range of p
    p = np.linspace(0, 1, 1000)

    # Compute the prior and posterior Beta distributions
    prior = beta.pdf(p, a, b)
    posterior = beta.pdf(p, a + h, b + n - h)

    # Plot the prior and posterior distributions
    plt.figure(figsize=(10, 6))
    plt.plot(p, prior, label='Prior Beta({}, {})'.format(a, b))
    plt.plot(p, posterior, label='Posterior Beta({}, {})'.format(a + h, b + n - h))
    plt.xlabel('p')
    plt.ylabel('Density')
    plt.title('Prior and Posterior Distributions')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the function
plot_beta(a=2, b=2, n=10, h=6)
