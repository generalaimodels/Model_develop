r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions. This allows the construction of stochastic computation
graphs and stochastic gradient estimators for optimization. This package
generally follows the design of the `TensorFlow Distributions`_ package.

.. _`TensorFlow Distributions`:
    https://arxiv.org/abs/1711.10604

It is not possible to directly backpropagate through random samples. However,
there are two main methods for creating surrogate functions that can be
backpropagated through. These are the score function estimator/likelihood ratio
estimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonly
seen as the basis for policy gradient methods in reinforcement learning, and the
pathwise derivative estimator is commonly seen in the reparameterization trick
in variational autoencoders. Whilst the score function only requires the value
of samples :math:`f(x)`, the pathwise derivative requires the derivative
:math:`f'(x)`. The next sections discuss these two in a reinforcement learning
example. For more details see
`Gradient Estimation Using Stochastic Computation Graphs`_ .

.. _`Gradient Estimation Using Stochastic Computation Graphs`:
     https://arxiv.org/abs/1506.05254

Score function
^^^^^^^^^^^^^^

When the probability density function is differentiable with respect to its
parameters, we only need :meth:`~torch.distributions.Distribution.sample` and
:meth:`~torch.distributions.Distribution.log_prob` to implement REINFORCE:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimizers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()

Pathwise derivative
^^^^^^^^^^^^^^^^^^^

The other way to implement these stochastic/policy gradients would be to use the
reparameterization trick from the
:meth:`~torch.distributions.Distribution.rsample` method, where the
parameterized random variable can be constructed via a parameterized
deterministic function of a parameter-free random variable. The reparameterized
sample therefore becomes differentiable. The code for implementing the pathwise
derivative would be as follows::

    params = policy_network(state)
    m = Normal(*params)
    # Any distribution with .has_rsample == True could work based on the application
    action = m.rsample()
    next_state, reward = env.step(action)  # Assuming that reward is differentiable
    loss = -reward
    loss.backward()
"""
import sys 
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[1]
sys.path.append(str(package_root))
from distributions.bernoulli import Bernoulli
from distributions.beta import Beta
from distributions.binomial import Binomial
from distributions.categorical import Categorical
from distributions.cauchy import Cauchy
from distributions.chi2 import Chi2
from distributions.constraint_registry import biject_to, transform_to
from distributions.continuous_bernoulli import ContinuousBernoulli
from distributions.dirichlet import Dirichlet
from distributions.distribution import Distribution
from distributions.exp_family import ExponentialFamily
from distributions.exponential import Exponential
from distributions.fishersnedecor import FisherSnedecor
from distributions.gamma import Gamma
from distributions.geometric import Geometric
from distributions.gumbel import Gumbel
from distributions.half_cauchy import HalfCauchy
from distributions.half_normal import HalfNormal
from distributions.independent import Independent
from distributions.inverse_gamma import InverseGamma
from distributions.kl import _add_kl_info, kl_divergence, register_kl
from distributions.kumaraswamy import Kumaraswamy
from distributions.laplace import Laplace
from distributions.lkj_cholesky import LKJCholesky
from distributions.log_normal import LogNormal
from distributions.logistic_normal import LogisticNormal
from distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from distributions.mixture_same_family import MixtureSameFamily
from distributions.multinomial import Multinomial
from distributions.multivariate_normal import MultivariateNormal
from distributions.negative_binomial import NegativeBinomial
from distributions.normal import Normal
from distributions.one_hot_categorical import OneHotCategorical, OneHotCategoricalStraightThrough
from distributions.pareto import Pareto
from distributions.poisson import Poisson
from distributions.relaxed_bernoulli import RelaxedBernoulli
from distributions.relaxed_categorical import RelaxedOneHotCategorical
from distributions.studentT import StudentT
from distributions.transformed_distribution import TransformedDistribution
from distributions.transforms import *  # noqa: F403
from distributions import transforms
from distributions.uniform import Uniform
from distributions.von_mises import VonMises
from distributions.weibull import Weibull
from distributions.wishart import Wishart

_add_kl_info()
del _add_kl_info

__all__ = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Dirichlet",
    "Distribution",
    "Exponential",
    "ExponentialFamily",
    "FisherSnedecor",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Independent",
    "InverseGamma",
    "Kumaraswamy",
    "LKJCholesky",
    "Laplace",
    "LogNormal",
    "LogisticNormal",
    "LowRankMultivariateNormal",
    "MixtureSameFamily",
    "Multinomial",
    "MultivariateNormal",
    "NegativeBinomial",
    "Normal",
    "OneHotCategorical",
    "OneHotCategoricalStraightThrough",
    "Pareto",
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    "StudentT",
    "Poisson",
    "Uniform",
    "VonMises",
    "Weibull",
    "Wishart",
    "TransformedDistribution",
    "biject_to",
    "kl_divergence",
    "register_kl",
    "transform_to",
]
__all__.extend(transforms.__all__)
