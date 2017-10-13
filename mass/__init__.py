"""
Mass
====

MCMC Massive Sampler. A package to do Bayesian inference in hiearchical
models using Markov Chain Monte Carlo sampling. It was created with the
aim of dealing with very large models with thousands of parameters. To
this end likelihoods and priors are computed in parallel, either locally
or in a computer cluster. This permits a performance gain relative to
other Python packages when dealing with large models. Another objective
was to make it as easy as possible to use.

Features
--------

* Perform MCMC sampling for very large Bayesian hierachical models, 
with function computations done in parallel.
* MCMC steps can be done in blocks or cycling the parameters one-at-a-time
* Automatic tuning of the proposal distribution
* Option to do parallel tempering: run several chains per parameter block
with different temperatures and do coupling updates

Usage
-----

The first step is always to define the hiearchical model. For example,
if the likelihood, prior and hyperprior functions are defined in the
`foo` module:

```python
import numpy as np
import foo
import mass

p0 = np.ones((4,3))
h0 = np.ones(1)
data = np.zeros(4)

f1 = foo.likelihood
f2 = foo.prior
f3 = foo.hyperprior

model1 = mass.Model(p0, h0, f1,f2,f3, data, "cluster.cfg",  depends=["foo"])
```

The `cluster.cfg` is a configuration file defining parameters needed for
parallel configuration.

To obtain MCMC samples from the posterior use the `sample` method of
`Mcmc`class:

```python
mcmc = mass.Mcmc(model1)

mcmc.sample(5000, p0, h0, thin=10, burnin=500, sampling_method="cycle")
```

This creates 500 samples after 5000 iterations thinned by 10 and a burnin
period of 500 iterations, using the "cycle" sampling method.

To do parallel tempering define a list of temperatures and make sure to
increase the size of p0 to include the extra walkers (one extra walker per
termperature):

```python
p0 = np.ones((8,3))

mcmc.sample(5000, p0, h0, thin=10, burnin=500, sampling_method="cycle", temperatures=[1., 0.5], tempering=True)
```
"""

from mass.mass import Model, Mcmc
__version__ = "0.1"