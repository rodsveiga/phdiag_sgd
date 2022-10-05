# Phase diagram of Stochastic Gradient Descent in high-dimensional two-layer neural networks

## Description

Repository for the paper [*Phase diagram of Stochastic Gradient Descent in high-dimensional two-layer neural networks*](https://arxiv.org/abs/2202.00293). To appear in [NeurIPS 2022](https://nips.cc/).

<p float="center">
  <img src="https://github.com/rodsveiga/phdiag_sgd/blob/main/figures/arXiv_fig01_image.jpg" height="350">
</p>


## Prerequisites
- [python](https://www.python.org/) >= 3.6
- [cython](https://cython.readthedocs.io/en/latest/#)

## Structure

In this repository we provide the code and some guided examples to help the reader to reproduce the figures. The repository is structured as follows.

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```/sim``` | ```sim.py``` is the simulation class, which imports cython code from ```simcy.pyx```. ```setup.py``` is an auxiliar buinding file for cython   |
| ```/ode``` | ```ode.py``` is the ODE solver class, which imports cython code from ```odecy.pyx```. ```setup.py``` is an auxiliar buinding file for cython |                              |

The notebooks are self-explanatory.

## Building cython code

Both ```/sim``` and ```/ode``` use cython code. To build, run `python setup.py build_ext --inplace` on the respective folder. Then simply start a python session and do whether `from sim import sim` or `from ode import ode` and use the imported function as described in the `how_to.ipynb` notebooks.

## Reference

- *Phase diagram of Stochastic Gradient Descent in high-dimensional two-layer neural networks*; R. Veiga, L. Stephan, B. Loureiro, F. Krzakala, L. Zdeborová; [arXiv:2202.00293](https://arxiv.org/abs/2202.00293) [stat.ML]
