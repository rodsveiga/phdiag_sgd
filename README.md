# Phase diagram of Stochastic Gradient Descent in high-dimensional two-layer neural networks

*Under construction*.

## Description

Code for *Phase diagram of Stochastic Gradient Descent in high-dimensional two-layer neural networks*.

<p float="center">
  <img src="https://github.com/rodsveiga/phdiag_sgd/blob/main/figures/arXiv_fig01_image.jpg" height="350">
</p>


## Prerequisites
- [python](https://www.python.org/) >= 3.6
- [cython](https://cython.readthedocs.io/en/latest/#)

## Structure

In this repository we provide the code and some guided example to help the reader to reproduce the figures. The repository is structured as follows.

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```/sim``` | Description      |
| ```/ode``` | Desciption 2                                |

The notebooks are self-explanatory.

## Building Cython code

Both ```/sim``` and ```/ode``` use Cython code. To build, run `python setup.py build_ext --inplace` on the respective folder. Then simply start a Python session and do whether `from sim import sim` or `from ode import ode` and use the imported function as described in the `how_to.ipynb` notebooks.
