# SITE: Symbolic Identification of Tensor Equations

This is an open-source repository of the Python code used in the paper [*Symbolic identification of tensor equations in multidimensional physical fields.*](https://arxiv.org/abs/2507.01466) arXiv preprint arXiv:2507.01466 (2025).

Some of the code implementations refer to the following repositories:
- [*geppy: a gene expression programming framework in Python*](https://github.com/ShuhuaGao/geppy)
- [*Dimensional homogeneity constrained gene expression programming*](https://github.com/Wenjun-Ma/DHC-GEP)
- [*EQDiscovery*](https://github.com/isds-neu/EQDiscovery)



## Overview
Recently, data-driven methods have shown great promise for discovering governing equations from simulation or experimental data. However, most existing approaches are limited to scalar equations, with few capable of identifying tensor relationships. In this work, we propose a general data-driven framework for identifying tensor equations, referred to as Symbolic Identification of Tensor Equations (SITE). The core idea of SITE—representing tensor equations using a host–plasmid structure—is inspired by the multidimensional gene expression programming (M-GEP) approach. To improve the robustness of the evolutionary process, SITE adopts a genetic information retention strategy. Moreover, SITE introduces two key innovations beyond conventional evolutionary algorithms. First, it incorporates a dimensional homogeneity check to restrict the search space and eliminate physically invalid expressions. Second, it replaces traditional linear scaling with a tensor linear regression technique, greatly enhancing the efficiency of numerical coefficient optimization. We validate SITE using two benchmark scenarios, where it accurately recovers target equations from synthetic data, showing robustness to noise and small sample sizes. Furthermore, SITE is applied to identify constitutive relations directly from molecular simulation data, which are generated without reliance on macroscopic constitutive models. It adapts to both compressible and incompressible flow conditions and successfully identifies the corresponding macroscopic forms, highlighting its potential for data-driven discovery of tensor equation.

As illustrated below, the process of symbolic identification using SITE begins with data preparation and preprocessing. A terminal library for both tensors and scalars is then constructed based on these physical quantities. Using this library and a predefined set of symbolic operators, SITE generates candidate equations by composing various symbolic structures, which are iteratively optimized through an evolutionary workflow.
![overview](overview.jpg)

## Hardware requirements
This work was done with

- a workstation equipped with a 13th Gen Intel®Core™ i9-13900K CPU at 3.00 GHz, 32 GB RAM, and running a 64-bit operating system

## Dependencies

The list of software this work has used is
- Python 3.11
- geppy
- deap
- numpy
- scipy
- sympy

We strongly recommend using conda or pip to create a virtual environment and install the corresponding packages.

## How to run our cases?
Most datas are included in the directory. The scripts are in the corresponding dictionaries. One can run the desired scripts with python IDE.

## Coming soon...
A demo notebook will be soon uploaded for understanding SITE.
