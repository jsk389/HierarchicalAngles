# HierarchicalAngles
#### Code to infer the inclination angle of a star using the Bayesian Hierarchical method outlined in Kuszlewicz et al. (accepted).

[![](https://img.shields.io/badge/Github-jsk389%2FHierarchicalAngles-blue.svg)](https://github.com/jsk389/HierarchicalAngles) [![](https://img.shields.io/badge/arXiv-1907.01565-orange.svg)](https://arxiv.org/abs/1907.01565)

The purpose of this code is to infer the underlying inclination angle of a star given a set of noise measurements (posterior distributions) of the inclination angle, for example, from individual mixed modes in a red giant.

This repository contains some example data in the form of inclination angle posterior distributions in the ExampleData folder. Examples are given for a low, intermediata and high inclination angle star, all of which are artificial data generated for the paper. The inclination angle used in generating the data is given in the inc_angle.txt file.

## Installation

The Fisher distribution used as the model requires the use of numerical integration to normalise the distribution. This is done quickly by numerical integration and uses the cython module `integrand.pyx`. As a result before running any code please run the `setup.py` file according to

```bash
python setup.py install
```

which will install all the necessary functions in the module and compile the cython code.

## Quick Example

To run and example, you can run the `example.py` file with an accompanying system argument that gives the specific example data to use. This can be a choice of `ExampleLow`, `ExampleMiddle` and `ExampleHigh` for the low, intermediate and high inclination angle artificial star respectively. In other words,

```bash
python example.py ExampleLow
```
would run the code over the low inclination angle star.

A folder will be created called `ExampleRun` which contains the saved chains from the run. The rest of the information is printed to the terminal or plotted.


