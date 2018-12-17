# ML_for_kirigami_design
Python package to model and to perform topology optimization for graphene kirigami using deep learning. We specically use convolutional neural networks (similar to VGGNet architecure) for regression. 

## Paper 
See our published paper: <a href="https://arxiv.org/abs/1808.06111" style="color:#268cd7
"> **P. Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, Accelerated search and design of stretchable graphene kirigami using machine learning (arXiv preprint arXiv:1808.06111).</a> Our paper is recently accepted to Physical Review Letters!

This package is still under developement. 

## General usage 
1. A simple jupyter notebook to perform predictions with scikit-learn package is avalaible in `models/simple/simple_machine_learning.ipynb`

2. A jupyter notebook to generate atomic configurations for LAMMPS input file is avalaible in `generate_LAMMPS_input/generate_LAMMPS_configuration_input.ipynb`

3. A python code to perform regression with TensorFlow is avalaible in `models/regression_CNN/tf_fgrid_dnn_validtrain.py`


More features (e.g., search algorithm with TensorFlow code) will be added soon.

## Authors
Paul Hanakata

## Citation

If you use this package/code please cite 
```bash
@article{hanakata2018accelerated,
title={Accelerated search and design of stretchable graphene kirigami using machine learning},
author={Hanakata, Paul Z and Cubuk, Ekin D and Campbell, David K and Park, Harold S},
journal={arXiv preprint arXiv:1808.06111},
year={2018}}
```
## References
<a href="https://arxiv.org/abs/1808.06111" style="color:#268cd7
"> **P. Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, Accelerated search and design of stretchable graphene kirigami using machine learning (arXiv preprint arXiv:1808.06111).</a>
