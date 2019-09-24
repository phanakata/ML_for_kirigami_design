# ML_for_kirigami_design
Python package to model and to perform topology optimization for graphene kirigami using deep learning. We use convolutional neural networks (similar to VGGNet architecure) for regression. 

## Paper 
See our published paper: <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.255304" style="color:#268cd7
"> **P. Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, Accelerated search and design of stretchable graphene kirigami using machine learning, *Phys. Rev. Lett*, 121, 255304  (2018).</a>

## General usage 
1. A python code to perform regression with TensorFlow is avalaible in `models/regression_CNN/tf_fgrid_dnn_validtrain.py`

2. A jupyter notebook to generate atomic configurations for <a href="https://lammps.sandia.gov/" style="color:#268cd7
">LAMMPS</a> input file is avalaible in `generate_LAMMPS_input/generate_LAMMPS_configuration_input.ipynb`

3. A simple jupyter notebook to perform predictions with scikit-learn package is avalaible in `models/simple/simple_machine_learning.ipynb`

4. A simple jupyter notebook to convert coarse-grained dataset to fine-grid dataset is avalaible in `models/regression_CNN/convert_coarse_to_fine.ipynb`

5. Raw dataset of coarse-grained grid can be found in `mddata`. This dataset generated using AIREBO potential with 1.7 mincutoff which is the default of CH.airebo. 


This package is still under developement. More features (e.g., search algorithm with TensorFlow code) will be added soon.

## Authors
Paul Hanakata

## Citation

If you use this package/code/dataset or find our research is useful for your work please cite 
```bash
@article{hanakata-PhysRevLett.121.255304,
  title = {Accelerated Search and Design of Stretchable Graphene Kirigami Using Machine Learning},
  author = {Hanakata, Paul Z. and Cubuk, Ekin D. and Campbell, David K. and Park, Harold S.},
  journal = {Phys. Rev. Lett.},
  volume = {121},
  issue = {25},
  pages = {255304},
  numpages = {6},
  year = {2018},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.121.255304},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.121.255304}
}
```
## References
* <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.255304" style="color:#268cd7">[1] **Paul Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, *Phys. Rev. Lett*, 121, 255304  (2018).</a>
* <a href="https://arxiv.org/abs/1808.06111" style="color:#268cd7">[2] **P. Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, Accelerated search and design of stretchable graphene kirigami using machine learning (arXiv preprint arXiv:1808.06111).</a>
