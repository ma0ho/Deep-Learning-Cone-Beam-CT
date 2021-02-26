# Deep Learning Cone Beam Computed Tomography

This repository provides an implementation of a Cone Beam Backprojection for Tensorflow. It can be used to data-driven learn preprocessing steps in the projection domain. As an example, we include the training of redundancy weights, as reported with the accompanying paper.

## How to build
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=[...] ..
make && make install
```

## References

In case you use this code for your own scientific work, please cite the following publication:

> Würfl, Tobias, et al. "Deep learning computed tomography: Learning projection-domain weights from image domain in limited angle problems." IEEE transactions on medical imaging 37.6 (2018): 1454-1463.

BibTeX:

<details>

```bibtex
@article{wurfl2018deep,
  title={Deep learning computed tomography: Learning projection-domain weights from image domain in limited angle problems},
  author={W{\"u}rfl, Tobias and Hoffmann, Mathis and Christlein, Vincent and Breininger, Katharina and Huang, Yixin and Unberath, Mathias and Maier, Andreas K},
  journal={IEEE transactions on medical imaging},
  volume={37},
  number={6},
  pages={1454--1463},
  year={2018},
  publisher={IEEE}
}
```

</details>

