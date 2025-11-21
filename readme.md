<p align="center">
  <img src="logo.png" />
</p>

[![Documentation](https://img.shields.io/badge/docs-passing-success)](https://star-age.github.io/NEST-docs/) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/star-age/NEST/blob/main/LICENSE) [![License](https://img.shields.io/badge/pypi-v0.8.0-blue)](https://pypi.org/project/astro-nest/)

**NEST** (**N**eural network **E**stimator of **S**tellar **T**imes) is a python package designed to make the use of pre-trained neural networks for stellar age estimation easy.

It is based on the upcoming paper Boin et al. 2025.

With it, you can estimate the ages of stars based on their position in the Color-Magnitude Diagram and their metallicity. It contains a suite of Neural Networks trained on different stellar evolutionary grids. If observational uncertainties are provided, it can compute age uncertainties.

[NEST Documentation](https://star-age.github.io/NEST-docs/)

# Installation :

- Using pip (preferred):
```bash
pip install astro-nest
```

- Download the source code of this repository, and run:
```bash
pip install .
```

- If you are looking for a quick way to test the Neural Networks, a web interface is also available here (click the image):

[![Web interface](website.png)](https://star-age.github.io/)

# Dependencies:
- numpy
- scikit (optional but recommended for speed)
- tqdm (optional)
- matplotlib (optional)

The `tutorial.ipynb` notebook guides you through the package usage.
