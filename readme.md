<p align="center">
  <img src="logo.png" />
</p>

[![Documentation](https://img.shields.io/badge/docs-passing-success)](https://star-age.github.io/NEST-docs/) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/star-age/NEST/blob/main/LICENSE) [![License](https://img.shields.io/badge/pypi-v1.0.5-blue)](https://pypi.org/project/astro-nest/)

**NEST** (**N**eural network **E**stimator of **S**tellar **T**imes) is a python package designed to make the use of pre-trained neural networks for stellar age estimation easy.

It is based on [Boin et al. 2026](https://www.aanda.org/articles/aa/full_html/2026/04/aa58436-25/aa58436-25.html).

If you use **NEST** for your research, please acknowledge this by citing

    @article{ Boin26,
        author = {{Boin, T.} and {Casamiquela, L.} and {Haywood, M.} and {Di Matteo, P.} and {Lebreton, Y.} and {Uddin, M.} and {Reese, D. R.}},
        title = {Stellar age determination using deep neural networks - Isochrone ages for 1.3 million stars, based on BaSTI, MIST, PARSEC, Dartmouth, and SYCLIST evolutionary grids},
        DOI= "10.1051/0004-6361/202558436",
        url= "https://doi.org/10.1051/0004-6361/202558436",
        journal = {A\&A},
        year = 2026,
        volume = 708,
        pages = "A215",
    }

You can download the BibTeX citation file here: [NEST.bib](https://raw.githubusercontent.com/star-age/NEST/refs/heads/main/NEST.bib)

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
