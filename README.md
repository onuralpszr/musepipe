<p align="center">
  <img style="width:40%;" src="logo/musepipelogo.png" />
</p>

<h1 align="center">MusePipe</h1>

<h3 align="center">Musepipe is mediapipe based on library.</h3>
<br>

![GitHub](https://img.shields.io/github/license/onuralpszr/musepipe?color=green)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/onuralpszr/musepipe/main.svg)](https://results.pre-commit.ci/latest/github/onuralpszr/musepipe/main)

<p>
  <img alt="Python38" src="https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="Python39" src="https://img.shields.io/badge/Python-3.9-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="Python310" src="https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="Cuda" src="https://img.shields.io/badge/Cuda-Enabled-76B900.svg?logo=Nvidia&logoColor=white"/>
  <img alt="Poetry" src="https://img.shields.io/badge/Poetry-60A5FA.svg?logo=Poetry&logoColor=white"/>
  <img alt="Black" src="https://img.shields.io/badge/code%20style-black-black"/>
  <img alt="Mypy" src="https://img.shields.io/badge/mypy-checked-blue"/>
  <img alt="isort" src="https://img.shields.io/badge/isort-checked-yellow"/>
</p>

## ⚙️ Requirements ⚙️

* Python 3.8  Python3.10 (Virtualenv recommended)
* Optional: Poetry
* Optional: Nvidia CUDA for cuda usage

## 🛠️ Installation 🛠️

### Pip installation

```bash
python -m venv .venv
pip install -r requirements.txt
source .venv/bin/activate
```

### Poetry installation

```bash
poetry shell
poetry install
```

### Pre-commit .git/hook installation

* Prerequisites pip or poetry installation

```bash
pre-commit install -f --config .pre-commit-config.yaml
```

### Feature list
TBH
