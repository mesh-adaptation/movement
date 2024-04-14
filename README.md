# Movement
## Mesh Movement in Firedrake
![GitHub top language](https://img.shields.io/github/languages/top/mesh-adaptation/movement)
![GitHub repo size](https://img.shields.io/github/repo-size/mesh-adaptation/movement)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Movement is a mesh adaptation toolkit enabling users to spatially redistribute mesh degrees of freedom using a variety of different methods from the literature, as well as tools for detecting mesh tangling.

Movement is built for use with the [Firedrake finite element library](https://firedrakeproject.org).

## Documentation

Movement has two sources of documentation:
* The [website](mesh-adaptation.github.io), which includes long-form documentation, demos, and API documentation.
* The [wiki](https://github.com/mesh-adaptation/mesh-adaptation-docs/wiki), which includes recommendations on how to interact with the codebase and development practices to be followed.

## Installation

To install Firedrake, follow the instructions on the Firedrake [download webpage](https://www.firedrakeproject.org/download.html). This will create a Python virtual environment, which Firedrake is installed into.

Activate the virtual environment and then execute the following commands:
```
cd ${VIRTUAL_ENV}/src
git clone https://github.com/mesh-adaptation/movement.git
cd movement
make install
```

The above assumes that you wish to clone the repo using the web URL. If you would prefer to clone using a password-protected SSH key then instead execute
```
cd ${VIRTUAL_ENV}/src
git clone git@github.com:mesh-adaptation/movement.git
cd movement
make install
```
