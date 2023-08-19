#!/bin/bash

# TODO: unify .readthedocs.yaml
pip install -e ".[docs,notebooks]"

# convert .ipynb to .rst format.
jupyter nbconvert --execute notebooks/Examples/Getting-Started.ipynb --to notebook --output-dir docs/_notebooks/
