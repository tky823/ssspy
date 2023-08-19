#!/bin/bash

# convert .ipynb to .rst format.
jupyter nbconvert --execute notebooks/Examples/Getting-Started.ipynb --to notebook --output-dir docs/_notebooks/
