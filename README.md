# testperanto
### a python package for generating artificial language

#### Quickstart (with Anaconda)

    conda create -n tpo python=3.8
    conda activate tpo
    pip install -e .

#### To run the unit tests (from testperanto root directory)

    python -m unittest

#### To generate sentences from an example JSON config (from testperanto root directory)

    python scripts/generate.py -c examples/white/white.json --sents -n 5 -s 011101




