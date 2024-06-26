#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh

function fail {
    echo FAIL: $@
    exit -1
}

export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}

conda activate pytorch-ci

pushd dglgo
rm -rf build *.egg-info dist
pip uninstall -y dglgo
python3 setup.py install
popd

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

python -m pip install psutil || fail "pip install"
python3 -m pytest -v --junitxml=pytest_go.xml --durations=100 tests/go/test_model.py || fail "go"
