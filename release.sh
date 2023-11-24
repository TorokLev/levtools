#!/bin/bash

set -e

pytest tests

rm dist/ -Rf
python3 -m build --wheel
