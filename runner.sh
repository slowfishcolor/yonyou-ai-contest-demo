#!/bin/bash
set -x

if [[ -f "requirements.txt" ]] ; then
    python -m pip install -r requirements.txt
fi

python al_runner.py