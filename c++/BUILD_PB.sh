#!/bin/bash
mkdir smpb
cd smpb
cmake .. -DBUILD_PYTHON_BINDINGS=True
make -j 8 shape_match_model_pb
