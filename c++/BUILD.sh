#!/bin/bash
mkdir build
cd build
cmake ..
make -j 4 shape_matching_dd
