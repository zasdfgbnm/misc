#!/bin/bash

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/gaoxiang/Downloads/libtorch ..
cmake --build . --config Release
