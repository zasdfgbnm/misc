#!/bin/bash

time ptxas -arch sm_75 -m64 --generate-line-info cuda10.0.ptx -o cuda10.0.cubin
time ptxas -arch sm_75 -m64 --generate-line-info cuda11.1.ptx -o cuda11.1.cubin
