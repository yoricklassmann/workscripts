#!/bin/bash
module load intel
module load mkl
module load openmpi
export WFOVERLAP=/ddn/home/dlqg54/CODE_AND_SCRIPTS/wfoverlap/bin/wfoverlap.x 
python tden.py

