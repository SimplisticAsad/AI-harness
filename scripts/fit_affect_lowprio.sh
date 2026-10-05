#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec nice -n 19 python3 -c "
import torch; torch.set_num_threads(1)
import runpy; runpy.run_path('experiments/fit_affect.py', run_name='__main__')"
