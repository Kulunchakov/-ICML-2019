#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch2/clear/mairal/intel/compilers_and_libraries/linux/lib/intel64/:/scratch2/clear/mairal/intel/compilers_and_libraries/linux/mkl/lib/intel64/:/scratch2/clear/mairal/cuda-9.0/lib64/
export LIB_INTEL=/scratch2/clear/mairal/intel/compilers_and_libraries/linux/lib/intel64/
export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
export LD_PRELOAD=$LIB_INTEL/libimf.so:$LIB_INTEL/libintlc.so.5:$LIB_INTEL/libiomp5.so:$LIB_INTEL/libsvml.so:/usr/lib/gcc/x86_64-linux-gnu/5//libstdc++.so
/softs/stow/matlab-2016a/bin/matlab -nodisplay -singleCompThread -r "addpath('./mex/'); "
