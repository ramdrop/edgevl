#!/bin/bash

# Get the first argument
first_argument=$1

# Shift the first argument so that "$@" contains the rest
shift

# Run trtexec with the first argument and append the rest of the arguments
trtexec --separateProfileRun --iterations=100 --useCudaGraph --duration=0 --onnx="${first_argument}.onnx" --saveEngine="engines/${first_argument}.trt" --dumpProfile --verbose "$@" 2>&1 | tee "logs/${first_argument}.log"

