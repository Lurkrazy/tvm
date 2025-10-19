#!/bin/bash
echo "there is a bug"

python3 bench_ms_cline.py --model resnet18 --batch-size 1 --target "cuda" --device "cuda" --trials 0 --number 10 --repeat 3 --warmup 3