#!/bin/bash

nohup python -u main.py --ckp_name 0_$1 --device cuda:0 > 0_$1.log 2>&1 &

nohup python -u main.py --ckp_name 1_$1 --device cuda:1 > 1_$1.log 2>&1 &

nohup python -u main.py --ckp_name 2_$1 --device cuda:2 > 2_$1.log 2>&1 &

nohup python -u main.py --ckp_name 3_$1 --device cuda:3 > 3_$1.log 2>&1 &

nohup python -u main.py --ckp_name 4_$1 --device cuda:4 > 4_$1.log 2>&1 &

nohup python -u main.py --ckp_name 5_$1 --device cuda:5 > 5_$1.log 2>&1 &

nohup python -u main.py --ckp_name 6_$1 --device cuda:6 > 6_$1.log 2>&1 &

nohup python -u main.py --ckp_name 7_$1 --device cuda:7 > 7_$1.log 2>&1 &