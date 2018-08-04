#!/bin/bash
echo "data/LC_dewet_prune2_w0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 2 "data/features_dewetting_woclass1.csv"  1> data/pruning2_dewetting_w0.txt
echo "data/LC_dewet_prune3_w0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 3 "data/features_dewetting_woclass1.csv"  1> data/pruning3_dewetting_w0.txt
echo "data/LC_dewet_prune4_w0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 4 "data/features_dewetting_woclass1.csv"  1> data/pruning4_dewetting_w0.txt
echo "data/LC_dewet_prune5_w0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 5 "data/features_dewetting_woclass1.csv"  1> data/pruning5_dewetting_w0.txt
echo "data/LC_dewet_prune6_w0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 6 "data/features_dewetting_woclass1.csv"  1> data/pruning6_dewetting_w0.txt
echo "data/LC_dewet_prune7_w0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 7 "data/features_dewetting_woclass1.csv"  1> data/pruning7_dewetting_w0.txt
echo "data/LC_dewet_prune8_w0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 8 "data/features_dewetting_woclass1.csv"  1> data/pruning8_dewetting_w0.txt

echo "data/LC_dewet_prune2_wo0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 2 "data/features_dewetting_woclass0,1.csv"  1> data/pruning2_dewetting_wo0.txt
echo "data/LC_dewet_prune3_wo0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 3 "data/features_dewetting_woclass0,1.csv"  1> data/pruning3_dewetting_wo0.txt
echo "data/LC_dewet_prune4_wo0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 4 "data/features_dewetting_woclass0,1.csv"  1> data/pruning4_dewetting_wo0.txt
echo "data/LC_dewet_prune5_wo0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 5 "data/features_dewetting_woclass0,1.csv"  1> data/pruning5_dewetting_wo0.txt
echo "data/LC_dewet_prune6_wo0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 6 "data/features_dewetting_woclass0,1.csv"  1> data/pruning6_dewetting_wo0.txt
echo "data/LC_dewet_prune7_wo0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 7 "data/features_dewetting_woclass0,1.csv"  1> data/pruning7_dewetting_wo0.txt
echo "data/LC_dewet_prune8_wo0.png" | python gridsearch_randomforest.py -k 10 -n 50 -P --seed 1 -D 8 "data/features_dewetting_woclass0,1.csv"  1> data/pruning8_dewetting_wo0.txt

