#!/bin/bash

BASE_DIR=https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English

# write a for loop over train , valid , test and vocab text files and use curl to get them from the
# above base directory and save them here
for file in train.txt valid.txt test.txt vocab.txt; do
    curl -O $BASE_DIR/$file
done