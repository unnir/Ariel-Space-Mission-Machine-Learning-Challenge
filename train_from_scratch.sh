#!/bin/bash

FILE1="./data/all_test_heads.npy"
FILE2="./data/all_train_heads.npy"
FILE3="./data/all_test.npy"
FILE4="./data/all_train.npy"

if [ -f $FILE1 ] && [ -f $FILE2 ] && [ -f $FILE3 ]  && [ -f $FILE4 ]; then
   echo "Data is prepared."
else
   echo "Data needs to be prepared"
   python create_dataset.py
fi

# train a model 
python train.py

# inference 
python inference.py
