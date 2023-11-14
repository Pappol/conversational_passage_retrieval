#!/bin/bash

mkdir data

wget --output-document data/msmarco-passage.tar.gz https://gustav1.ux.uis.no/dat640/msmarco-passage.tar.gz
tar -xzvf data/msmarco-passage.tar.gz -C data/
rm msmarco-passage.tar.gz

wget --output-document data/queries_train.csv https://stavanger.instructure.com/courses/12872/files/1528117/download?download_frd=1
wget --output-document data/queries_test.csv https://stavanger.instructure.com/courses/12872/files/1528116/download?download_frd=
wget --output-document data/qrels_train.txt https://stavanger.instructure.com/courses/12872/files/1536029/download?download_frd=1
