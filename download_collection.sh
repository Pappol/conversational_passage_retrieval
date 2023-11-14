#!/bin/bash

mkdir data

wget --output-document data/msmarco-passage.tar.gz https://gustav1.ux.uis.no/dat640/msmarco-passage.tar.gz
tar -xzvf data/msmarco-passage.tar.gz -C data/
rm msmarco-passage.tar.gz
