# conversational_passage_retrieval

# Running the baseline

```
python3 baseline.py
```

## Environment setup

To install from environment file run:
```
conda env create -f environment.yml
```

# Evaluation

For evaluation we are using the TREC eval tool.

```
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
./trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 qrels_train.txt {YOUR_TREC_RUNFILE}
```
