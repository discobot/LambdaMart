LambdaMart
==========

A small, readable LambdaMART implementation in Python: gradient-boosted
regression trees trained on LambdaRank gradients (NDCG-based pairwise
pseudo-responses), with the Newton-step leaf values from the paper.

Reference: Burges, [*From RankNet to LambdaRank to LambdaMART: An
Overview*](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) (2010).

Usage
-----

Input is a CSV with a header row and columns
`label, query_id, doc_id, feature_1, ..., feature_n`, with all rows of a
query contiguous.

```
pip install numpy scikit-learn
python mart.py -t train.csv -p test.csv        # writes result.csv
python mart.py -t train.csv -n 100 -r 0.05     # 100 trees, lr 0.05
```
