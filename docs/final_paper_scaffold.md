# Final Paper Scaffold

## Abstract
We study music taste matchmaking as a user-user recommendation problem using the HetRec 2011 Last.fm dataset. The system ranks candidate users for a target user using listening histories, tags, niche artist overlap, and discovery potential. We compare simple baselines, an interpretable hybrid compatibility score, a novel dual-space complementary model, and a reliability-filtered learned ranker trained on held-out friend links.

## Introduction
Explain why music taste can support social recommendation, not just song recommendation. State the retrieval task and the novelty axis.

## Related Work
Use the three proposal papers: graph bottlenecked social recommendation, attribute-aware music personalization, and popularity-bias mitigation in music recommenders.

## Methodology
Describe preprocessing, friend-link splitting, sampled-negative evaluation, the filtered taste-aligned ground-truth analysis, baselines, proposal hybrid scoring, the dual-space complementary model, and learned ranker.

## Experiments and Results
Include this table:

| model | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 | map@5 | map@10 | auc | evaluated_targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dual_space_complementary | 0.2581 | 0.1921 | 0.4518 | 0.6283 | 0.4276 | 0.4823 | 0.3373 | 0.3596 | 0.9006 | 1126.0000 |
| teammate_learned_ranker | 0.2361 | 0.1795 | 0.4050 | 0.5701 | 0.3923 | 0.4452 | 0.3071 | 0.3289 | 0.8807 | 1126.0000 |
| artist_cosine | 0.1961 | 0.1473 | 0.3510 | 0.4824 | 0.3317 | 0.3736 | 0.2557 | 0.2701 | 0.8209 | 1126.0000 |
| proposal_hybrid | 0.1933 | 0.1467 | 0.3383 | 0.4776 | 0.3213 | 0.3656 | 0.2445 | 0.2604 | 0.8320 | 1126.0000 |
| svd_embedding | 0.1858 | 0.1463 | 0.3238 | 0.4787 | 0.2980 | 0.3491 | 0.2215 | 0.2414 | 0.8339 | 1126.0000 |
| tag_cosine | 0.1465 | 0.1194 | 0.2647 | 0.3887 | 0.2440 | 0.2872 | 0.1932 | 0.2101 | 0.6317 | 1126.0000 |
| popularity_degree | 0.1432 | 0.1218 | 0.2029 | 0.3290 | 0.2028 | 0.2448 | 0.1416 | 0.1573 | 0.7850 | 1126.0000 |
| random | 0.0286 | 0.0298 | 0.0453 | 0.0863 | 0.0389 | 0.0543 | 0.0226 | 0.0260 | 0.4925 | 1126.0000 |

Discuss NDCG@10, Recall@10, MAP@10, and qualitative matches.

## Discussion
Compare interpretability, complementary-matching structure, and predictive flexibility. Explain why friend links are useful but noisy.

## Limitations and Ethics
Mention old dataset, small sample, social links not pure compatibility, and privacy/cultural-signal concerns.

## Contribution Statement
Owen implemented and evaluated the proposal hybrid compatibility model, including artist similarity, tag similarity, niche overlap, discovery potential, validation tuning, and presentation-ready analysis artifacts.
