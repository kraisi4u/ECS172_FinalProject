# Music Taste Matchmaking Experiment Summary

## Project framing

The project treats music taste matchmaking as a user-user recommendation problem. For each target Last.fm user, the system ranks candidate users by likely compatibility, using held-out Last.fm friend links as the main evaluation signal.

## Dataset

- Source: HetRec 2011 Last.fm 2K.
- Users: 1892
- Artists: 17632
- Listening records: 92834
- Undirected friend links: 12717
- Split: 8901 train, 1908 validation, 1908 test edges.
- Evaluation ranks each held-out friend against 100 sampled non-friends for the same target user.
- Niche artists: 10679 artists at or below the 30th listener-count percentile.

## Cleaned-tag variant

We also evaluated a cleaned-tag variant based on the original proposal. The cleaning keeps tags with at least 20 assignments, 5 users, and 5 artists, removes obvious noise such as years/decades, "seen live", favorites, personal-list tags, and pure opinion tags, preserves ambiguous but useful tags like `acoustic`, `soundtrack`, and `idm`, and counts unique tagged artists per user-tag pair to reduce repeated-tagging skew.

## Gemini/context evidence and ground truth choice

The Gemini thread correctly raised the key question: friends are not perfect music-compatibility labels. We therefore keep the held-out friend-link task for measurable evaluation, but we also quantify the signal and construct a filtered compatibility view.

- Average artist-cosine similarity among friend pairs: 0.1999.
- Average artist-cosine similarity among sampled non-friend pairs: 0.0469.
- Friend/non-friend similarity ratio: 4.26x.
- Filtered taste-aligned ground truth: 8652 friend pairs with artist cosine >= 0.10, balanced with 8652 non-friend pairs with artist cosine <= 0.02.

This means friendship is treated as a noisy proxy rather than absolute truth. The filtered file is saved as `artifacts/filtered_ground_truth_pairs.csv` for future experiments or a two-tower model.

## Owen/proposal approach: hybrid compatibility scoring

The proposal model scores each target-candidate pair as a weighted sum of four components:

- Artist similarity: cosine similarity over log-normalized listening vectors.
- Tag similarity: cosine similarity over user tag-frequency vectors.
- Niche overlap: shared niche listening mass normalized by the target user's niche listening mass.
- Discovery potential: candidate artists absent from the target history, weighted by latent artist relevance to the target.

The validation-selected weights were `{"artist": 0.55, "tag": 0.15, "niche": 0.2, "discovery": 0.1}`.

## Teammate/message approach: reliability-filtered learned ranker

The second approach follows the message-thread plan: clean and split the Last.fm data, filter unreliable users, construct pairwise features, and train a learned model on friend versus sampled non-friend pairs. The implemented ranker uses a reliability filter of at least 10 artists and 3 tags per user, then trains a histogram gradient boosting classifier on artist similarity, tag similarity, niche overlap, discovery score, activity differences, candidate popularity degree, and candidate listening mass. Training negatives exclude all known friend pairs, including validation and test friends.

The Gemini thread also proposed a two-tower neural retrieval model. We did not train that model here because this dataset is small, the presentation window is short, and the learned ranker gives a stronger low-risk comparison. The two-tower setup is included as a future-work path using `filtered_ground_truth_pairs.csv` or friend pairs with in-batch negatives.

## Test results

| model | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 | map@5 | map@10 | auc | evaluated_targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teammate_learned_ranker | 0.2361 | 0.1795 | 0.4050 | 0.5701 | 0.3923 | 0.4452 | 0.3071 | 0.3289 | 0.8807 | 1126.0000 |
| artist_cosine | 0.1961 | 0.1473 | 0.3510 | 0.4824 | 0.3317 | 0.3736 | 0.2557 | 0.2701 | 0.8209 | 1126.0000 |
| proposal_hybrid | 0.1933 | 0.1467 | 0.3383 | 0.4776 | 0.3213 | 0.3656 | 0.2445 | 0.2604 | 0.8320 | 1126.0000 |
| svd_embedding | 0.1858 | 0.1463 | 0.3238 | 0.4787 | 0.2980 | 0.3491 | 0.2215 | 0.2414 | 0.8339 | 1126.0000 |
| tag_cosine | 0.1465 | 0.1194 | 0.2647 | 0.3887 | 0.2440 | 0.2872 | 0.1932 | 0.2101 | 0.6317 | 1126.0000 |
| popularity_degree | 0.1432 | 0.1218 | 0.2029 | 0.3290 | 0.2028 | 0.2448 | 0.1416 | 0.1573 | 0.7850 | 1126.0000 |
| random | 0.0286 | 0.0298 | 0.0453 | 0.0863 | 0.0389 | 0.0543 | 0.0226 | 0.0260 | 0.4925 | 1126.0000 |

The strongest model by NDCG@10 was `teammate_learned_ranker` with NDCG@10=0.4452. The proposal hybrid produced NDCG@10=0.3656; the learned ranker produced NDCG@10=0.4452. The learned ranker performs best because it can combine the compatibility signals nonlinearly. The proposal hybrid remains the more interpretable version of the idea, but it does not beat plain artist cosine on top-K ranking; its advantage over artist cosine is in AUC. All results should still be interpreted with the sampled-negative setup and noisy friend-link proxy in mind.

## Cleaned-tag comparison

Relative to the baseline above, the cleaned-tag run changed the headline metrics as follows:

- Learned ranker: NDCG@10 0.4467 vs 0.4452, Recall@10 0.5716 vs 0.5701, AUC 0.8785 vs 0.8807.
- Proposal hybrid: NDCG@10 0.3662 vs 0.3656, Recall@10 0.4781 vs 0.4776, AUC 0.8325 vs 0.8320.
- Tag cosine: NDCG@10 0.3090 vs 0.2872, Recall@10 0.4083 vs 0.3887, AUC 0.6291 vs 0.6317.

Conclusion: cleaning was modestly useful overall. It clearly improved the pure tag baseline, while leaving the artist-driven, hybrid, and learned models almost unchanged.

## Interpretation

The comparison supports the final presentation goal: the hybrid model is interpretable and directly tied to the original motivation, while the learned model tests whether a feature-based supervised ranker can use the same signals more flexibly. Friend links are still an imperfect ground truth, so the qualitative examples and music-specific components should be shown alongside the ranking metrics.
