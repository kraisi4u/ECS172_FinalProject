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

## Novel approach: dual-space complementary matching

The added model represents each user in two learned spaces:

- Taste/comfort space: who has similar listening and tag context.
- Seeker-curator space: which candidate can introduce music that fits the target's latent interests, with rarer artists emphasized during initialization.

It initializes these spaces from artist, tag, rarity-weighted artist, and train-friendship features, then trains with a pairwise ranking objective over held-out-safe friend versus non-friend comparisons. Its final score is a validation-tuned blend of the learned dual-space score and the interpretable compatibility signals. The selected model config was `{"dim": 24, "epochs": 12, "lr": 0.035, "reg": 0.0015, "comfort": 0.5, "complement": 0.4, "reciprocal": 0.1}` with blend weights `{"dual": 0.5, "artist": 0.3, "tag": 0.05, "niche": 0.1, "discovery": 0.05}`.

## Teammate/message approach: reliability-filtered learned ranker

The second approach follows the message-thread plan: clean and split the Last.fm data, filter unreliable users, construct pairwise features, and train a learned model on friend versus sampled non-friend pairs. The implemented ranker uses a reliability filter of at least 10 artists and 3 tags per user, then trains a histogram gradient boosting classifier on artist similarity, tag similarity, niche overlap, discovery score, activity differences, candidate popularity degree, and candidate listening mass. Training negatives exclude all known friend pairs, including validation and test friends.

The Gemini thread also proposed a two-tower neural retrieval model. We did not train that model here because this dataset is small, the presentation window is short, and the learned ranker gives a stronger low-risk comparison. The two-tower setup is included as a future-work path using `filtered_ground_truth_pairs.csv` or friend pairs with in-batch negatives.

## Test results

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

The strongest model by NDCG@10 was `dual_space_complementary` with NDCG@10=0.4823. The proposal hybrid produced NDCG@10=0.3656; the dual-space complementary model produced NDCG@10=0.4823; the learned ranker produced NDCG@10=0.4452. The dual-space model performs best because it models both shared taste comfort and complementary discovery value while still using validation-tuned interpretable signals. The proposal hybrid remains the most interpretable version of the idea, and the learned ranker remains a strong flexible supervised comparison. All results should still be interpreted with the sampled-negative setup and noisy friend-link proxy in mind.

## Interpretation

The comparison supports the final presentation goal: the hybrid model is interpretable and directly tied to the original motivation, the dual-space model adds the novel comfort-plus-discovery structure, and the learned ranker tests whether a feature-based supervised model can use the same signals flexibly. Friend links are still an imperfect ground truth, so the qualitative examples and music-specific components should be shown alongside the ranking metrics.
