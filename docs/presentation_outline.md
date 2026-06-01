# Six-Minute Presentation Outline

1. Problem and motivation
   - Most music recommenders recommend songs to users.
   - Our project recommends users to users based on music taste.

2. Data and task
   - HetRec Last.fm 2K: listening counts, tags, and friend links.
   - Cleaned-tag variant: tag reliability thresholds plus noisy-tag removal.
   - Friends are musically closer than random non-friends: cosine 0.1999 vs 0.0469.
   - Train/validation/test split on friend links.
   - Evaluation: held-out friends ranked against 100 sampled non-friends per target.
   - Rank candidate users for each target user.

3. Approach 1: proposal hybrid model
   - Artist cosine, tag cosine, niche overlap, discovery potential.
   - Validation-tuned weighted sum: {"artist": 0.55, "tag": 0.15, "niche": 0.2, "discovery": 0.1}.

4. Approach 2: learned ranker
   - Reliability-filtered users.
   - Positive friend pairs plus sampled non-friend pairs, excluding all known friends from negatives.
   - Gradient boosting over the same compatibility features plus activity/popularity features.

5. Results
   - Best model: teammate_learned_ranker, NDCG@10=0.4452.
   - Proposal hybrid: NDCG@10=0.3656, Recall@10=0.4776.
   - Learned ranker: NDCG@10=0.4452, Recall@10=0.5701.
   - Cleaned-tag run: learned ranker NDCG@10=0.4467, proposal hybrid NDCG@10=0.3662, tag cosine NDCG@10=0.3090.
   - Main takeaway: learned weighting performs best; the proposal hybrid is simpler and more interpretable.
   - Cleaning takeaway: useful mostly for the tag-only model; little change for the stronger overall models.
   - Caveat: artist cosine slightly beats the proposal hybrid on top-K metrics, so the hybrid's value is interpretability and AUC rather than raw top-K lift.
   - Show `artifacts/ndcg10_comparison.png` and `artifacts/top10_metrics.png`.

6. Qualitative example
   - Use one row from `artifacts/qualitative_matches.csv`.
   - Explain which shared artists/tags and which score component drove the match.

7. Limitations and future work
   - Friend links are a noisy proxy for taste compatibility.
   - We created a filtered taste-aligned ground-truth file for future experiments.
   - Dataset is old and small.
   - Future work: richer sequential listening data, stronger neural two-tower retrieval, user-facing explanations.
