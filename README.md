# ECS 172 Final Project

User-User Recommendation via Music Taste

Collaborators: Owen Holt, Kian Raisi, Ilan Eliashberg, Ayush Tripathi

## Project Summary

This project treats music taste matchmaking as a user-user recommendation problem. Given a target Last.fm user, the system ranks candidate users by likely musical compatibility using listening histories, tags, niche artist overlap, discovery potential, and the Last.fm friendship graph as a noisy evaluation signal.

## Implemented Approaches

- Proposal/hybrid method: artist cosine similarity, tag cosine similarity, niche artist overlap, and discovery potential combined with validation-tuned weights.
- Teammate/Gemini method: reliability-filtered learned ranker trained on friend pairs versus sampled non-friend pairs using the compatibility signals plus activity and popularity features.
- Gemini/context analysis: friend-vs-random music similarity evidence and a filtered taste-aligned ground-truth file for future model training.
- Baselines: random ranking, popularity degree, artist cosine, tag cosine, and SVD embedding similarity.

## Current Results

- Learned ranker: NDCG@10 = 0.4452, Recall@10 = 0.5701, AUC = 0.8807.
- Proposal hybrid: NDCG@10 = 0.3656, Recall@10 = 0.4776, AUC = 0.8320.
- Artist cosine baseline: NDCG@10 = 0.3736, Recall@10 = 0.4824, AUC = 0.8209.

Interpretation for the presentation: the learned ranker performs best because it learns how to combine compatibility signals. The proposal hybrid remains useful as the interpretable version of the idea, but artist cosine slightly beats it on top-k ranking metrics; the hybrid's advantage over artist cosine is AUC and explainability.

## Evaluation Setup

- Dataset: HetRec 2011 Last.fm 2K.
- Split: 8,901 train friend edges, 1,908 validation friend edges, 1,908 test friend edges.
- Evaluation: each held-out friend is ranked against 100 sampled non-friends for the same target user.
- Negatives exclude all known friend links, including validation and test links.
- Friend links are treated as a noisy proxy, not perfect truth. The experiment also saves `artifacts/filtered_ground_truth_pairs.csv`, where positives are friends with artist cosine >= 0.10 and negatives are non-friends with artist cosine <= 0.02.

## Important Files

- `run_experiment.py`: full experiment runner.
- `src/music_match.py`: data loading, splitting, scoring, modeling, and metrics.
- `tests/test_music_match.py`: focused tests for core split/scoring behavior.
- `src/data_cleaning_decision_making.md`: original data-cleaning and loader plan.
- `artifacts/results.json`: full metrics and experiment metadata.
- `artifacts/leaderboard.csv`: model comparison table.
- `artifacts/ndcg10_comparison.png`: slide-ready NDCG@10 chart.
- `artifacts/top10_metrics.png`: slide-ready top-10 metrics chart.
- `artifacts/qualitative_matches.csv`: example matches for discussion.
- `artifacts/filtered_ground_truth_pairs.csv`: taste-aligned compatibility labels based on the Gemini discussion.
- `docs/method_summary.md`: concise method/results writeup.
- `docs/presentation_outline.md`: six-minute presentation outline.
- `docs/final_paper_scaffold.md`: starter structure for the final paper.

## Reproduce

From the repository root:

```bash
python3 -m unittest discover -s tests
python3 run_experiment.py
```

The runner expects the HetRec files under `dataset/`, which are already included in this repository.
