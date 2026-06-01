from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_hetrec_proposal_cleaned
from src.music_match import (
    Dataset,
    dense_similarity,
    discovery_matrix,
    filtered_ground_truth_pairs,
    friend_random_similarity_summary,
    load_hetrec,
    make_eval_cases,
    minmax_rows,
    niche_overlap_matrix,
    rank_metrics,
    save_json,
    split_friend_edges_by_user,
    svd_user_similarity,
    top_match_examples,
    train_learned_ranker,
    tune_hybrid_weights,
)


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "dataset"
ARTIFACT_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"


def output_path(directory: Path, filename: str, suffix: str = "") -> Path:
    if not suffix:
        return directory / filename
    stem, ext = filename.rsplit(".", 1)
    return directory / f"{stem}-{suffix}.{ext}"


def evaluate_models(dataset: Dataset, output_suffix: str = "") -> dict:
    train_edges, validation_edges, test_edges = split_friend_edges_by_user(dataset.edges, seed=172)
    validation_cases = make_eval_cases(validation_edges, dataset.edges, dataset.user_to_idx, negatives_per_user=100, seed=172)
    test_cases = make_eval_cases(test_edges, dataset.edges, dataset.user_to_idx, negatives_per_user=100, seed=173)

    artist_cosine_raw = dense_similarity(dataset.user_artist_norm)
    friend_random_summary = friend_random_similarity_summary(artist_cosine_raw, dataset.edges, dataset.user_to_idx)
    filtered_gt = filtered_ground_truth_pairs(artist_cosine_raw, dataset.edges, dataset.user_to_idx)
    filtered_gt.to_csv(output_path(ARTIFACT_DIR, "filtered_ground_truth_pairs.csv", output_suffix), index=False)

    artist_sim = minmax_rows(artist_cosine_raw)
    tag_sim = minmax_rows(dense_similarity(dataset.user_tag_norm))
    svd_sim, user_emb, item_emb = svd_user_similarity(dataset.user_artist_norm)
    svd_sim = minmax_rows(svd_sim)
    niche_sim, niche_mask = niche_overlap_matrix(dataset.user_artist_raw)
    niche_sim = minmax_rows(niche_sim)
    discovery_sim = discovery_matrix(dataset.user_artist_raw, user_emb, item_emb)

    components = {
        "artist": artist_sim,
        "tag": tag_sim,
        "niche": niche_sim,
        "discovery": discovery_sim,
    }
    hybrid_weights, hybrid_score, validation_metrics = tune_hybrid_weights(components, validation_cases)
    learned_score, learned_training = train_learned_ranker(train_edges, dataset.edges, dataset, components)

    degree_scores = np.zeros_like(artist_sim)
    train_degree = np.zeros(len(dataset.users), dtype=np.float32)
    for u, v in train_edges:
        train_degree[dataset.user_to_idx[u]] += 1.0
        train_degree[dataset.user_to_idx[v]] += 1.0
    degree_scores[:] = train_degree[None, :]
    np.fill_diagonal(degree_scores, -np.inf)
    degree_scores = minmax_rows(degree_scores)

    rng = np.random.default_rng(172)
    random_scores = rng.random(artist_sim.shape, dtype=np.float32)
    np.fill_diagonal(random_scores, -np.inf)

    models = {
        "random": random_scores,
        "popularity_degree": degree_scores,
        "artist_cosine": artist_sim,
        "tag_cosine": tag_sim,
        "svd_embedding": svd_sim,
        "proposal_hybrid": hybrid_score,
        "teammate_learned_ranker": learned_score,
    }
    metrics = {name: rank_metrics(score, test_cases) for name, score in models.items()}
    leaderboard = pd.DataFrame(metrics).T.sort_values("ndcg@10", ascending=False)
    leaderboard.to_csv(output_path(ARTIFACT_DIR, "leaderboard.csv", output_suffix), float_format="%.6f")

    examples = top_match_examples(hybrid_score, dataset, components)
    examples.to_csv(output_path(ARTIFACT_DIR, "qualitative_matches.csv", output_suffix), index=False)

    payload = {
        "dataset": {
            "users": int(len(dataset.users)),
            "artists": int(len(dataset.artists)),
            "listening_records": int(dataset.user_artist_raw.nnz),
            "friend_edges_undirected": int(len(dataset.edges)),
            "train_edges": int(len(train_edges)),
            "validation_edges": int(len(validation_edges)),
            "test_edges": int(len(test_edges)),
            "niche_artist_count": int(niche_mask.sum()),
        },
        "friend_random_similarity": friend_random_summary,
        "filtered_ground_truth": {
            "positive_threshold": 0.10,
            "negative_threshold": 0.02,
            "rows": int(len(filtered_gt)),
            "positives": int((filtered_gt["label"] == 1).sum()) if len(filtered_gt) else 0,
            "negatives": int((filtered_gt["label"] == 0).sum()) if len(filtered_gt) else 0,
        },
        "hybrid_weights": hybrid_weights,
        "hybrid_validation_metrics": validation_metrics,
        "learned_ranker_training": learned_training,
        "test_metrics": metrics,
    }
    save_json(output_path(ARTIFACT_DIR, "results.json", output_suffix), payload)
    save_figures(leaderboard, output_suffix)
    if not output_suffix:
        save_docs(payload, leaderboard, examples)
    return payload


def save_figures(leaderboard: pd.DataFrame, output_suffix: str = "") -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    order = leaderboard.sort_values("ndcg@10")
    ax.barh(order.index, order["ndcg@10"], color="#3B82F6")
    ax.set_title("Friend-link ranking performance")
    ax.set_xlabel("NDCG@10")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path(ARTIFACT_DIR, "ndcg10_comparison.png", output_suffix), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    columns = ["precision@10", "recall@10", "map@10"]
    leaderboard[columns].plot(kind="bar", ax=ax)
    ax.set_title("Top-10 ranking metrics")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_path(ARTIFACT_DIR, "top10_metrics.png", output_suffix), dpi=180)
    plt.close(fig)


def save_docs(payload: dict, leaderboard: pd.DataFrame, examples: pd.DataFrame) -> None:
    best_name = leaderboard.index[0]
    best = leaderboard.iloc[0]
    hybrid = leaderboard.loc["proposal_hybrid"]
    learned = leaderboard.loc["teammate_learned_ranker"]

    leaderboard_table = markdown_table(leaderboard)
    method_summary = f"""# Music Taste Matchmaking Experiment Summary

## Project framing

The project treats music taste matchmaking as a user-user recommendation problem. For each target Last.fm user, the system ranks candidate users by likely compatibility, using held-out Last.fm friend links as the main evaluation signal.

## Dataset

- Source: HetRec 2011 Last.fm 2K.
- Users: {payload["dataset"]["users"]}
- Artists: {payload["dataset"]["artists"]}
- Listening records: {payload["dataset"]["listening_records"]}
- Undirected friend links: {payload["dataset"]["friend_edges_undirected"]}
- Split: {payload["dataset"]["train_edges"]} train, {payload["dataset"]["validation_edges"]} validation, {payload["dataset"]["test_edges"]} test edges.
- Evaluation ranks each held-out friend against 100 sampled non-friends for the same target user.
- Niche artists: {payload["dataset"]["niche_artist_count"]} artists at or below the 30th listener-count percentile.

## Gemini/context evidence and ground truth choice

The Gemini thread correctly raised the key question: friends are not perfect music-compatibility labels. We therefore keep the held-out friend-link task for measurable evaluation, but we also quantify the signal and construct a filtered compatibility view.

- Average artist-cosine similarity among friend pairs: {payload["friend_random_similarity"]["friend_mean_similarity"]:.4f}.
- Average artist-cosine similarity among sampled non-friend pairs: {payload["friend_random_similarity"]["random_nonfriend_mean_similarity"]:.4f}.
- Friend/non-friend similarity ratio: {payload["friend_random_similarity"]["friend_to_random_similarity_ratio"]:.2f}x.
- Filtered taste-aligned ground truth: {payload["filtered_ground_truth"]["positives"]} friend pairs with artist cosine >= {payload["filtered_ground_truth"]["positive_threshold"]:.2f}, balanced with {payload["filtered_ground_truth"]["negatives"]} non-friend pairs with artist cosine <= {payload["filtered_ground_truth"]["negative_threshold"]:.2f}.

This means friendship is treated as a noisy proxy rather than absolute truth. The filtered file is saved as `artifacts/filtered_ground_truth_pairs.csv` for future experiments or a two-tower model.

## Owen/proposal approach: hybrid compatibility scoring

The proposal model scores each target-candidate pair as a weighted sum of four components:

- Artist similarity: cosine similarity over log-normalized listening vectors.
- Tag similarity: cosine similarity over user tag-frequency vectors.
- Niche overlap: shared niche listening mass normalized by the target user's niche listening mass.
- Discovery potential: candidate artists absent from the target history, weighted by latent artist relevance to the target.

The validation-selected weights were `{json.dumps(payload["hybrid_weights"])}`.

## Teammate/message approach: reliability-filtered learned ranker

The second approach follows the message-thread plan: clean and split the Last.fm data, filter unreliable users, construct pairwise features, and train a learned model on friend versus sampled non-friend pairs. The implemented ranker uses a reliability filter of at least {int(payload["learned_ranker_training"]["min_artists"])} artists and {int(payload["learned_ranker_training"]["min_tags"])} tags per user, then trains a histogram gradient boosting classifier on artist similarity, tag similarity, niche overlap, discovery score, activity differences, candidate popularity degree, and candidate listening mass. Training negatives exclude all known friend pairs, including validation and test friends.

The Gemini thread also proposed a two-tower neural retrieval model. We did not train that model here because this dataset is small, the presentation window is short, and the learned ranker gives a stronger low-risk comparison. The two-tower setup is included as a future-work path using `filtered_ground_truth_pairs.csv` or friend pairs with in-batch negatives.

## Test results

{leaderboard_table}

The strongest model by NDCG@10 was `{best_name}` with NDCG@10={best["ndcg@10"]:.4f}. The proposal hybrid produced NDCG@10={hybrid["ndcg@10"]:.4f}; the learned ranker produced NDCG@10={learned["ndcg@10"]:.4f}. The learned ranker performs best because it can combine the compatibility signals nonlinearly. The proposal hybrid remains the more interpretable version of the idea, but it does not beat plain artist cosine on top-K ranking; its advantage over artist cosine is in AUC. All results should still be interpreted with the sampled-negative setup and noisy friend-link proxy in mind.

## Interpretation

The comparison supports the final presentation goal: the hybrid model is interpretable and directly tied to the original motivation, while the learned model tests whether a feature-based supervised ranker can use the same signals more flexibly. Friend links are still an imperfect ground truth, so the qualitative examples and music-specific components should be shown alongside the ranking metrics.
"""
    (DOCS_DIR / "method_summary.md").write_text(method_summary)

    slide_outline = f"""# Six-Minute Presentation Outline

1. Problem and motivation
   - Most music recommenders recommend songs to users.
   - Our project recommends users to users based on music taste.

2. Data and task
   - HetRec Last.fm 2K: listening counts, tags, and friend links.
   - Friends are musically closer than random non-friends: cosine {payload["friend_random_similarity"]["friend_mean_similarity"]:.4f} vs {payload["friend_random_similarity"]["random_nonfriend_mean_similarity"]:.4f}.
   - Train/validation/test split on friend links.
   - Evaluation: held-out friends ranked against 100 sampled non-friends per target.
   - Rank candidate users for each target user.

3. Approach 1: proposal hybrid model
   - Artist cosine, tag cosine, niche overlap, discovery potential.
   - Validation-tuned weighted sum: {json.dumps(payload["hybrid_weights"])}.

4. Approach 2: learned ranker
   - Reliability-filtered users.
   - Positive friend pairs plus sampled non-friend pairs, excluding all known friends from negatives.
   - Gradient boosting over the same compatibility features plus activity/popularity features.

5. Results
   - Best model: {best_name}, NDCG@10={best["ndcg@10"]:.4f}.
   - Proposal hybrid: NDCG@10={hybrid["ndcg@10"]:.4f}, Recall@10={hybrid["recall@10"]:.4f}.
   - Learned ranker: NDCG@10={learned["ndcg@10"]:.4f}, Recall@10={learned["recall@10"]:.4f}.
   - Main takeaway: learned weighting performs best; the proposal hybrid is simpler and more interpretable.
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
"""
    (DOCS_DIR / "presentation_outline.md").write_text(slide_outline)

    paper_scaffold = f"""# Final Paper Scaffold

## Abstract
We study music taste matchmaking as a user-user recommendation problem using the HetRec 2011 Last.fm dataset. The system ranks candidate users for a target user using listening histories, tags, niche artist overlap, and discovery potential. We compare simple baselines, an interpretable hybrid compatibility score, and a reliability-filtered learned ranker trained on held-out friend links.

## Introduction
Explain why music taste can support social recommendation, not just song recommendation. State the retrieval task and the novelty axis.

## Related Work
Use the three proposal papers: graph bottlenecked social recommendation, attribute-aware music personalization, and popularity-bias mitigation in music recommenders.

## Methodology
Describe preprocessing, friend-link splitting, sampled-negative evaluation, the filtered taste-aligned ground-truth analysis, baselines, proposal hybrid scoring, and learned ranker.

## Experiments and Results
Include this table:

{leaderboard_table}

Discuss NDCG@10, Recall@10, MAP@10, and qualitative matches.

## Discussion
Compare interpretability versus predictive flexibility. Explain why friend links are useful but noisy.

## Limitations and Ethics
Mention old dataset, small sample, social links not pure compatibility, and privacy/cultural-signal concerns.

## Contribution Statement
Owen implemented and evaluated the proposal hybrid compatibility model, including artist similarity, tag similarity, niche overlap, discovery potential, validation tuning, and presentation-ready analysis artifacts.
"""
    (DOCS_DIR / "final_paper_scaffold.md").write_text(paper_scaffold)

    (DOCS_DIR / "example_matches.md").write_text(markdown_table(examples.head(3)) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    table = frame.copy()
    table.insert(0, "model", table.index.astype(str))
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the music taste matching experiment.")
    parser.add_argument(
        "--loader",
        choices=("raw", "cleaned"),
        default="raw",
        help="Choose whether to load the raw dataset or the cleaned-tag dataset.",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    if args.loader == "cleaned":
        dataset = load_hetrec_proposal_cleaned(RAW_DIR)
        output_suffix = "cleaned"
    else:
        dataset = load_hetrec(RAW_DIR)
        output_suffix = ""
    payload = evaluate_models(dataset, output_suffix=output_suffix)
    print(json.dumps(payload["dataset"], indent=2))
    print(pd.DataFrame(payload["test_metrics"]).T.sort_values("ndcg@10", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
