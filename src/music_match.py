from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from src.data_loader import Dataset, build_undirected_edges, load_hetrec


RNG_SEED = 172


def split_friend_edges_by_user(
    edges: list[tuple[int, int]],
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RNG_SEED,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    rng = random.Random(seed)
    degree: dict[int, int] = {}
    for edge in edges:
        u, v = edge
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    validation_target = int(round(len(edges) * validation_ratio))
    test_target = int(round(len(edges) * test_ratio))
    validation: list[tuple[int, int]] = []
    test: list[tuple[int, int]] = []
    train: list[tuple[int, int]] = []
    remaining_degree = dict(degree)
    shuffled = list(edges)
    rng.shuffle(shuffled)

    for edge in shuffled:
        u, v = edge
        can_hold_out = remaining_degree[u] > 1 and remaining_degree[v] > 1
        if can_hold_out and len(validation) < validation_target:
            validation.append(edge)
            remaining_degree[u] -= 1
            remaining_degree[v] -= 1
        elif can_hold_out and len(test) < test_target:
            test.append(edge)
            remaining_degree[u] -= 1
            remaining_degree[v] -= 1
        else:
            train.append(edge)

    if not validation and edges:
        validation.append(edges[0])
        train = [edge for edge in train if edge != edges[0]]
    if not test and len(edges) > 1:
        fallback = next(edge for edge in edges if edge not in validation)
        test.append(fallback)
        train = [edge for edge in train if edge != fallback]
    return train, sorted(validation), sorted(test)


def compute_niche_overlap(matrix: np.ndarray, target_idx: int, candidate_idx: int, niche_mask: np.ndarray) -> float:
    target = matrix[target_idx, niche_mask]
    candidate = matrix[candidate_idx, niche_mask]
    target_mass = float(target.sum())
    if target_mass <= 0:
        return 0.0
    shared_mass = float(np.minimum(target, candidate).sum())
    return shared_mass / target_mass


def dense_similarity(matrix: sparse.csr_matrix) -> np.ndarray:
    sim = (matrix @ matrix.T).toarray().astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    return sim


def minmax_rows(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    out[~finite] = 0.0
    row_min = out.min(axis=1, keepdims=True)
    row_max = out.max(axis=1, keepdims=True)
    denom = np.where(row_max > row_min, row_max - row_min, 1.0)
    return (out - row_min) / denom


def niche_overlap_matrix(user_artist_raw: sparse.csr_matrix, percentile: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    artist_listener_counts = np.asarray((user_artist_raw > 0).sum(axis=0)).ravel()
    threshold = np.percentile(artist_listener_counts[artist_listener_counts > 0], percentile)
    niche_mask = artist_listener_counts <= threshold
    niche = user_artist_raw[:, niche_mask].astype(bool).astype(np.float32)
    shared = (niche @ niche.T).toarray().astype(np.float32)
    target_mass = np.asarray(niche.sum(axis=1)).ravel().astype(np.float32)
    overlap = shared / np.maximum(target_mass[:, None], 1.0)
    overlap[target_mass == 0, :] = 0.0
    np.fill_diagonal(overlap, -np.inf)
    return overlap, niche_mask


def svd_user_similarity(user_artist_norm: sparse.csr_matrix, n_components: int = 32, seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_components = min(n_components, min(user_artist_norm.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    user_emb = svd.fit_transform(user_artist_norm).astype(np.float32)
    user_emb_norm = normalize(user_emb, norm="l2", axis=1)
    sim = (user_emb_norm @ user_emb_norm.T).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    item_emb = svd.components_.T.astype(np.float32)
    return sim, user_emb_norm.astype(np.float32), item_emb


def discovery_matrix(user_artist_raw: sparse.csr_matrix, user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    binary = (user_artist_raw > 0).astype(np.float32).tocsr()
    discovery = np.zeros((user_artist_raw.shape[0], user_artist_raw.shape[0]), dtype=np.float32)
    for u in range(user_artist_raw.shape[0]):
        rel = item_emb @ user_emb[u]
        rel = rel - rel.min()
        if rel.max() > 0:
            rel = rel / rel.max()
        total = binary @ rel
        target_artists = binary[u].indices
        if len(target_artists):
            shared = binary[:, target_artists] @ rel[target_artists]
            unseen_count = np.asarray(binary.sum(axis=1)).ravel() - np.asarray(binary[:, target_artists].sum(axis=1)).ravel()
        else:
            shared = 0.0
            unseen_count = np.asarray(binary.sum(axis=1)).ravel()
        discovery[u] = (np.asarray(total).ravel() - np.asarray(shared).ravel()) / np.maximum(unseen_count, 1.0)
    np.fill_diagonal(discovery, -np.inf)
    return minmax_rows(discovery)


def adjacency_from_edges(edges: list[tuple[int, int]], user_to_idx: dict[int, int]) -> dict[int, set[int]]:
    friends: dict[int, set[int]] = {idx: set() for idx in range(len(user_to_idx))}
    for u, v in edges:
        if u in user_to_idx and v in user_to_idx:
            ui = user_to_idx[u]
            vi = user_to_idx[v]
            friends[ui].add(vi)
            friends[vi].add(ui)
    return friends


def make_eval_cases(
    positive_edges: list[tuple[int, int]],
    all_edges: list[tuple[int, int]],
    user_to_idx: dict[int, int],
    negatives_per_user: int = 100,
    seed: int = RNG_SEED,
) -> dict[int, list[tuple[int, int]]]:
    rng = random.Random(seed)
    all_friend_adj = adjacency_from_edges(all_edges, user_to_idx)
    positives: dict[int, set[int]] = {}
    for u, v in positive_edges:
        if u in user_to_idx and v in user_to_idx:
            ui = user_to_idx[u]
            vi = user_to_idx[v]
            positives.setdefault(ui, set()).add(vi)
            positives.setdefault(vi, set()).add(ui)

    all_user_idxs = set(range(len(user_to_idx)))
    cases: dict[int, list[tuple[int, int]]] = {}
    for target, pos_set in positives.items():
        unavailable = all_friend_adj[target] | {target}
        candidates = sorted(all_user_idxs - unavailable)
        sample_size = min(negatives_per_user, len(candidates))
        negatives = rng.sample(candidates, sample_size)
        cases[target] = [(candidate, int(candidate in pos_set)) for candidate in sorted(pos_set) + negatives]
    return cases


def friend_random_similarity_summary(
    similarity_matrix: np.ndarray,
    edges: list[tuple[int, int]],
    user_to_idx: dict[int, int],
    sample_size: int = 50000,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    friend_pairs = {(user_to_idx[u], user_to_idx[v]) for u, v in edges if u in user_to_idx and v in user_to_idx}
    friend_pairs |= {(v, u) for u, v in friend_pairs}
    undirected_friend_values = [
        float(similarity_matrix[user_to_idx[u], user_to_idx[v]])
        for u, v in edges
        if u in user_to_idx and v in user_to_idx and np.isfinite(similarity_matrix[user_to_idx[u], user_to_idx[v]])
    ]

    rng = random.Random(seed)
    n_users = len(user_to_idx)
    random_values: list[float] = []
    attempts = 0
    max_attempts = sample_size * 100
    while len(random_values) < sample_size and attempts < max_attempts:
        attempts += 1
        u = rng.randrange(n_users)
        v = rng.randrange(n_users)
        if u == v or (u, v) in friend_pairs:
            continue
        value = float(similarity_matrix[u, v])
        if np.isfinite(value):
            random_values.append(value)

    friend_mean = float(np.mean(undirected_friend_values)) if undirected_friend_values else 0.0
    random_mean = float(np.mean(random_values)) if random_values else 0.0
    return {
        "friend_pair_count": float(len(undirected_friend_values)),
        "random_nonfriend_pair_count": float(len(random_values)),
        "friend_mean_similarity": friend_mean,
        "random_nonfriend_mean_similarity": random_mean,
        "friend_to_random_similarity_ratio": friend_mean / random_mean if random_mean > 0 else 0.0,
    }


def filtered_ground_truth_pairs(
    similarity_matrix: np.ndarray,
    edges: list[tuple[int, int]],
    user_to_idx: dict[int, int],
    positive_threshold: float = 0.10,
    negative_threshold: float = 0.02,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    friend_pairs = {(user_to_idx[u], user_to_idx[v]) for u, v in edges if u in user_to_idx and v in user_to_idx}
    friend_pairs |= {(v, u) for u, v in friend_pairs}

    positives: list[dict[str, float | int]] = []
    for u, v in edges:
        if u not in user_to_idx or v not in user_to_idx:
            continue
        ui = user_to_idx[u]
        vi = user_to_idx[v]
        score = float(similarity_matrix[ui, vi])
        if np.isfinite(score) and score >= positive_threshold:
            positives.append({"user_A": int(u), "user_B": int(v), "taste_similarity": score, "label": 1})

    negatives: list[dict[str, float | int]] = []
    rng = random.Random(seed)
    n_users = len(user_to_idx)
    attempts = 0
    max_attempts = max(len(positives), 1) * 500
    seen_negatives: set[tuple[int, int]] = set()
    while len(negatives) < len(positives) and attempts < max_attempts:
        attempts += 1
        ui = rng.randrange(n_users)
        vi = rng.randrange(n_users)
        if ui == vi or (ui, vi) in friend_pairs:
            continue
        edge = (min(ui, vi), max(ui, vi))
        if edge in seen_negatives:
            continue
        score = float(similarity_matrix[ui, vi])
        if np.isfinite(score) and score <= negative_threshold:
            seen_negatives.add(edge)
            negatives.append(
                {
                    "user_A": int(idx_to_user[ui]),
                    "user_B": int(idx_to_user[vi]),
                    "taste_similarity": score,
                    "label": 0,
                }
            )

    return pd.DataFrame(positives + negatives).sample(frac=1, random_state=seed).reset_index(drop=True)


def rank_metrics(score_matrix: np.ndarray, cases: dict[int, list[tuple[int, int]]], ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    totals = {f"precision@{k}": [] for k in ks}
    totals.update({f"recall@{k}": [] for k in ks})
    totals.update({f"ndcg@{k}": [] for k in ks})
    totals.update({f"map@{k}": [] for k in ks})
    auc_labels: list[int] = []
    auc_scores: list[float] = []

    for target, labels in cases.items():
        ranked = sorted(labels, key=lambda item: score_matrix[target, item[0]], reverse=True)
        positives = sum(label for _, label in labels)
        if positives == 0:
            continue
        auc_labels.extend(label for _, label in labels)
        auc_scores.extend(float(score_matrix[target, candidate]) for candidate, _ in labels)
        for k in ks:
            top = ranked[:k]
            hits = [label for _, label in top]
            precision = sum(hits) / k
            recall = sum(hits) / positives
            dcg = sum(label / math.log2(rank + 2) for rank, label in enumerate(hits))
            ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(positives, k)))
            avg_precision_terms = []
            hit_count = 0
            for rank, label in enumerate(hits, start=1):
                if label:
                    hit_count += 1
                    avg_precision_terms.append(hit_count / rank)
            totals[f"precision@{k}"].append(precision)
            totals[f"recall@{k}"].append(recall)
            totals[f"ndcg@{k}"].append(dcg / ideal if ideal else 0.0)
            totals[f"map@{k}"].append(sum(avg_precision_terms) / min(positives, k) if positives else 0.0)

    metrics = {name: float(np.mean(values)) if values else 0.0 for name, values in totals.items()}
    if len(set(auc_labels)) == 2:
        metrics["auc"] = float(roc_auc_score(auc_labels, auc_scores))
    else:
        metrics["auc"] = 0.0
    metrics["evaluated_targets"] = float(len(cases))
    return metrics


def tune_hybrid_weights(
    components: dict[str, np.ndarray],
    validation_cases: dict[int, list[tuple[int, int]]],
) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    grids = [
        {"artist": 0.45, "tag": 0.20, "niche": 0.20, "discovery": 0.15},
        {"artist": 0.35, "tag": 0.25, "niche": 0.25, "discovery": 0.15},
        {"artist": 0.30, "tag": 0.20, "niche": 0.30, "discovery": 0.20},
        {"artist": 0.25, "tag": 0.25, "niche": 0.25, "discovery": 0.25},
        {"artist": 0.55, "tag": 0.15, "niche": 0.20, "discovery": 0.10},
    ]
    best_weights = grids[0]
    best_score = -1.0
    best_matrix = None
    best_metrics: dict[str, float] = {}
    for weights in grids:
        matrix = sum(weights[name] * components[name] for name in weights)
        np.fill_diagonal(matrix, -np.inf)
        metrics = rank_metrics(matrix, validation_cases)
        score = metrics["ndcg@10"]
        if score > best_score:
            best_score = score
            best_weights = weights
            best_matrix = matrix
            best_metrics = metrics
    assert best_matrix is not None
    return best_weights, best_matrix.astype(np.float32), best_metrics


def pair_features(
    pairs: list[tuple[int, int]],
    components: dict[str, np.ndarray],
    user_artist_raw: sparse.csr_matrix,
    user_tag_norm: sparse.csr_matrix,
    train_adj: dict[int, set[int]],
) -> np.ndarray:
    artist_counts = np.asarray((user_artist_raw > 0).sum(axis=1)).ravel()
    listen_mass = np.asarray(user_artist_raw.sum(axis=1)).ravel()
    tag_counts = np.asarray((user_tag_norm > 0).sum(axis=1)).ravel()
    degrees = np.array([len(train_adj[idx]) for idx in range(user_artist_raw.shape[0])])
    rows = []
    for u, v in pairs:
        rows.append(
            [
                components["artist"][u, v],
                components["tag"][u, v],
                components["niche"][u, v],
                components["discovery"][u, v],
                abs(artist_counts[u] - artist_counts[v]) / max(artist_counts.max(), 1),
                abs(tag_counts[u] - tag_counts[v]) / max(tag_counts.max(), 1),
                math.log1p(degrees[v]),
                math.log1p(listen_mass[v]),
            ]
        )
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def train_learned_ranker(
    train_edges: list[tuple[int, int]],
    all_edges: list[tuple[int, int]],
    dataset: Dataset,
    components: dict[str, np.ndarray],
    min_artists: int = 10,
    min_tags: int = 3,
    negatives_per_positive: int = 10,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = random.Random(seed)
    train_adj = adjacency_from_edges(train_edges, dataset.user_to_idx)
    all_adj = adjacency_from_edges(all_edges, dataset.user_to_idx)
    artist_counts = np.asarray((dataset.user_artist_raw > 0).sum(axis=1)).ravel()
    tag_counts = np.asarray((dataset.user_tag_norm > 0).sum(axis=1)).ravel()
    reliable = {idx for idx in range(len(dataset.users)) if artist_counts[idx] >= min_artists and tag_counts[idx] >= min_tags}
    all_idxs = sorted(reliable)

    positive_pairs = []
    for u, v in train_edges:
        ui = dataset.user_to_idx[u]
        vi = dataset.user_to_idx[v]
        if ui in reliable and vi in reliable:
            positive_pairs.extend([(ui, vi), (vi, ui)])

    negative_pairs = []
    for u, _ in positive_pairs:
        unavailable = all_adj[u] | {u}
        candidates = [idx for idx in all_idxs if idx not in unavailable]
        for _ in range(negatives_per_positive):
            if candidates:
                negative_pairs.append((u, rng.choice(candidates)))

    pairs = positive_pairs + negative_pairs
    labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))
    features = pair_features(pairs, components, dataset.user_artist_raw, dataset.user_tag_norm, train_adj)

    if len(set(labels)) < 2:
        model = LogisticRegression(max_iter=1000)
    else:
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07, random_state=seed)
    model.fit(features, labels)

    score = np.full((len(dataset.users), len(dataset.users)), -np.inf, dtype=np.float32)
    for u in range(len(dataset.users)):
        candidates = [v for v in range(len(dataset.users)) if v != u]
        feats = pair_features([(u, v) for v in candidates], components, dataset.user_artist_raw, dataset.user_tag_norm, train_adj)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feats)[:, 1]
        else:
            probs = model.decision_function(feats)
        score[u, candidates] = probs

    summary = {
        "training_positive_pairs": float(len(positive_pairs)),
        "training_negative_pairs": float(len(negative_pairs)),
        "reliable_users": float(len(reliable)),
        "min_artists": float(min_artists),
        "min_tags": float(min_tags),
    }
    return score, summary


def top_match_examples(
    score_matrix: np.ndarray,
    dataset: Dataset,
    components: dict[str, np.ndarray],
    count: int = 8,
) -> pd.DataFrame:
    artist_matrix = dataset.user_artist_raw.tocsr()
    tag_matrix = dataset.user_tag_norm.tocsr()
    rows = []
    seen_targets = set()
    flat_order = np.dstack(np.unravel_index(np.argsort(score_matrix.ravel())[::-1], score_matrix.shape))[0]
    for u, v in flat_order:
        if u == v or u in seen_targets:
            continue
        seen_targets.add(int(u))
        shared_artists = sorted(set(artist_matrix[u].indices) & set(artist_matrix[v].indices))[:5]
        target_top = artist_matrix[u].toarray().ravel().argsort()[::-1][:5]
        candidate_top = artist_matrix[v].toarray().ravel().argsort()[::-1][:5]
        target_tags = tag_matrix[u].toarray().ravel().argsort()[::-1][:5]
        candidate_tags = tag_matrix[v].toarray().ravel().argsort()[::-1][:5]
        rows.append(
            {
                "target_user": dataset.idx_to_user[int(u)],
                "matched_user": dataset.idx_to_user[int(v)],
                "score": float(score_matrix[u, v]),
                "artist_similarity": float(components["artist"][u, v]),
                "tag_similarity": float(components["tag"][u, v]),
                "niche_overlap": float(components["niche"][u, v]),
                "discovery": float(components["discovery"][u, v]),
                "shared_artists": "; ".join(dataset.artist_names.get(dataset.idx_to_artist[idx], str(dataset.idx_to_artist[idx])) for idx in shared_artists),
                "target_top_artists": "; ".join(dataset.artist_names.get(dataset.idx_to_artist[idx], str(dataset.idx_to_artist[idx])) for idx in target_top),
                "match_top_artists": "; ".join(dataset.artist_names.get(dataset.idx_to_artist[idx], str(dataset.idx_to_artist[idx])) for idx in candidate_top),
                "target_top_tags": "; ".join(dataset.tag_names.get(dataset.idx_to_tag[int(idx)], str(dataset.idx_to_tag[int(idx)])) for idx in target_tags),
                "match_top_tags": "; ".join(dataset.tag_names.get(dataset.idx_to_tag[int(idx)], str(dataset.idx_to_tag[int(idx)])) for idx in candidate_tags),
            }
        )
        if len(rows) >= count:
            break
    return pd.DataFrame(rows)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
