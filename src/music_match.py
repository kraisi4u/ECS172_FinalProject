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


def friend_matrix(train_edges: list[tuple[int, int]], user_to_idx: dict[int, int]) -> sparse.csr_matrix:
    rows = []
    cols = []
    for u, v in train_edges:
        if u in user_to_idx and v in user_to_idx:
            ui = user_to_idx[u]
            vi = user_to_idx[v]
            rows.extend([ui, vi])
            cols.extend([vi, ui])
    data = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_to_idx), len(user_to_idx)))
    return normalize(matrix, norm="l2", axis=1)


def svd_embedding(matrix: sparse.spmatrix, n_components: int, seed: int = RNG_SEED) -> np.ndarray:
    limit = min(matrix.shape) - 1
    if limit < 1:
        return np.zeros((matrix.shape[0], 1), dtype=np.float32)
    n_components = min(n_components, limit)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    emb = svd.fit_transform(matrix).astype(np.float32)
    return normalize(emb, norm="l2", axis=1).astype(np.float32)


def dual_space_initial_embeddings(
    dataset: Dataset,
    train_edges: list[tuple[int, int]],
    n_components: int = 32,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_friend_norm = friend_matrix(train_edges, dataset.user_to_idx)
    artist_binary = (dataset.user_artist_raw > 0).astype(np.float32)
    listener_counts = np.asarray(artist_binary.sum(axis=0)).ravel()
    artist_idf = np.log1p((dataset.user_artist_raw.shape[0] + 1) / np.maximum(listener_counts, 1.0))
    rarity_artist = normalize(dataset.user_artist_norm.multiply(artist_idf), norm="l2", axis=1)

    taste_features = sparse.hstack(
        [dataset.user_artist_norm, dataset.user_tag_norm * 0.45, train_friend_norm * 0.30],
        format="csr",
    )
    curator_features = sparse.hstack(
        [rarity_artist, dataset.user_tag_norm * 0.30, train_friend_norm * 0.20],
        format="csr",
    )

    taste = svd_embedding(taste_features, n_components, seed)
    curator = svd_embedding(curator_features, n_components, seed + 1)
    seeker = taste.copy()
    return taste, seeker, curator


def dual_space_score_matrix(
    taste: np.ndarray,
    seeker: np.ndarray,
    curator: np.ndarray,
    comfort_weight: float = 0.45,
    complement_weight: float = 0.45,
    reciprocal_weight: float = 0.10,
    candidate_bias: np.ndarray | None = None,
    normalize_scores: bool = True,
) -> np.ndarray:
    score = (
        comfort_weight * (taste @ taste.T)
        + complement_weight * (seeker @ curator.T)
        + reciprocal_weight * (curator @ seeker.T)
    ).astype(np.float32)
    if candidate_bias is not None:
        score += candidate_bias[None, :].astype(np.float32)
    np.fill_diagonal(score, -np.inf)
    if normalize_scores:
        score = minmax_rows(score)
        np.fill_diagonal(score, -np.inf)
    return score.astype(np.float32)


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


def tune_dual_space_blend(
    dual_score: np.ndarray,
    components: dict[str, np.ndarray],
    validation_cases: dict[int, list[tuple[int, int]]],
) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    blend_options = [
        {"dual": 1.00, "artist": 0.00, "tag": 0.00, "niche": 0.00, "discovery": 0.00},
        {"dual": 0.80, "artist": 0.05, "tag": 0.05, "niche": 0.05, "discovery": 0.05},
        {"dual": 0.70, "artist": 0.10, "tag": 0.05, "niche": 0.10, "discovery": 0.05},
        {"dual": 0.60, "artist": 0.20, "tag": 0.05, "niche": 0.10, "discovery": 0.05},
        {"dual": 0.50, "artist": 0.30, "tag": 0.05, "niche": 0.10, "discovery": 0.05},
        {"dual": 0.45, "artist": 0.35, "tag": 0.05, "niche": 0.10, "discovery": 0.05},
    ]
    best_weights = blend_options[0]
    best_score = -1.0
    best_matrix = None
    best_metrics: dict[str, float] = {}
    for weights in blend_options:
        matrix = weights["dual"] * dual_score
        for name in ("artist", "tag", "niche", "discovery"):
            matrix = matrix + weights[name] * components[name]
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


def train_dual_space_matcher(
    train_edges: list[tuple[int, int]],
    all_edges: list[tuple[int, int]],
    dataset: Dataset,
    components: dict[str, np.ndarray],
    validation_cases: dict[int, list[tuple[int, int]]],
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, dict[str, object]]:
    configs = [
        {"dim": 24, "epochs": 12, "lr": 0.035, "reg": 0.0015, "comfort": 0.50, "complement": 0.40, "reciprocal": 0.10},
    ]
    all_adj = adjacency_from_edges(all_edges, dataset.user_to_idx)
    n_users = len(dataset.users)
    directed_positive_pairs = []
    for u, v in train_edges:
        ui = dataset.user_to_idx[u]
        vi = dataset.user_to_idx[v]
        directed_positive_pairs.extend([(ui, vi), (vi, ui)])

    candidate_pool = []
    all_idxs = np.arange(n_users, dtype=np.int32)
    for u in range(n_users):
        unavailable = all_adj[u] | {u}
        candidates = np.array([idx for idx in all_idxs if int(idx) not in unavailable], dtype=np.int32)
        candidate_pool.append(candidates)

    best_score_matrix = None
    best_summary: dict[str, object] = {}
    best_validation = -1.0

    for config_idx, config in enumerate(configs):
        rng = np.random.default_rng(seed + config_idx)
        taste, seeker, curator = dual_space_initial_embeddings(dataset, train_edges, int(config["dim"]), seed + config_idx)
        candidate_bias = np.zeros(n_users, dtype=np.float32)
        positive_order = np.arange(len(directed_positive_pairs))

        for _ in range(int(config["epochs"])):
            rng.shuffle(positive_order)
            for start in range(0, len(positive_order), 512):
                batch = positive_order[start : start + 512]
                users = np.array([directed_positive_pairs[int(pos_idx)][0] for pos_idx in batch], dtype=np.int32)
                positives = np.array([directed_positive_pairs[int(pos_idx)][1] for pos_idx in batch], dtype=np.int32)
                negatives = np.array(
                    [int(rng.choice(candidate_pool[int(u)])) for u in users if len(candidate_pool[int(u)]) > 0],
                    dtype=np.int32,
                )
                if len(negatives) != len(users):
                    keep = np.array([len(candidate_pool[int(u)]) > 0 for u in users], dtype=bool)
                    users = users[keep]
                    positives = positives[keep]
                    if len(users) == 0:
                        continue
                    negatives = np.array([int(rng.choice(candidate_pool[int(u)])) for u in users], dtype=np.int32)

                taste_u = taste[users].copy()
                taste_v = taste[positives].copy()
                taste_neg = taste[negatives].copy()
                seeker_u = seeker[users].copy()
                seeker_v = seeker[positives].copy()
                seeker_neg = seeker[negatives].copy()
                curator_u = curator[users].copy()
                curator_v = curator[positives].copy()
                curator_neg = curator[negatives].copy()

                comfort = float(config["comfort"])
                complement = float(config["complement"])
                reciprocal = float(config["reciprocal"])
                pos_score = (
                    comfort * np.sum(taste_u * taste_v, axis=1)
                    + complement * np.sum(seeker_u * curator_v, axis=1)
                    + reciprocal * np.sum(curator_u * seeker_v, axis=1)
                    + candidate_bias[positives]
                )
                neg_score = (
                    comfort * np.sum(taste_u * taste_neg, axis=1)
                    + complement * np.sum(seeker_u * curator_neg, axis=1)
                    + reciprocal * np.sum(curator_u * seeker_neg, axis=1)
                    + candidate_bias[negatives]
                )
                coeff = (1.0 / (1.0 + np.exp(np.clip(pos_score - neg_score, -30.0, 30.0)))).astype(np.float32)
                lr = float(config["lr"])
                reg = float(config["reg"])
                coeff_col = coeff[:, None]

                np.add.at(taste, users, lr * (coeff_col * comfort * (taste_v - taste_neg) - reg * taste_u))
                np.add.at(taste, positives, lr * (coeff_col * comfort * taste_u - reg * taste_v))
                np.add.at(taste, negatives, lr * (-coeff_col * comfort * taste_u - reg * taste_neg))

                np.add.at(seeker, users, lr * (coeff_col * complement * (curator_v - curator_neg) - reg * seeker_u))
                np.add.at(curator, positives, lr * (coeff_col * complement * seeker_u - reg * curator_v))
                np.add.at(curator, negatives, lr * (-coeff_col * complement * seeker_u - reg * curator_neg))

                np.add.at(curator, users, lr * (coeff_col * reciprocal * (seeker_v - seeker_neg) - reg * curator_u))
                np.add.at(seeker, positives, lr * (coeff_col * reciprocal * curator_u - reg * seeker_v))
                np.add.at(seeker, negatives, lr * (-coeff_col * reciprocal * curator_u - reg * seeker_neg))

                np.add.at(candidate_bias, positives, lr * (coeff - reg * candidate_bias[positives]))
                np.add.at(candidate_bias, negatives, lr * (-coeff - reg * candidate_bias[negatives]))

        taste = normalize(taste, norm="l2", axis=1).astype(np.float32)
        seeker = normalize(seeker, norm="l2", axis=1).astype(np.float32)
        curator = normalize(curator, norm="l2", axis=1).astype(np.float32)
        dual_score = dual_space_score_matrix(
            taste,
            seeker,
            curator,
            comfort_weight=float(config["comfort"]),
            complement_weight=float(config["complement"]),
            reciprocal_weight=float(config["reciprocal"]),
            candidate_bias=candidate_bias,
        )
        blend_weights, blended_score, validation_metrics = tune_dual_space_blend(dual_score, components, validation_cases)
        validation_score = validation_metrics["ndcg@10"]
        if validation_score > best_validation:
            best_validation = validation_score
            best_score_matrix = blended_score
            best_summary = {
                "config": config,
                "blend_weights": blend_weights,
                "validation_metrics": validation_metrics,
                "training_positive_pairs": float(len(directed_positive_pairs)),
                "negative_candidates_mean": float(np.mean([len(candidates) for candidates in candidate_pool])),
            }

    assert best_score_matrix is not None
    return best_score_matrix.astype(np.float32), best_summary


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
