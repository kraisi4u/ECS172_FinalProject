from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class Dataset:
    users: np.ndarray
    artists: np.ndarray
    user_to_idx: dict[int, int]
    idx_to_user: dict[int, int]
    artist_to_idx: dict[int, int]
    idx_to_artist: dict[int, int]
    idx_to_tag: dict[int, int]
    artist_names: dict[int, str]
    tag_names: dict[int, str]
    user_artist_raw: sparse.csr_matrix
    user_artist_norm: sparse.csr_matrix
    user_tag_norm: sparse.csr_matrix
    edges: list[tuple[int, int]]


def build_undirected_edges(friend_df: pd.DataFrame) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for u, v in friend_df[["userID", "friendID"]].itertuples(index=False):
        u_int = int(u)
        v_int = int(v)
        if u_int == v_int:
            continue
        edges.add((min(u_int, v_int), max(u_int, v_int)))
    return sorted(edges)


class BaseHetrecLoader:
    """Subclass this loader to swap in custom cleaning or preprocessing."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def table(self, name: str, **kwargs: object) -> pd.DataFrame:
        dat_path = self.raw_dir / f"{name}.dat"
        csv_path = self.raw_dir / f"{name}.csv"
        path = dat_path if dat_path.exists() else csv_path
        return pd.read_csv(path, sep="\t", **kwargs)

    def load_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "user_artists": self.table("user_artists"),
            "user_tags": self.table("user_taggedartists"),
            "friends": self.table("user_friends"),
            "artists": self.table("artists"),
            "tags": self.table("tags", encoding="latin1"),
        }

    def preprocess_user_artists(
        self,
        user_artists: pd.DataFrame,
        user_to_idx: dict[int, int],
        artist_to_idx: dict[int, int],
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        ua_rows = user_artists.userID.map(user_to_idx).to_numpy()
        ua_cols = user_artists.artistID.map(artist_to_idx).to_numpy()
        ua_data = np.log1p(user_artists.weight.to_numpy(dtype=float))
        user_artist_raw = sparse.csr_matrix((ua_data, (ua_rows, ua_cols)), shape=(len(user_to_idx), len(artist_to_idx)))
        user_artist_norm = normalize(user_artist_raw, norm="l2", axis=1)
        return user_artist_raw, user_artist_norm

    def preprocess_user_tags(
        self,
        user_tags: pd.DataFrame,
        user_to_idx: dict[int, int],
        tag_to_idx: dict[int, int],
    ) -> sparse.csr_matrix:
        tag_counts = user_tags.groupby(["userID", "tagID"]).size().reset_index(name="count")
        ut_rows = tag_counts.userID.map(user_to_idx).to_numpy()
        ut_cols = tag_counts.tagID.map(tag_to_idx).to_numpy()
        user_tag_raw = sparse.csr_matrix((tag_counts["count"].to_numpy(dtype=float), (ut_rows, ut_cols)), shape=(len(user_to_idx), len(tag_to_idx)))
        return normalize(user_tag_raw, norm="l2", axis=1)

    def build_dataset(self, tables: dict[str, pd.DataFrame]) -> Dataset:
        user_artists = tables["user_artists"]
        user_tags = tables["user_tags"]
        friends = tables["friends"]
        artists = tables["artists"]
        tags = tables["tags"]

        users = np.array(sorted(set(user_artists.userID) | set(friends.userID) | set(friends.friendID)))
        artist_ids = np.array(sorted(user_artists.artistID.unique()))
        tag_ids = np.array(sorted(user_tags.tagID.unique()))

        user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
        artist_to_idx = {int(artist_id): idx for idx, artist_id in enumerate(artist_ids)}
        tag_to_idx = {int(tag_id): idx for idx, tag_id in enumerate(tag_ids)}

        user_artist_raw, user_artist_norm = self.preprocess_user_artists(user_artists, user_to_idx, artist_to_idx)
        user_tag_norm = self.preprocess_user_tags(user_tags, user_to_idx, tag_to_idx)

        artist_id_column = "artistID" if "artistID" in artists.columns else "id"
        artist_names = dict(zip(artists[artist_id_column].astype(int), artists.name.astype(str)))
        tag_names = dict(zip(tags.tagID.astype(int), tags.tagValue.astype(str)))

        return Dataset(
            users=users,
            artists=artist_ids,
            user_to_idx=user_to_idx,
            idx_to_user={idx: int(user_id) for user_id, idx in user_to_idx.items()},
            artist_to_idx=artist_to_idx,
            idx_to_artist={idx: int(artist_id) for artist_id, idx in artist_to_idx.items()},
            idx_to_tag={idx: int(tag_id) for tag_id, idx in tag_to_idx.items()},
            artist_names=artist_names,
            tag_names=tag_names,
            user_artist_raw=user_artist_raw,
            user_artist_norm=user_artist_norm,
            user_tag_norm=user_tag_norm,
            edges=build_undirected_edges(friends),
        )

    def load(self) -> Dataset:
        return self.build_dataset(self.load_tables())


class HetrecLoader(BaseHetrecLoader):
    """Default loader that preserves the project's current raw-data behavior."""


AMBIGUOUS_USEFUL_TAGS = {
    "singer-songwriter",
    "soundtrack",
    "acoustic",
    "chillout",
    "metalcore",
    "synthpop",
    "downtempo",
    "idm",
}

PROPOSAL_BAD_TAG_PATTERNS = (
    re.compile(r"(?:^|\b)(?:19|20)\d{2}(?:\b|$)"),
    re.compile(r"(?:^|\b)(?:19|20)?\d0s(?:\b|$)"),
    re.compile(r"(?:^|\b)\d{2}s(?:\b|$)"),
    re.compile(r"\blive\b"),
    re.compile(r"\b(?:fav|favs|favorite|favorites|favourite|favourites)\b"),
    re.compile(r"\b(?:best|awesome|amazing|beautiful|cool|good music|great|love|hate|sexy|lovely)\b"),
    re.compile(r"\b(?:my|songs?|albums?|artists?|tracks?|music)\s+(?:i\s+)?(?:own|like|love|want|need)\b"),
    re.compile(r"\b(?:my\s+(?:fav|favorite|favourite)|playlist|wish\s*list|under\s+\d+\s+listeners)\b"),
)


def filter_tags_by_proposal_rules(
    user_tags: pd.DataFrame,
    tags: pd.DataFrame,
    min_assignments: int = 20,
    min_users: int = 5,
    min_artists: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the proposal's tag cleaning rules to the raw tag tables."""

    tag_stats = (
        user_tags.groupby("tagID")
        .agg(
            assignments=("userID", "size"),
            users=("userID", "nunique"),
            artists=("artistID", "nunique"),
        )
        .reset_index()
    )
    tags_with_stats = tags.merge(tag_stats, on="tagID", how="left").fillna({"assignments": 0, "users": 0, "artists": 0})
    tags_with_stats["assignments"] = tags_with_stats["assignments"].astype(int)
    tags_with_stats["users"] = tags_with_stats["users"].astype(int)
    tags_with_stats["artists"] = tags_with_stats["artists"].astype(int)

    normalized = tags_with_stats["tagValue"].astype(str).str.strip().str.lower()
    keep_mask = (
        (tags_with_stats["assignments"] >= min_assignments)
        & (tags_with_stats["users"] >= min_users)
        & (tags_with_stats["artists"] >= min_artists)
    )
    keep_mask &= ~normalized.apply(
        lambda value: value not in AMBIGUOUS_USEFUL_TAGS and any(pattern.search(value) for pattern in PROPOSAL_BAD_TAG_PATTERNS)
    )

    cleaned_tags = tags_with_stats.loc[keep_mask, ["tagID", "tagValue"]].reset_index(drop=True)
    cleaned_user_tags = user_tags[user_tags["tagID"].isin(cleaned_tags["tagID"])].copy()
    return cleaned_user_tags, cleaned_tags


class ProposalCleanHetrecLoader(HetrecLoader):
    """Loader that applies the proposal's tag-cleaning rules before vectorization."""

    def load_tables(self) -> dict[str, pd.DataFrame]:
        tables = super().load_tables()
        cleaned_user_tags, cleaned_tags = filter_tags_by_proposal_rules(tables["user_tags"], tables["tags"])
        tables["user_tags"] = cleaned_user_tags
        tables["tags"] = cleaned_tags
        return tables

    def preprocess_user_tags(
        self,
        user_tags: pd.DataFrame,
        user_to_idx: dict[int, int],
        tag_to_idx: dict[int, int],
    ) -> sparse.csr_matrix:
        # Count distinct artists per user-tag pair to reduce repeat-tagging skew.
        tag_counts = user_tags.groupby(["userID", "tagID"])["artistID"].nunique().reset_index(name="count")
        ut_rows = tag_counts.userID.map(user_to_idx).to_numpy()
        ut_cols = tag_counts.tagID.map(tag_to_idx).to_numpy()
        user_tag_raw = sparse.csr_matrix((tag_counts["count"].to_numpy(dtype=float), (ut_rows, ut_cols)), shape=(len(user_to_idx), len(tag_to_idx)))
        return normalize(user_tag_raw, norm="l2", axis=1)


def load_hetrec(raw_dir: Path) -> Dataset:
    return HetrecLoader(raw_dir).load()


def load_hetrec_proposal_cleaned(raw_dir: Path) -> Dataset:
    return ProposalCleanHetrecLoader(raw_dir).load()
