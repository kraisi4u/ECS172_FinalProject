from __future__ import annotations

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


def load_hetrec(raw_dir: Path) -> Dataset:
    return HetrecLoader(raw_dir).load()
