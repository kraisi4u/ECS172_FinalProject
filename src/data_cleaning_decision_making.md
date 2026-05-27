# Data Decision Making

## 1. Rules for cleaning tags

1. Reliability filter:
- `assignments >= 20`
- `users >= 5`
- `artists >= 5`

2. Drop obvious bad tags:
   years, decades, "seen live", favorites, personal lists, pure opinions

3. Review ambiguous useful tags:
   singer-songwriter, soundtrack, acoustic, chillout, metalcore, synthpop, downtempo, idm

4. Use unique listeners per artist - a few heavy users can skew data. 

### Model-Agnostic Cleaned Files:

Propose 5 cleaned .csv profiles, which are relational tables nested in `dataset/clean/`.

1. `user_artists.csv`
2. `user_taggedartists.csv`
3. `user_friends.csv`
4. `tags.csv`
5. `artists.csv`


## 2. User Setup

Each user gets two profiles, the artist and the tag.
1. Artist profile:
   `what artists they listen to, weighted by play count`

2. Tag profile:
   `what genres/styles/moods describe the artists they tag`


## 3. Feature Setup

Propose 5 feature files in `dataset/features/`, which are as follows: 
1. `user_artist_features.csv`
2. `user_tag_features.csv`
3. `artist_popularity.csv`
4. `niche_artists.csv`
5. `friend_edges_undirected.csv`. 

These are specifically compiled for Model Architecture 1 described in our proposal, with different feature setups being required for a different model.


## 4. Data Loader Setup

`src/data_loader.py` loads the cleaned tables, feature files, and tag reports without changing the data.

The loader exposes:

1. Clean relational tables through `load_clean_tables()`
2. Model-ready feature files through `load_features()`
3. Tag reports through `load_tag_reports()`
4. Sparse user artist vectors through `artist_vectors()`
5. Sparse user tag vectors through `tag_vectors()`
6. Friend labels through `friend_edge_set()`
7. Niche artist IDs through `niche_artist_ids()`

The artist and tag vectors are dictionaries keyed by `userID`. Each user's vector is another dictionary keyed by `artistID` or `tagID`.

Example:

```python
from data_loader import LastFMLoader

loader = LastFMLoader()
artist_vectors = loader.artist_vectors()
tag_vectors = loader.tag_vectors()
friend_edges = loader.friend_edge_set()

u = 2
v = 275

artist_sim = loader.cosine_from_vectors(
    artist_vectors.get(u, {}),
    artist_vectors.get(v, {}),
)

tag_sim = loader.cosine_from_vectors(
    tag_vectors.get(u, {}),
    tag_vectors.get(v, {}),
)

is_friend = loader.normalize_edge(u, v) in friend_edges
```

This setup supports the proposed hybrid model directly, while still leaving the cleaned CSVs available for other architectures.

