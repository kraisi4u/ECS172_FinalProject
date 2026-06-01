import unittest

import numpy as np
import pandas as pd

from src.data_loader import filter_tags_by_proposal_rules
from src.music_match import (
    build_undirected_edges,
    compute_niche_overlap,
    filtered_ground_truth_pairs,
    friend_random_similarity_summary,
    split_friend_edges_by_user,
)


class MusicMatchTests(unittest.TestCase):
    def test_build_undirected_edges_deduplicates_reversed_pairs(self):
        raw = pd.DataFrame(
            {
                "userID": [1, 2, 2, 3],
                "friendID": [2, 1, 3, 2],
            }
        )

        edges = build_undirected_edges(raw)

        self.assertEqual(edges, [(1, 2), (2, 3)])

    def test_split_friend_edges_keeps_each_user_holdout_when_possible(self):
        edges = [(1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 5)]

        train, validation, test = split_friend_edges_by_user(
            edges, validation_ratio=0.2, test_ratio=0.2, seed=7
        )

        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(validation).isdisjoint(test))
        self.assertEqual(set(train) | set(validation) | set(test), set(edges))
        self.assertGreater(len(validation), 0)
        self.assertGreater(len(test), 0)

    def test_split_friend_edges_uses_global_ratios_on_dense_graph(self):
        edges = [(u, v) for u in range(1, 9) for v in range(u + 1, 9)]

        train, validation, test = split_friend_edges_by_user(
            edges, validation_ratio=0.25, test_ratio=0.25, seed=3
        )

        self.assertGreaterEqual(len(validation), 5)
        self.assertGreaterEqual(len(test), 5)
        self.assertGreater(len(train), len(validation))

    def test_niche_overlap_is_normalized_by_target_niche_mass(self):
        matrix = np.array(
            [
                [3.0, 0.0, 7.0, 0.0],
                [1.0, 5.0, 7.0, 0.0],
                [0.0, 0.0, 7.0, 9.0],
            ]
        )
        niche_mask = np.array([True, True, False, True])

        score = compute_niche_overlap(matrix, 0, 1, niche_mask)

        self.assertAlmostEqual(score, 1.0 / 3.0)
        self.assertAlmostEqual(compute_niche_overlap(matrix, 0, 2, niche_mask), 0.0)

    def test_friend_random_similarity_summary_excludes_friend_negatives(self):
        sim = np.array(
            [
                [-np.inf, 0.8, 0.1],
                [0.8, -np.inf, 0.2],
                [0.1, 0.2, -np.inf],
            ]
        )
        edges = [(10, 20)]
        user_to_idx = {10: 0, 20: 1, 30: 2}

        summary = friend_random_similarity_summary(sim, edges, user_to_idx, sample_size=2, seed=1)

        self.assertAlmostEqual(summary["friend_mean_similarity"], 0.8)
        self.assertLess(summary["random_nonfriend_mean_similarity"], 0.8)

    def test_filtered_ground_truth_pairs_uses_margin_thresholds(self):
        sim = np.array(
            [
                [-np.inf, 0.8, 0.01, 0.2],
                [0.8, -np.inf, 0.03, 0.01],
                [0.01, 0.03, -np.inf, 0.9],
                [0.2, 0.01, 0.9, -np.inf],
            ]
        )
        edges = [(10, 20), (30, 40)]
        user_to_idx = {10: 0, 20: 1, 30: 2, 40: 3}

        pairs = filtered_ground_truth_pairs(
            sim,
            edges,
            user_to_idx,
            positive_threshold=0.75,
            negative_threshold=0.02,
            seed=2,
        )

        self.assertEqual(set(pairs["label"]), {0, 1})
        self.assertTrue((pairs[pairs["label"] == 1]["taste_similarity"] >= 0.75).all())
        self.assertTrue((pairs[pairs["label"] == 0]["taste_similarity"] <= 0.02).all())

    def test_filter_tags_by_proposal_rules_filters_noisy_tags_and_keeps_ambiguous_useful_tags(self):
        user_tags = pd.DataFrame(
            [
                {"userID": 1, "artistID": 10, "tagID": 1},
                {"userID": 2, "artistID": 11, "tagID": 1},
                {"userID": 3, "artistID": 12, "tagID": 1},
                {"userID": 4, "artistID": 13, "tagID": 1},
                {"userID": 5, "artistID": 14, "tagID": 1},
                {"userID": 1, "artistID": 15, "tagID": 1},
                {"userID": 2, "artistID": 16, "tagID": 1},
                {"userID": 3, "artistID": 17, "tagID": 1},
                {"userID": 4, "artistID": 18, "tagID": 1},
                {"userID": 5, "artistID": 19, "tagID": 1},
                {"userID": 1, "artistID": 20, "tagID": 1},
                {"userID": 2, "artistID": 21, "tagID": 1},
                {"userID": 3, "artistID": 22, "tagID": 1},
                {"userID": 4, "artistID": 23, "tagID": 1},
                {"userID": 5, "artistID": 24, "tagID": 1},
                {"userID": 1, "artistID": 25, "tagID": 1},
                {"userID": 2, "artistID": 26, "tagID": 1},
                {"userID": 3, "artistID": 27, "tagID": 1},
                {"userID": 4, "artistID": 28, "tagID": 1},
                {"userID": 5, "artistID": 29, "tagID": 1},
                {"userID": 1, "artistID": 30, "tagID": 2},
                {"userID": 2, "artistID": 31, "tagID": 2},
                {"userID": 3, "artistID": 32, "tagID": 2},
                {"userID": 4, "artistID": 33, "tagID": 2},
                {"userID": 5, "artistID": 34, "tagID": 2},
                {"userID": 1, "artistID": 35, "tagID": 2},
                {"userID": 2, "artistID": 36, "tagID": 2},
                {"userID": 3, "artistID": 37, "tagID": 2},
                {"userID": 4, "artistID": 38, "tagID": 2},
                {"userID": 5, "artistID": 39, "tagID": 2},
                {"userID": 1, "artistID": 40, "tagID": 2},
                {"userID": 2, "artistID": 41, "tagID": 2},
                {"userID": 3, "artistID": 42, "tagID": 2},
                {"userID": 4, "artistID": 43, "tagID": 2},
                {"userID": 5, "artistID": 44, "tagID": 2},
                {"userID": 1, "artistID": 45, "tagID": 2},
                {"userID": 2, "artistID": 46, "tagID": 2},
                {"userID": 3, "artistID": 47, "tagID": 2},
                {"userID": 4, "artistID": 48, "tagID": 2},
                {"userID": 5, "artistID": 49, "tagID": 2},
                {"userID": 1, "artistID": 50, "tagID": 3},
                {"userID": 2, "artistID": 51, "tagID": 3},
                {"userID": 3, "artistID": 52, "tagID": 3},
                {"userID": 4, "artistID": 53, "tagID": 3},
                {"userID": 5, "artistID": 54, "tagID": 3},
                {"userID": 1, "artistID": 55, "tagID": 3},
                {"userID": 2, "artistID": 56, "tagID": 3},
                {"userID": 3, "artistID": 57, "tagID": 3},
                {"userID": 4, "artistID": 58, "tagID": 3},
                {"userID": 5, "artistID": 59, "tagID": 3},
                {"userID": 1, "artistID": 60, "tagID": 3},
                {"userID": 2, "artistID": 61, "tagID": 3},
                {"userID": 3, "artistID": 62, "tagID": 3},
                {"userID": 4, "artistID": 63, "tagID": 3},
                {"userID": 5, "artistID": 64, "tagID": 3},
                {"userID": 1, "artistID": 65, "tagID": 3},
                {"userID": 2, "artistID": 66, "tagID": 3},
                {"userID": 3, "artistID": 67, "tagID": 3},
                {"userID": 4, "artistID": 68, "tagID": 3},
                {"userID": 5, "artistID": 69, "tagID": 3},
                {"userID": 1, "artistID": 70, "tagID": 4},
                {"userID": 2, "artistID": 71, "tagID": 4},
                {"userID": 3, "artistID": 72, "tagID": 4},
            ]
        )
        tags = pd.DataFrame(
            [
                {"tagID": 1, "tagValue": "acoustic"},
                {"tagID": 2, "tagValue": "seen live"},
                {"tagID": 3, "tagValue": "favorites"},
                {"tagID": 4, "tagValue": "tiny"},
            ]
        )

        cleaned_user_tags, cleaned_tags = filter_tags_by_proposal_rules(user_tags, tags)

        self.assertEqual(set(cleaned_tags["tagValue"]), {"acoustic"})
        self.assertEqual(set(cleaned_user_tags["tagID"]), {1})


if __name__ == "__main__":
    unittest.main()
