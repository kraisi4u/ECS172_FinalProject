import unittest

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
