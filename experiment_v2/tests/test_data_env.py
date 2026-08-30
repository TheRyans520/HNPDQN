"""Unit tests for leakage-safe data, environment, and transparent baselines."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.baselines import (
    ClairvoyantOraclePolicy,
    JammerAwareGreedyPolicy,
    MinInterferencePolicy,
    OraclePolicy,
    RandomPolicy,
    ScheduleAwareSweepPolicy,
    StayPolicy,
    ThresholdPolicy,
    build_baselines,
)
from src.data import (
    CHANNELS,
    DEFAULT_SCAN_SPLITS,
    DataLeakageError,
    assert_no_scan_leakage,
    build_dataset,
    build_ood_dataset,
    discover_scan_files,
)
from src.env import AntiJammingEnv, make_train_eval_envs


def _write_synthetic_corpus(directory: Path) -> None:
    """Create all 8 x 10 files for each of the three physical conditions."""

    for distance_cm, power_dbm in ((20, 10), (40, 10), (20, 5)):
        condition_offset = (distance_cm - 20) * 0.01 + (10 - power_dbm) * 0.02
        for jammer_index, jammer_channel in enumerate(CHANNELS):
            for scan_id in range(10):
                rows = []
                for channel_index, channel in enumerate(CHANNELS):
                    for time_index in range(6):
                        rows.append(
                            {
                                "freq1": channel,
                                "snr": (
                                    10.0 * channel_index
                                    + time_index
                                    + 0.1 * jammer_index
                                    + 0.01 * scan_id
                                    + condition_offset
                                ),
                                "rssi": 50.0 + channel_index + time_index,
                                "noise": -100.0 + time_index,
                            }
                        )
                path = directory / (
                    f"samples_chamber_{jammer_channel}MHz_"
                    f"{distance_cm}cm_{power_dbm}dBm_{scan_id}.csv"
                )
                pd.DataFrame(rows).to_csv(path, index=False)


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.raw_dir = Path(cls._temp.name)
        _write_synthetic_corpus(cls.raw_dir)
        cls.dataset = build_dataset(
            cls.raw_dir,
            window_size=3,
            stride=2,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_condition_filter_selects_exact_80_file_matrix(self) -> None:
        default_files = discover_scan_files(self.raw_dir)
        distance_ood = discover_scan_files(
            self.raw_dir, distance_cm=40, power_dbm=10
        )
        power_ood = discover_scan_files(
            self.raw_dir, distance_cm=20, power_dbm=5
        )
        self.assertEqual(len(default_files), 80)
        self.assertEqual(len(distance_ood), 80)
        self.assertEqual(len(power_ood), 80)
        self.assertTrue(all("_20cm_10dBm_" in p.name for p in default_files.values()))
        self.assertTrue(all("_40cm_10dBm_" in p.name for p in distance_ood.values()))
        self.assertTrue(all("_20cm_5dBm_" in p.name for p in power_ood.values()))

        evaluation = build_ood_dataset(
            self.raw_dir,
            distance_cm=40,
            power_dbm=10,
            window_size=3,
            stride=2,
        )
        self.assertEqual(tuple(evaluation.keys()), ("ood",))
        self.assertEqual(evaluation["ood"].scan_ids, tuple(range(10)))

    def test_default_scan_split_happens_before_windowing(self) -> None:
        self.assertEqual(self.dataset.train.scan_ids, DEFAULT_SCAN_SPLITS["train"])
        self.assertEqual(self.dataset.val.scan_ids, DEFAULT_SCAN_SPLITS["val"])
        self.assertEqual(self.dataset.test.scan_ids, DEFAULT_SCAN_SPLITS["test"])
        self.assertEqual(self.dataset.train.observations.shape, (7 * 8 * 2, 40))
        self.assertTrue(
            all(
                np.array_equal(starts, np.array([0, 2]))
                for starts in self.dataset.train.window_starts.values()
            )
        )
        assert_no_scan_leakage(self.dataset)
        source_sets = [
            set(split.source_files.values()) for split in self.dataset.values()
        ]
        self.assertTrue(source_sets[0].isdisjoint(source_sets[1]))
        self.assertTrue(source_sets[0].isdisjoint(source_sets[2]))
        self.assertTrue(source_sets[1].isdisjoint(source_sets[2]))

    def test_feature_order_and_sample_standard_deviation(self) -> None:
        trajectory = self.dataset.train.trajectory(0, CHANNELS[0])
        first_channel = trajectory[0, :5]
        np.testing.assert_allclose(
            first_channel,
            np.array([1.0, 1.0, 51.0, 1.0, -99.0], dtype=np.float32),
        )
        second_channel = trajectory[0, 5:10]
        np.testing.assert_allclose(
            second_channel,
            np.array([11.0, 1.0, 52.0, 1.0, -99.0], dtype=np.float32),
        )

    def test_overlapping_scan_assignment_is_rejected(self) -> None:
        with self.assertRaises(DataLeakageError):
            build_dataset(
                self.raw_dir,
                split_scan_ids={"train": (0,), "test": (0,)},
                window_size=3,
            )


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.raw_dir = Path(cls._temp.name)
        _write_synthetic_corpus(cls.raw_dir)
        cls.dataset = build_dataset(cls.raw_dir, window_size=3, stride=1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_previous_action_is_observable_and_reset_clears_episode_fields(self) -> None:
        env = AntiJammingEnv(
            self.dataset.train,
            jammer_mode="sweeping",
            episode_length=3,
            switch_cost=0.1,
            seed=11,
            random_start=False,
        )
        observation, info = env.reset(
            options={"scan_id": 0, "start_index": 0, "jammer_channel": CHANNELS[0]}
        )
        self.assertEqual(observation.shape, (48,))
        np.testing.assert_array_equal(observation[-8:], np.zeros(8))
        self.assertEqual(info["jammer_channel"], CHANNELS[0])

        next_observation, reward, terminated, truncated, step_info = env.step(0)
        self.assertEqual(reward, -1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(step_info["collision"])
        self.assertFalse(step_info["switched"])
        self.assertEqual(step_info["acted_jammer_channel"], CHANNELS[0])
        self.assertEqual(step_info["jammer_channel"], CHANNELS[1])
        np.testing.assert_array_equal(next_observation[-8:], np.eye(8)[0])

        _, reward, _, _, step_info = env.step(2)
        self.assertAlmostEqual(reward, 0.9)
        self.assertTrue(step_info["switched"])
        self.assertEqual(env.switch_count, 1)

        # A collision remains the single dominant -1 case.
        _, reward, terminated, truncated, step_info = env.step(2)
        self.assertEqual(reward, -1.0)
        self.assertTrue(step_info["collision"])
        self.assertFalse(step_info["switched"])
        self.assertFalse(terminated)
        self.assertTrue(truncated)

        env.reset(
            options={"scan_id": 0, "start_index": 0, "jammer_channel": CHANNELS[0]}
        )
        env.step(2)  # safe initial action; sweeping jammer advances to action 1
        _, reward, _, _, step_info = env.step(1)
        self.assertTrue(step_info["collision"])
        self.assertTrue(step_info["switched"])
        self.assertEqual(reward, -1.0)  # no extra switching cost on collision

        reset_observation, reset_info = env.reset(
            options={"scan_id": 0, "start_index": 0, "jammer_channel": CHANNELS[0]}
        )
        self.assertIsNone(env.previous_action)
        self.assertIsNone(env.freq)
        self.assertEqual(env.step_count, 0)
        self.assertEqual(env.collision_count, 0)
        self.assertEqual(env.switch_count, 0)
        np.testing.assert_array_equal(reset_observation[-8:], np.zeros(8))
        self.assertFalse(reset_info["collision"])

    def test_random_jammer_is_reproducible_but_train_eval_seeds_differ(self) -> None:
        kwargs = dict(
            jammer_mode="random", episode_length=3, seed=123, random_start=False
        )
        env_a = AntiJammingEnv(self.dataset.train, **kwargs)
        env_b = AntiJammingEnv(self.dataset.train, **kwargs)
        obs_a, info_a = env_a.reset(options={"scan_id": 0, "start_index": 0})
        obs_b, info_b = env_b.reset(options={"scan_id": 0, "start_index": 0})
        np.testing.assert_array_equal(obs_a, obs_b)
        sequence_a = [info_a["jammer_channel"]]
        sequence_b = [info_b["jammer_channel"]]
        for _ in range(3):
            _, _, _, _, info_a = env_a.step(0)
            _, _, _, _, info_b = env_b.step(0)
            sequence_a.append(info_a["jammer_channel"])
            sequence_b.append(info_b["jammer_channel"])
        self.assertEqual(sequence_a, sequence_b)

        train_env, eval_env = make_train_eval_envs(
            self.dataset.train, self.dataset.test
        )
        self.assertNotEqual(train_env.seed_value, eval_env.seed_value)
        with self.assertRaises(ValueError):
            make_train_eval_envs(
                self.dataset.train,
                self.dataset.test,
                train_seed=5,
                eval_seed=5,
            )


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.raw_dir = Path(cls._temp.name)
        _write_synthetic_corpus(cls.raw_dir)
        cls.dataset = build_dataset(cls.raw_dir, window_size=3, stride=1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    @staticmethod
    def _observation(snr_means: np.ndarray, previous: int | None = None) -> np.ndarray:
        features = np.zeros((8, 5), dtype=np.float32)
        features[:, 0] = snr_means
        one_hot = np.zeros(8, dtype=np.float32)
        if previous is not None:
            one_hot[previous] = 1.0
        return np.concatenate((features.reshape(-1), one_hot))

    def test_non_oracles_do_not_read_jammer_truth(self) -> None:
        class PoisonInfo(dict):
            def __getitem__(self, key):  # pragma: no cover - fails if accessed
                raise AssertionError(f"non-oracle read info[{key!r}]")

            def __contains__(self, key):  # pragma: no cover - fails if accessed
                raise AssertionError(f"non-oracle inspected info[{key!r}]")

        observation = self._observation(np.arange(8, dtype=np.float32))
        poison = PoisonInfo(jammer_channel=CHANNELS[0])
        self.assertIn(RandomPolicy(seed=1).act(observation, poison), range(8))
        self.assertEqual(StayPolicy(initial_action=3).act(observation, poison), 3)
        self.assertEqual(MinInterferencePolicy().act(observation, poison), 0)

        threshold = ThresholdPolicy(threshold=-4.0)
        self.assertIn(threshold.act(observation, poison), range(8))

    def test_threshold_is_fitted_only_on_train_and_validation(self) -> None:
        policy = ThresholdPolicy.fit(self.dataset.train, self.dataset.val)
        self.assertTrue(np.isfinite(policy.threshold))
        self.assertEqual(policy.fitted_splits, ("train", "val"))
        with self.assertRaises(ValueError):
            ThresholdPolicy.fit(self.dataset.test)

        suite = build_baselines(self.dataset.train, self.dataset.val, seed=4)
        self.assertEqual(
            set(suite),
            {
                "random",
                "stay",
                "max_quality",
                "threshold",
                "schedule_sweep",
                "jammer_greedy",
                "clairvoyant_oracle",
            },
        )

    def test_jammer_greedy_reads_current_truth_but_is_not_named_oracle(self) -> None:
        observation = self._observation(np.arange(8, dtype=np.float32))
        policy = JammerAwareGreedyPolicy(CHANNELS)
        action = policy.act(observation, {"jammer_channel": CHANNELS[0]})
        self.assertNotEqual(action, 0)
        with self.assertRaises(ValueError):
            policy.act(observation, None)
        self.assertIs(OraclePolicy, JammerAwareGreedyPolicy)

    def test_schedule_aware_sweep_has_declared_return_and_no_privileged_reads(self) -> None:
        class Poison:
            def __getattribute__(self, name):  # pragma: no cover - fails on read
                raise AssertionError(f"schedule policy read {name}")

        policy = ScheduleAwareSweepPolicy(n_actions=8, phase_index=0)
        policy.reset(seed=123)
        previous = None
        total_return = 0.0
        collisions = 0
        switches = 0
        for step in range(100):
            action = policy.act(Poison(), Poison())
            collision = action == step % 8
            switched = previous is not None and action != previous
            collisions += int(collision)
            switches += int(switched)
            total_return += -1.0 if collision else 1.0 - 0.1 * int(switched)
            previous = action
        self.assertEqual(collisions, 0)
        self.assertEqual(switches, 14)
        self.assertAlmostEqual(total_return, 98.6)

    def test_clairvoyant_dp_matches_brute_force_and_is_an_upper_bound(self) -> None:
        from itertools import product

        channels = (0, 1, 2)
        jammer = (0, 1, 2, 1, 0)

        def score(actions):
            total = 0.0
            previous = None
            for action, jammed in zip(actions, jammer):
                collision = action == jammed
                switched = previous is not None and action != previous
                total += -1.0 if collision else 1.0 - 0.1 * int(switched)
                previous = action
            return total

        brute_force_scores = {
            actions: score(actions)
            for actions in product(range(len(channels)), repeat=len(jammer))
        }
        oracle = ClairvoyantOraclePolicy(channels, switch_cost=0.1)
        oracle.prepare_episode(jammer)
        oracle.reset(seed=99)
        planned = tuple(oracle.act(None, None) for _ in jammer)
        self.assertAlmostEqual(score(planned), max(brute_force_scores.values()))
        self.assertAlmostEqual(oracle.planned_return, max(brute_force_scores.values()))

        # Explicit candidate strategies cannot exceed the full-horizon DP plan.
        candidates = [
            (0,) * len(jammer),
            (1,) * len(jammer),
            (2,) * len(jammer),
            (2, 2, 1, 1, 1),
            (1, 0, 1, 0, 1),
        ]
        self.assertTrue(all(score(planned) >= score(candidate) for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
