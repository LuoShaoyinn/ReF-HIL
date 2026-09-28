from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from actor.episode_snapshots import (
    episode_actor_named,
    episode_actors_from,
    list_episode_actors,
    next_episode_actor_index,
    write_episode_actor,
)


class EpisodeActorSnapshotTest(unittest.TestCase):
    def test_snapshot_round_trip_and_ordering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-episode-actors-") as tmp:
            directory = Path(tmp)
            for episode in (2, 1):
                write_episode_actor(
                    directory=directory,
                    episode=episode,
                    policy_revision=episode + 10,
                    actor_snapshot={
                        "format": "test",
                        "runtime": {
                            "train_steps": episode * 100,
                            "actor_steps": episode * 10,
                            "actor_episodes": episode - 1,
                            "elapsed_seconds": episode * 5,
                        },
                    },
                )
            paths = list_episode_actors(directory)
            self.assertEqual(
                [path.name for path in paths],
                ["actor_episode_000001.pkl", "actor_episode_000002.pkl"],
            )
            self.assertEqual(episode_actors_from(paths, 2), paths[1:])
            self.assertEqual(next_episode_actor_index(directory), 3)
            self.assertEqual(episode_actor_named(directory, "1"), paths[0])
            self.assertEqual(
                episode_actor_named(directory, "actor_episode_000002.pkl"),
                paths[1],
            )
            with paths[0].open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["actor_episode"], 1)
            self.assertEqual(payload["policy_revision"], 11)
            self.assertEqual(payload["learner"]["train_steps"], 100)

    def test_start_episode_must_exist(self) -> None:
        with self.assertRaises(FileNotFoundError):
            episode_actors_from([Path("actor_episode_000001.pkl")], 2)
        with tempfile.TemporaryDirectory(prefix="srt-episode-actors-") as tmp:
            with self.assertRaises(FileNotFoundError):
                episode_actor_named(Path(tmp), "7")


if __name__ == "__main__":
    unittest.main()
