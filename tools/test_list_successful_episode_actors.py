from __future__ import annotations

import unittest

from tools.list_successful_episode_actors import format_elapsed


class SuccessfulEpisodeActorsTest(unittest.TestCase):
    def test_formats_elapsed_time(self) -> None:
        self.assertEqual(format_elapsed(3661), "01:01:01")
        self.assertEqual(format_elapsed(-1), "unknown")


if __name__ == "__main__":
    unittest.main()
