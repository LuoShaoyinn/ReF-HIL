from __future__ import annotations

import unittest

from actor.operator.run import build_parser


class OperatorCliTest(unittest.TestCase):
    def test_rotation_override_is_opt_in(self) -> None:
        parser = build_parser()

        self.assertFalse(parser.parse_args([]).override_rotation)
        self.assertTrue(
            parser.parse_args(["--override-rotation"]).override_rotation
        )


if __name__ == "__main__":
    unittest.main()
