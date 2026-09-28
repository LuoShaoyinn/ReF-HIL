#!/usr/bin/env python3
"""Execute the assemble task's dismantle and episode-reset procedure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actor.transport import ZmqRobotClient
from shared.zmq import ZmqEndpointConfig
from tasks.assemble.reset import ResetConfig, run_reset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=7001)
    parser.add_argument("--robot-response-host", default="0.0.0.0")
    parser.add_argument("--robot-response-port", type=int, default=7008)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    import numpy as np

    robot = ZmqRobotClient(
        request_endpoint=ZmqEndpointConfig(host=args.robot_host, port=args.robot_port),
        response_endpoint=ZmqEndpointConfig(
            host=args.robot_response_host,
            port=args.robot_response_port,
        ),
        timeout_ms=args.timeout_ms,
    )
    try:
        final_position = run_reset(
            robot.send_action,
            robot.read_observation,
            config=ResetConfig(),
            rng=np.random.default_rng(args.seed),
        )
        print(f"assemble reset complete: {(final_position * 1000.0).round(1).tolist()} mm")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
