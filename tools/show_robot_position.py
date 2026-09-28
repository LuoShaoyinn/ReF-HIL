#!/usr/bin/env python3
"""Print the UR TCP position and rotation without commanding the robot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def format_pose(pose: np.ndarray) -> str:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    xyz_mm = pose[:3] * 1000.0
    rotvec = pose[3:]
    return (
        f"xyz_mm=[{xyz_mm[0]:.3f}, {xyz_mm[1]:.3f}, {xyz_mm[2]:.3f}] "
        f"rotvec_rad=[{rotvec[0]:.6f}, {rotvec[1]:.6f}, {rotvec[2]:.6f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-host", default="127.0.0.1")
    parser.add_argument("--request-port", type=int, default=7001)
    parser.add_argument("--response-host", default="0.0.0.0")
    parser.add_argument("--response-port", type=int, default=7008)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--watch", action="store_true", help="continue printing until Ctrl-C")
    parser.add_argument("--interval", type=float, default=0.2, help="watch interval in seconds")
    args = parser.parse_args()

    from shared.zmq import Receiver, Sender, ZmqEndpointConfig

    request = Sender(ZmqEndpointConfig(host=args.request_host, port=args.request_port))
    response = Receiver(
        ZmqEndpointConfig(host=args.response_host, port=args.response_port),
        timeout_ms=args.timeout_ms,
    )
    try:
        while True:
            request.send("read_tcp_pose")
            pose = response.recv(None)
            if pose is None:
                raise RuntimeError("timed out waiting for the robot driver's TCP pose")
            print(format_pose(np.asarray(pose)), flush=True)
            if not args.watch:
                break
            time.sleep(max(0.01, float(args.interval)))
    except KeyboardInterrupt:
        pass
    finally:
        request.close()
        response.close()


if __name__ == "__main__":
    main()
