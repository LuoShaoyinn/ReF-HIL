import argparse

HOST = "0.0.0.0"
REQUEST_PORT = 7002
RESPONSE_HOST = "127.0.0.1"
RESPONSE_PORT = 7003


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SpaceMouse operator server")
    parser.add_argument(
        "--override-rotation",
        action="store_true",
        help="replace SpaceMouse rotation with the task copilot's upward command",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Keep the hardware-only dependency out of CLI parsing and documentation
    # tooling. The real driver imports it only when the process starts.
    from actor.operator.spacemouse import SpacemouseServer

    server = SpacemouseServer(
        host=HOST,
        request_port=REQUEST_PORT,
        response_host=RESPONSE_HOST,
        response_port=RESPONSE_PORT,
        override_rotation=args.override_rotation,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
