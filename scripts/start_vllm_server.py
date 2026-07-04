import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_config, start_judge_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_4gpu.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    server = start_judge_server(config["judge_model"])

    def shutdown(*_):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Server running. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(60)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
