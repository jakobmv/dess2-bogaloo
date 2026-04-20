from __future__ import annotations

import argparse
from pathlib import Path

from dess2_bogaloo.data import DatasetPaths, download_required_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the official SQID and ESCI files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = DatasetPaths(args.data_dir)
    downloaded = download_required_data(paths, overwrite=args.overwrite)
    if downloaded:
        for path in downloaded:
            print(path)
    else:
        print("All required files already exist.")


if __name__ == "__main__":
    main()
