"""Prepare a small Open Images Man/Woman subset for Ultralytics YOLO.

The Open Images CSV files contain normalized corner coordinates. This script
filters the two requested classes, writes YOLO labels, creates image-ID lists,
and can download the corresponding JPEG files from the public Open Images S3
bucket.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "dataset" / "openimages" / "annotations"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset_gender"
DEFAULT_LISTS = PROJECT_ROOT / "dataset" / "openimages" / "image_lists"

# Open Images boxable class identifiers.
TARGET_CLASSES = {
    "/m/04yx4": 0,  # Man
    "/m/03bt1vf": 1,  # Woman
}

# Images containing these more general/younger person categories are excluded
# so those people are not accidentally treated as unlabelled background.
OTHER_PERSON_CLASSES = {
    "/m/01g317",  # Person
    "/m/01bl7v",  # Boy
    "/m/05r655",  # Girl
}

SPLIT_NAMES = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def yolo_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def parse_box(row: dict[str, str]) -> Box | None:
    """Convert one Open Images box row to normalized YOLO coordinates."""
    try:
        x_min = float(row["XMin"])
        x_max = float(row["XMax"])
        y_min = float(row["YMin"])
        y_max = float(row["YMax"])
        class_id = TARGET_CLASSES[row["LabelName"]]
    except (KeyError, TypeError, ValueError):
        return None

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None

    return Box(
        class_id=class_id,
        x_center=(x_min + x_max) / 2,
        y_center=(y_min + y_max) / 2,
        width=width,
        height=height,
    )


def is_true(row: dict[str, str], field: str) -> bool:
    return row.get(field, "0") == "1"


def read_annotations(csv_path: Path) -> dict[str, list[Box]]:
    """Read target boxes and reject images with incompatible person labels."""
    boxes_by_image: dict[str, list[Box]] = defaultdict(list)
    excluded_images: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "ImageID",
            "LabelName",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Colonnes absentes dans {csv_path.name}: {', '.join(sorted(missing))}"
            )

        for row in reader:
            image_id = row["ImageID"]
            label = row["LabelName"]

            if label in OTHER_PERSON_CLASSES:
                excluded_images.add(image_id)
                continue

            if label not in TARGET_CLASSES:
                continue
            if (
                is_true(row, "IsGroupOf")
                or is_true(row, "IsDepiction")
                or is_true(row, "IsInside")
            ):
                continue

            box = parse_box(row)
            if box is not None:
                boxes_by_image[image_id].append(box)

    for image_id in excluded_images:
        boxes_by_image.pop(image_id, None)
    return dict(boxes_by_image)


def balanced_sample(
    boxes_by_image: dict[str, list[Box]], maximum: int, seed: int
) -> list[str]:
    """Select a deterministic subset with similar representation per class."""
    image_ids = sorted(boxes_by_image)
    if maximum <= 0 or len(image_ids) <= maximum:
        return image_ids

    rng = random.Random(seed)
    by_class = {
        class_id: [
            image_id
            for image_id in image_ids
            if any(box.class_id == class_id for box in boxes_by_image[image_id])
        ]
        for class_id in TARGET_CLASSES.values()
    }
    for pool in by_class.values():
        rng.shuffle(pool)

    selected: set[str] = set()
    target_per_class = maximum // len(by_class)
    for class_id, pool in sorted(by_class.items()):
        represented = 0
        for image_id in pool:
            if image_id not in selected:
                selected.add(image_id)
            represented = sum(
                any(box.class_id == class_id for box in boxes_by_image[item])
                for item in selected
            )
            if represented >= target_per_class or len(selected) >= maximum:
                break

    remaining = [image_id for image_id in image_ids if image_id not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: maximum - len(selected)])
    return sorted(selected)


def write_split(
    source_split: str,
    output_split: str,
    boxes_by_image: dict[str, list[Box]],
    selected_ids: list[str],
    output_root: Path,
    lists_root: Path,
) -> None:
    labels_dir = output_root / "labels" / output_split
    images_dir = output_root / "images" / output_split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    lists_root.mkdir(parents=True, exist_ok=True)

    for image_id in selected_ids:
        label_path = labels_dir / f"{image_id}.txt"
        contents = "\n".join(box.yolo_line() for box in boxes_by_image[image_id])
        label_path.write_text(f"{contents}\n", encoding="utf-8")

    list_path = lists_root / f"{source_split}.txt"
    list_contents = "\n".join(f"{source_split}/{image_id}" for image_id in selected_ids)
    list_path.write_text(f"{list_contents}\n", encoding="utf-8")


def download_one(source_split: str, image_id: str, destination: Path) -> str:
    if destination.is_file() and destination.stat().st_size > 0:
        return "cached"

    url = (
        "https://open-images-dataset.s3.amazonaws.com/"
        f"{source_split}/{image_id}.jpg"
    )
    part_path = destination.with_suffix(".jpg.part")
    request = urllib.request.Request(url, headers={"User-Agent": "yolo-school-project/1.0"})

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with part_path.open("wb") as output_file:
                    while chunk := response.read(1024 * 1024):
                        output_file.write(chunk)
            os.replace(part_path, destination)
            return "downloaded"
        except (OSError, urllib.error.URLError) as exc:
            part_path.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Echec pour {source_split}/{image_id}: {exc}") from exc
            time.sleep(attempt)
    raise AssertionError("unreachable")


def download_images(
    source_split: str,
    output_split: str,
    selected_ids: list[str],
    output_root: Path,
    workers: int,
) -> None:
    images_dir = output_root / "images" / output_split
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    cached = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one,
                source_split,
                image_id,
                images_dir / f"{image_id}.jpg",
            ): image_id
            for image_id in selected_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                status = future.result()
                if status == "cached":
                    cached += 1
                else:
                    downloaded += 1
            except RuntimeError as exc:
                errors.append(str(exc))
            if index % 50 == 0 or index == len(futures):
                print(
                    f"  Images {output_split}: {index}/{len(futures)} "
                    f"(telechargees={downloaded}, deja presentes={cached}, erreurs={len(errors)})",
                    flush=True,
                )

    if errors:
        for error in errors[:10]:
            print(f"  {error}", file=sys.stderr)
        raise RuntimeError(f"{len(errors)} image(s) n'ont pas pu etre telechargees")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lists", type=Path, default=DEFAULT_LISTS)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_NAMES),
        default=list(SPLIT_NAMES),
    )
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-validation", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    maximums = {
        "train": args.max_train,
        "validation": args.max_validation,
        "test": args.max_test,
    }
    prepared = 0

    for source_split in args.splits:
        csv_path = args.annotations / f"{source_split}-annotations-bbox.csv"
        if not csv_path.is_file():
            print(f"Ignore (CSV absent): {csv_path}")
            continue

        output_split = SPLIT_NAMES[source_split]
        print(f"Lecture de {csv_path} ...", flush=True)
        boxes_by_image = read_annotations(csv_path)
        selected_ids = balanced_sample(
            boxes_by_image, maximum=maximums[source_split], seed=args.seed
        )
        write_split(
            source_split,
            output_split,
            boxes_by_image,
            selected_ids,
            args.output,
            args.lists,
        )
        class_counts = {
            class_id: sum(
                any(box.class_id == class_id for box in boxes_by_image[image_id])
                for image_id in selected_ids
            )
            for class_id in TARGET_CLASSES.values()
        }
        print(
            f"Prepare {output_split}: {len(selected_ids)} images, "
            f"classe 0 dans {class_counts[0]} images, "
            f"classe 1 dans {class_counts[1]} images"
        )
        if args.download:
            download_images(
                source_split,
                output_split,
                selected_ids,
                args.output,
                workers=max(1, args.workers),
            )
        prepared += 1

    if prepared == 0:
        raise SystemExit(
            "Aucun CSV trouve. Placez les fichiers *-annotations-bbox.csv dans "
            f"{args.annotations}"
        )


if __name__ == "__main__":
    main()
