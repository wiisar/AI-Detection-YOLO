"""Interface commune pour YOLOv3 classique et les modeles Ultralytics recents."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOV3_ROOT = PROJECT_ROOT / "vendor" / "yolov3"
WEIGHTS_ROOT = PROJECT_ROOT / "weights"
DEFAULT_DATA = PROJECT_ROOT / "data" / "dataset.yaml"
CLASSIC_MODELS = {
    "yolov3": "yolov3.pt",
    "yolov3-spp": "yolov3-spp.pt",
    "yolov3-tiny": "yolov3-tiny.pt",
}


def project_path(value: str | Path) -> Path:
    """Return an absolute path, interpreting relative paths from the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_file(value: str | Path, description: str) -> Path:
    path = project_path(value)
    if not path.is_file():
        raise SystemExit(f"{description} introuvable : {path}")
    return path


def model_key(value: str) -> str:
    """Normalize an official model name such as yolov3.pt to yolov3."""
    return Path(value).stem.lower()


def select_model(args: argparse.Namespace) -> tuple[str, str]:
    """Return the backend name and the model reference selected by the user."""
    if args.weights:
        return "classic", str(require_file(args.weights, "Poids YOLOv3"))

    if model_key(args.model) in CLASSIC_MODELS:
        filename = CLASSIC_MODELS[model_key(args.model)]
        return "classic", str(WEIGHTS_ROOT / filename)

    if "://" in args.model:
        return "ultralytics", args.model

    requested = Path(args.model).expanduser()
    if requested.is_absolute() or requested.parent != Path("."):
        return "ultralytics", str(project_path(requested))

    filename = requested.name if requested.suffix else f"{requested.name}.pt"
    return "ultralytics", str(WEIGHTS_ROOT / filename)


def source_value(value: str) -> str | int:
    """Convert a webcam index to int and resolve local paths from the project root."""
    if value.isdigit():
        return int(value)
    if "://" in value or value == "screen":
        return value
    return str(project_path(value))


def run_yolov3_script(script: str, arguments: list[str]) -> None:
    """Execute one of the classic YOLOv3 scripts with the current Python."""
    script_path = require_file(YOLOV3_ROOT / script, "Script YOLOv3")
    command = [sys.executable, str(script_path), *arguments]
    print("Commande :", shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ultralytics_model(reference: str):
    """Load a recent Ultralytics model, downloading official weights if needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Installez les dependances avec `python -m pip install -e .`.") from exc
    return YOLO(reference)


def train(args: argparse.Namespace) -> None:
    backend, reference = select_model(args)
    data = str(require_file(args.data, "Configuration du dataset"))

    if backend == "classic":
        command = [
            "--data",
            data,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
            "--project",
            str(project_path(args.project)),
            "--name",
            args.name,
        ]
        if args.scratch:
            command += ["--weights", "", "--cfg", str(YOLOV3_ROOT / "models" / "yolov3.yaml")]
        else:
            command += ["--weights", reference]
        if args.freeze:
            command += ["--freeze", str(args.freeze)]
        if args.cache:
            command += ["--cache", args.cache]
        run_yolov3_script("train.py", command)
        return

    if args.scratch:
        name = Path(args.model).name
        if name.endswith(".pt"):
            name = f"{name[:-3]}.yaml"
        elif not name.endswith(".yaml"):
            name = f"{name}.yaml"
        reference = name

    options = {
        "data": data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device or None,
        "workers": args.workers,
        "project": str(project_path(args.project)),
        "name": args.name,
        "freeze": args.freeze or None,
        "cache": args.cache or False,
    }
    ultralytics_model(reference).train(**options)


def predict(args: argparse.Namespace) -> None:
    backend, reference = select_model(args)
    source = source_value(args.source)

    if backend == "classic":
        command = [
            "--weights",
            reference,
            "--source",
            str(source),
            "--imgsz",
            str(args.imgsz),
            "--conf-thres",
            str(args.conf),
            "--device",
            args.device,
            "--vid-stride",
            str(args.vid_stride),
            "--project",
            str(project_path(args.project)),
            "--name",
            args.name,
        ]
        if args.classes:
            command += ["--classes", *(str(value) for value in args.classes)]
        if args.view:
            command.append("--view-img")
        if args.save_txt:
            command.append("--save-txt")
        if args.half:
            command.append("--half")
        run_yolov3_script("detect.py", command)
        return

    options = {
        "source": source,
        "stream": True,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device or None,
        "classes": args.classes,
        "vid_stride": args.vid_stride,
        "show": args.view,
        "save": True,
        "save_txt": args.save_txt,
        "project": str(project_path(args.project)),
        "name": args.name,
    }
    if args.half:
        options["half"] = True
    results = ultralytics_model(reference).predict(**options)
    for _ in results:
        pass


def validate(args: argparse.Namespace) -> None:
    backend, reference = select_model(args)
    data = str(require_file(args.data, "Configuration du dataset"))

    if backend == "classic":
        command = [
            "--weights",
            reference,
            "--data",
            data,
            "--imgsz",
            str(args.imgsz),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--project",
            str(project_path(args.project)),
            "--name",
            args.name,
            "--verbose",
        ]
        run_yolov3_script("val.py", command)
        return

    ultralytics_model(reference).val(
        data=data,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device or None,
        project=str(project_path(args.project)),
        name=args.name,
    )


def check_installation(_: argparse.Namespace) -> None:
    require_file(YOLOV3_ROOT / "train.py", "Depot YOLOv3")
    try:
        import torch
        import torchvision
        import ultralytics
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Dependances absentes. Activez .venv-yolov3 puis lancez "
            "`python -m pip install -e .`."
        ) from exc

    print(f"YOLOv3 classique : {YOLOV3_ROOT}")
    print(f"Ultralytics      : {ultralytics.__version__}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"Torchvision      : {torchvision.__version__}")
    print(f"YAML             : {yaml.__version__}")
    print(f"MPS disponible   : {torch.backends.mps.is_available()}")
    for path in sorted(WEIGHTS_ROOT.glob("*.pt")):
        print(f"Modele local     : {path.name} ({path.stat().st_size / 1024**2:.1f} Mio)")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="yolov3",
        help="Modele a utiliser : yolov3, yolov3-tiny, yolo26n, chemin/best.pt, etc.",
    )
    parser.add_argument(
        "--weights",
        help="Ancienne option reservee a un checkpoint YOLOv3 classique.",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--imgsz", type=int, default=640, help="Taille carree des images.")
    parser.add_argument("--device", default="", help="cpu, mps, 0, 0,1, etc.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrainer et utiliser YOLOv3 classique ou un modele Ultralytics recent."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrainer le modele selectionne.")
    add_model_arguments(train_parser)
    train_parser.add_argument("--data", default=str(DEFAULT_DATA))
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--project", default="runs/train")
    train_parser.add_argument("--name", default="custom-yolo")
    train_parser.add_argument(
        "--freeze", type=int, default=0, help="Nombre de premieres couches a geler."
    )
    train_parser.add_argument(
        "--cache", choices=("ram", "disk"), help="Mettre les images en cache."
    )
    train_parser.add_argument(
        "--scratch",
        action="store_true",
        help="Partir de poids aleatoires au lieu d'un modele pre-entraine.",
    )
    add_runtime_arguments(train_parser)
    train_parser.set_defaults(handler=train)

    predict_parser = subparsers.add_parser(
        "predict", help="Detecter des objets dans une image, video, dossier ou webcam."
    )
    add_model_arguments(predict_parser)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--classes", nargs="+", type=int)
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--vid-stride", type=int, default=1)
    predict_parser.add_argument("--project", default="runs/predict")
    predict_parser.add_argument("--name", default="resultat")
    predict_parser.add_argument("--view", action="store_true")
    predict_parser.add_argument("--save-txt", action="store_true")
    predict_parser.add_argument("--half", action="store_true")
    add_runtime_arguments(predict_parser)
    predict_parser.set_defaults(handler=predict)

    val_parser = subparsers.add_parser("val", help="Mesurer precision, rappel et mAP.")
    add_model_arguments(val_parser)
    val_parser.add_argument("--data", default=str(DEFAULT_DATA))
    val_parser.add_argument("--batch-size", type=int, default=8)
    val_parser.add_argument("--project", default="runs/val")
    val_parser.add_argument("--name", default="evaluation")
    add_runtime_arguments(val_parser)
    val_parser.set_defaults(handler=validate)

    check_parser = subparsers.add_parser("check", help="Verifier l'installation locale.")
    check_parser.set_defaults(handler=check_installation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
