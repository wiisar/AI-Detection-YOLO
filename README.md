# Projet YOLOv3 et YOLO26

Ce projet utilise le **YOLOv3 original avec backbone Darknet-53**, via
l'implementation PyTorch officielle d'Ultralytics. Le code amont est place dans
`vendor/yolov3` et les poids COCO pre-entraines dans `weights/yolov3.pt`.
Le meme script peut aussi utiliser les modeles Ultralytics recents comme
YOLO26n. Les poids officiels manquants sont telecharges dans `weights/` lors
de leur premiere utilisation.

> Un modele recent peut lui aussi apprendre une nouvelle classe. Le choix de
> YOLOv3 est donc surtout utile pour respecter le sujet du projet ou comparer
> une ancienne architecture. Pour un petit dataset, partir des poids COCO est
> generalement bien meilleur qu'un entrainement complet depuis zero.

## 1. Installation

Sur cette machine, utilisez Python 3.12 (le `python3` par defaut est une version
3.15 trop recente pour les roues PyTorch courantes) :

```bash
python3.12 -m venv .venv-yolov3
source .venv-yolov3/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python yolov3.py check
```

## 2. Preparer les donnees

Le dataset doit avoir le format YOLO suivant :

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/       # facultatif
└── labels/
    ├── train/
    ├── val/
    └── test/       # facultatif
```

Pour chaque `images/train/photo.jpg`, il faut un
`labels/train/photo.txt`. Chaque ligne de label vaut :

```text
class_id x_centre y_centre largeur hauteur
```

Les quatre coordonnees sont normalisees entre 0 et 1. Modifiez ensuite
`data/dataset.yaml` pour nommer vos classes. Exemple :

```yaml
names:
  0: roi_blanc
  1: roi_noir
  2: pion_blanc
```

Attention : le dataset Hugging Face `jrahn/yolochess_deepblue` actuellement
lu par `dataset_inspection.py` contient 511 positions FEN et des coups, mais
aucune image ni boite englobante. Malgre son nom, ce n'est pas un dataset de
detection YOLO et il ne peut pas etre passe directement a cette commande.

## 3. Choisir le modele dans la commande

L'option `--model` permet de changer de modele sans changer de script :

```bash
# YOLOv3 classique
python yolov3.py predict --model yolov3 --source 0 --classes 0 --device mps --view

# YOLOv3-Tiny, plus rapide mais moins precis
python yolov3.py predict --model yolov3-tiny --source 0 --classes 0 --device mps --view

# YOLO26 Nano, recommande pour la webcam sur le Mac M3
python yolov3.py predict --model yolo26n --source 0 --classes 0 --device mps --view
```

Les autres tailles YOLO26 fonctionnent de la meme maniere : `yolo26s`,
`yolo26m`, `yolo26l` et `yolo26x`. Plus le modele est grand, plus il est
generalement precis et lent.

Un checkpoint moderne personnalise peut aussi etre selectionne directement :

```bash
python yolov3.py predict --model runs/train/custom-yolo/weights/best.pt --source 0 --device mps --view
```

Pour un ancien checkpoint YOLOv3 classique, conservez l'option `--weights`.

### Dataset Open Images `Man` / `Woman`

Les CSV Open Images bruts sont ranges dans :

```text
dataset/openimages/annotations/
├── train-annotations-bbox.csv
├── validation-annotations-bbox.csv
└── test-annotations-bbox.csv
```

Le script `prepare_openimages_gender.py` filtre `Man` et `Woman`, convertit les
coordonnees Open Images au format YOLO et telecharge les JPEG selectionnes :

```bash
source .venv-yolov3/bin/activate
python prepare_openimages_gender.py --download --workers 16
```

Par defaut, il produit un sous-ensemble reproductible de 2 000 images
d'entrainement, 400 de validation et 400 de test dans `dataset_gender/`.
Les fichiers deja telecharges sont conserves et ignores lors d'une reprise.
La configuration correspondante est `data/gender.yaml`.

## 4. Entrainer sur de nouvelles classes

Fine-tuning recommande avec les poids COCO :

```bash
python yolov3.py train \
  --model yolov3 \
  --data data/dataset.yaml \
  --epochs 100 \
  --batch-size 8 \
  --device cpu
```

Sur Linux avec un GPU NVIDIA, utilisez `--device 0`. Sur un Mac Apple Silicon,
vous pouvez utiliser `--device mps` seulement si `python yolov3.py check`
affiche `MPS disponible : True`. Sinon, gardez `--device cpu` et reduisez
eventuellement `--batch-size` et `--imgsz`.

Pour entrainer toute l'architecture avec des poids aleatoires :

```bash
python yolov3.py train --data data/dataset.yaml --scratch --device cpu
```

Avec YOLO26n :

```bash
python yolov3.py train \
  --model yolo26n \
  --data data/gender.yaml \
  --epochs 50 \
  --batch-size 8 \
  --device mps \
  --name yolo26-gender
```

Profil plus rapide recommande sur un Mac M3 :

```bash
python yolov3.py train \
  --model yolo26n \
  --data data/gender.yaml \
  --epochs 30 \
  --batch-size 16 \
  --imgsz 416 \
  --cache ram \
  --workers 4 \
  --device mps \
  --name yolo26-gender-fast
```

Ce profil met les images en cache dans la memoire unifiee et traite deux fois
plus d'images par lot. La resolution 416 reduit le calcul par rapport a 640,
avec un compromis raisonnable pour la detection de personnes. Si macOS signale
un manque de memoire, reduisez seulement `--batch-size` de 16 a 8.

Le meilleur modele est normalement ecrit dans :

```text
runs/train/custom-yolo/weights/best.pt
```

## 5. Tester le modele

Sur une image :

```bash
python yolov3.py predict \
  --weights runs/train/custom-yolov3/weights/best.pt \
  --source chemin/vers/image.jpg \
  --device cpu
```

Sur la webcam, remplacez la source par `0`. Les images annotees sont ecrites
dans `runs/predict/`.

Evaluation sur le jeu de validation :

```bash
python yolov3.py val \
  --weights runs/train/custom-yolov3/weights/best.pt \
  --data data/dataset.yaml \
  --device cpu
```

## Fichiers importants

- `yolov3.py` : script Python a utiliser.
- `data/dataset.yaml` : chemins et noms des classes.
- `data/gender.yaml` : dataset Open Images `Man` / `Woman` converti en YOLO.
- `prepare_openimages_gender.py` : conversion et telechargement du dataset.
- `weights/yolov3.pt` : poids YOLOv3 pre-entraines sur COCO.
- `weights/yolo26n.pt` : poids YOLO26 Nano pre-entraines sur COCO.
- `vendor/yolov3` : implementation amont, revision
  `34c3f2108b73031647b293a7e73603d90e5a208b`.

Le code Ultralytics inclus est distribue sous licence AGPL-3.0.
