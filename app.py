import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Charger un modèle pré-entraîné pour la détection d’objets
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

# Liste des classes COCO
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Charger et prétraiter l’image
img_path = "téléchargé (2).jpeg"
img = Image.open(img_path).convert("RGB")  # <-- conversion ici






img = Image.open(img_path)
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

# Prédiction
with torch.no_grad():
    predictions = model(img_tensor)

# Afficher les objets détectés avec un score de confiance > 0.7
for idx, score in enumerate(predictions[0]['scores']):
    if score > 0.7:
        label = COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'][idx]]
        print(f"Objet détecté : {label} (confiance : {score:.2f})")
