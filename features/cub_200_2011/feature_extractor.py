import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

CUB_ROOT = "datasets\\cub_200_2011\\CUB_200_2011"
CUB_IMG_ROOT = "datasets\\cub_200_2011\\CUB_200_2011\\images"
OUT_DIR  = "features\\cub_200_2011"

def read_list(path):
    """Read two-column <id> <value> file, return dict[id]=value."""
    d = {}
    with open(path, 'r') as f:
        for line in f:
            idx, val = line.strip().split()
            d[int(idx)] = val
    return d

def read_split(path):
    """Read train_test_split.txt → dict[id]=0/1."""
    d = {}
    with open(path, 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            d[int(idx)] = int(is_train)
    return d

def read_labels(path):
    """Read image_class_labels.txt → dict[id]=class_id."""
    d = {}
    with open(path, 'r') as f:
        for line in f:
            idx, cls = line.strip().split()
            d[int(idx)] = int(cls) - 1  # zero-based
    return d

def extract_features():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read metadata
    images_txt = os.path.join(CUB_ROOT, 'images.txt')
    split_txt  = os.path.join(CUB_ROOT, 'train_test_split.txt')
    label_txt  = os.path.join(CUB_ROOT, 'image_class_labels.txt')

    id2file = read_list(images_txt)       # id → "images/…jpg"
    id2split= read_split(split_txt)       # id → 0/1
    id2label= read_labels(label_txt)      # id → 0..199

    # Build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    X_train, y_train, ids_train = [], [], []
    X_test,  y_test,  ids_test  = [], [], []

    with torch.no_grad():
        for img_id, rel_path in id2file.items():
            fullpath = os.path.join(CUB_IMG_ROOT, rel_path)
            img = Image.open(fullpath).convert('RGB')
            inp = tf(img).unsqueeze(0).to(device)      # 1×3×224×224
            feat = model(inp).cpu().squeeze(0).numpy() # 2048-d

            if id2split[img_id] == 1:
                X_train.append(feat)
                y_train.append(id2label[img_id])
                ids_train.append(img_id)
            else:
                X_test.append(feat)
                y_test.append(id2label[img_id])
                ids_test.append(img_id)

    # Save to .npz
    np.savez_compressed(os.path.join(OUT_DIR, 'cub200_train_feats.npz'),
                        X=np.stack(X_train), y=np.array(y_train), ids=np.array(ids_train))
    np.savez_compressed(os.path.join(OUT_DIR, 'cub200_test_feats.npz'),
                        X=np.stack(X_test),  y=np.array(y_test),  ids=np.array(ids_test))

if __name__ == '__main__':
    extract_features()
    print("Feature extraction complete.")