import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from datasets import load_dataset

OUT_DIR = "features\\core50"

def extract_features():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load CORe50 dataset from HuggingFace
    ds = load_dataset("adrake17/core50")

    # Model: ResNet18 (drop final FC layer → 512-d features)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Containers
    X_train, y_train, ids_train = [], [], []
    X_test,  y_test,  ids_test  = [], [], []

    # Iterate over split (train/test available in CORe50 HF dataset)
    with torch.no_grad():
        for split in ["train", "test"]:
            dataset = ds[split]
            total = len(dataset)
            print(f"\n[INFO] Extracting {split} features ({total} samples)...")

            for idx, sample in enumerate(dataset):
                img = sample["image"].convert("RGB")
                label = sample["label"]   # 0..49 (50 objects)

                inp = tf(img).unsqueeze(0).to(device)       # 1×3×224×224
                feat = model(inp).cpu().squeeze(0).numpy()  # 512-d

                if split == "train":
                    X_train.append(feat)
                    y_train.append(label)
                    ids_train.append(idx)
                else:
                    X_test.append(feat)
                    y_test.append(label)
                    ids_test.append(idx)
                
                if (idx + 1) % 1000 == 0 or (idx + 1) == total:
                    print(f"[{split.upper()}] Processed {idx+1}/{total} samples")

    # Save compressed NPZ files
    np.savez_compressed(os.path.join(OUT_DIR, "core50_train_feats.npz"),
                        X=np.stack(X_train), y=np.array(y_train), ids=np.array(ids_train))
    np.savez_compressed(os.path.join(OUT_DIR, "core50_test_feats.npz"),
                        X=np.stack(X_test),  y=np.array(y_test),  ids=np.array(ids_test))

    print("Feature extraction complete.")
    print("Train:", len(X_train), "Test:", len(X_test))

if __name__ == "__main__":
    extract_features()