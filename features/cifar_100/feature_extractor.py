import os
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

OUT_DIR = "features\\cifar_100"
BATCH   = 256
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Build ResNet50-feature extractor
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(DEVICE).eval()

    # 2) Data transforms
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    # 3) Load CIFAR-100 train & test
    train_ds = datasets.CIFAR100(root='.', train=True,  download=True, transform=tf)
    test_ds  = datasets.CIFAR100(root='.', train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,  num_workers=4)

    def extract(dl, split):
        feats, labels, indices = [], [], []
        with torch.no_grad():
            for i, (imgs, labs) in enumerate(dl):
                imgs = imgs.to(DEVICE)
                out  = model(imgs).cpu().numpy()      # (B, 2048)
                feats.append(out)
                labels.append(labs.numpy())
                start = i * BATCH
                idxs  = np.arange(start, start + imgs.size(0))
                indices.append(idxs)

        feats = np.concatenate(feats, axis=0)
        labs  = np.concatenate(labels, axis=0)
        idxs  = np.concatenate(indices, axis=0)
        np.savez_compressed(os.path.join(OUT_DIR, f"cifar100_{split}_feats.npz"),
                            X=feats, y=labs, idx=idxs)
        print(f"Saved {split} →", feats.shape, labs.shape)

    extract(train_loader, 'train')
    extract(test_loader,  'test')

if __name__ == '__main__':
    main()