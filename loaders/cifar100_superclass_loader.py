import os
import pickle
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict

class CIFAR100SuperclassPythonLoader:
    """
    Loads CIFAR-100 (Python version) and splits into 20 tasks,
    one per coarse (superclass) label.  Extracts ResNet18 features.
    """
    def __init__(self, data_path, batch_size=128, device='cpu'):
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device

        # Pretrained ResNet18 (drop final FC)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        self.feat_extractor = model.to(self.device).eval()

        self.train_task_data, self.test_task_data = self._prepare_tasks()

    def _unpickle(self, fname):
        with open(fname,'rb') as f:
            return pickle.load(f, encoding='bytes')
        

    def _prepare_tasks(self):
        def reshape(arr):
            return arr.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        def process_split(images, fine_labels, target_dict, super_class_name):
            label_to_task = {label: label // 5 for label in range(100)}  # 20 tasks

            total = len(images)
            print(f"\nExtracting ResNet18 features for {super_class_name}...")

            with torch.no_grad():
                for i in range(0, total, self.batch_size):
                    batch_imgs = images[i:i + self.batch_size]
                    batch_labels = fine_labels[i:i + self.batch_size]

                    tensor_batch = torch.stack([
                        self.transform(Image.fromarray(img)) for img in batch_imgs
                    ]).to(self.device)

                    features = self.feat_extractor(tensor_batch).cpu().numpy()

                    for feat, label in zip(features, batch_labels):
                        task_id = label_to_task[label]
                        target_dict[task_id].append((feat, label))

                    processed = min(i + self.batch_size, total)
                    percent = (processed / total) * 100
                    print(f"\rProgress: [{int(percent):>3}%] {processed}/{total} images processed", end='')

        # --- Load and reshape train data ---
        print("Loading CIFAR-100 Superclass train data...")
        train_dict = self._unpickle(os.path.join(self.data_path, 'train'))
        train_imgs = reshape(train_dict[b'data'])
        train_fine = train_dict[b'fine_labels']
        self.train_task_data = defaultdict(list)
        process_split(train_imgs, train_fine, self.train_task_data, 'train')

        # --- Load and reshape test data ---
        print("Loading CIFAR-100 Superclass test data...")
        test_dict = self._unpickle(os.path.join(self.data_path, 'test'))
        test_imgs = reshape(test_dict[b'data'])
        test_fine = test_dict[b'fine_labels']
        self.test_task_data = defaultdict(list)
        process_split(test_imgs,  test_fine,  self.test_task_data,  'test')

        print("\n\nFeature extraction complete.")

        for task_id in self.train_task_data:
            feats, labels = zip(*self.train_task_data[task_id])
            self.train_task_data[task_id] = (np.stack(feats), np.array(labels))

        for task_id in self.test_task_data:
            feats, labels = zip(*self.test_task_data[task_id])
            self.test_task_data[task_id] = (np.stack(feats), np.array(labels))

        return self.train_task_data, self.test_task_data
 
    def stream(self):
        """
        Yields (task_id, X_train) for each coarse label.
        """
        for task_id in range(20):
            X_tr, _ = self.train_task_data[task_id]
            yield task_id, np.stack(X_tr)