import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# 1. VERİ ARTIRMA VE DÖNÜŞÜM BORU HATLARI

def get_transforms():
    """
    Eğitim için veri artırma (augmentation),
    Doğrulama ve test için standart boyutlandırma ve normalizasyon döndürür.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_test_transform

# 2. ÖZEL PYTORCH DATASET SINIFI

class CornDataset(Dataset):
    """
    Klasör hiyerarşisine göre etiketlenmiş tohum görsellerini yükleyen veri seti sınıfı.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []

        if os.path.exists(root_dir):
            self.classes = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label


# 3. DATALOADER OLUŞTURUCU FONKSİYONLAR

def build_dataloaders(root_dir, batch_size=32, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Veri setini Train (%70), Validation (%15) ve Test (%15) olarak böler, DataLoader üretir.
    """
    train_tf, val_tf = get_transforms()
    full_dataset = CornDataset(root_dir, transform=train_tf)

    if len(full_dataset) == 0:
        return None, None, None


    # Toplam 1050 görseli otomatik hesaplar
    total_len = len(full_dataset)                   # 1050
    train_len = int(total_len * train_ratio)        # 735
    val_len = int(total_len * val_ratio)            # 157
    test_len = total_len - train_len - val_len      # 158

    # Verileri rastgele ama her çalıştırmada aynı çıkacak şekilde (seed=42) 3'e böler:
    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader