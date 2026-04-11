#!/usr/bin/env python3
"""
Script 01: Entrenamiento Re-ID de ganado.
Lógica de División:
- Si hay >1 subcarpetas: Toma la ÚLTIMA subcarpeta completa para Test, el resto para Train.
- Si hay 1 sola subcarpeta: Hace un split interno 80/20 dentro de esa misma carpeta.
"""

import os
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


class ReIDModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)               
        x = torch.flatten(x, 1)            
        z = self.embedding(x)              
        e = F.normalize(z, p=2, dim=1)     
        return e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        logits = self.classifier(emb)
        return logits


class ReIDFeatureExtractor(nn.Module):
    def __init__(self, trained_model: ReIDModel):
        super().__init__()
        self.backbone = trained_model.backbone
        self.embedding = trained_model.embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class SafeImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        
        print("⏳ Escaneando carpetas de forma segura (búsqueda profunda)...")
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"No se encontró la ruta: {root_dir}")
            
        carpetas = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes = sorted(carpetas)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            try:
                for carpeta_actual, subcarpetas, archivos in os.walk(class_path):
                    subcarpeta_name = os.path.relpath(carpeta_actual, class_path)
                    
                    for file_name in archivos:
                        if not file_name.startswith('.'):
                            full_path = os.path.join(carpeta_actual, file_name)
                            if '.' not in file_name or file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                self.samples.append((full_path, class_idx, subcarpeta_name))
                                
            except OSError as e:
                print(f"🚨 ALERTA DE HARDWARE: Error al leer sector de la clase {class_name}. Saltando. Detalle: {e}")
                continue 
                    
        print(f"✅ Escaneo exitoso: {len(self.samples)} imágenes válidas detectadas.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx, subfolder = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224)) 

        if self.transform:
            image = self.transform(image)

        return image, class_idx


def dividir_train_test_cross_pose(dataset: Dataset):
    """
    Reglas de división:
    1. Si hay >1 subcarpeta: La última subcarpeta va 100% a Test. El resto 100% a Train.
    2. Si hay 1 subcarpeta: Divide internamente 80% Train y 20% Test.
    """
    indices_train = []
    indices_test = []

    agrupado = defaultdict(lambda: defaultdict(list))

    for idx, (path, class_idx, subfolder) in enumerate(dataset.samples):
        agrupado[class_idx][subfolder].append(idx)

    print("\n📊 Resumen de División (Lógica: Última carpeta a Test / Fallback 80-20):")
    print("-" * 80)
    
    for class_idx in sorted(agrupado.keys()):
        nombre_clase = dataset.classes[class_idx]
        subfolders = sorted(agrupado[class_idx].keys())
        num_subfolders = len(subfolders)

        if num_subfolders == 1:
            # Plan B: Solo 1 subcarpeta. Split tradicional interno 80/20.
            sf = subfolders[0]
            fotos = agrupado[class_idx][sf]
            n_total = len(fotos)
            n_train = int(n_total * 0.8)

            train_idx = fotos[:n_train]
            test_idx = fotos[n_train:]

            indices_train.extend(train_idx)
            indices_test.extend(test_idx)

            print(f"🐄 Clase '{nombre_clase:2}' (1 subcarpeta) -> Split interno | Train: {len(train_idx)} | Test: {len(test_idx)}")

        else:
            # Plan A: Varias subcarpetas. Tomar la última entera para Test.
            subfolders_train = subfolders[:-1]
            subfolder_test = subfolders[-1]

            c_train, c_test = 0, 0
            
            for sf in subfolders_train:
                indices_train.extend(agrupado[class_idx][sf])
                c_train += len(agrupado[class_idx][sf])

            indices_test.extend(agrupado[class_idx][subfolder_test])
            c_test += len(agrupado[class_idx][subfolder_test])

            print(f"🐄 Clase '{nombre_clase:2}' ({num_subfolders} subcarpetas) -> Train ({len(subfolders_train)} subc.): {c_train} fotos | Test (Carpeta '{subfolder_test}'): {c_test} fotos")

    print("-" * 80)
    return Subset(dataset, indices_train), Subset(dataset, indices_test)


def entrenar_modelo(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    os.makedirs("models", exist_ok=True)

    print("\n🚀 Iniciando entrenamiento (Validando solo con datos Train en consola)...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0

        for images, labels in train_loader:
            try:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                running_corrects += torch.sum(preds == labels).item()
                processed_samples += batch_size
                
            except Exception as e:
                continue

        if processed_samples > 0:
            epoch_loss = running_loss / processed_samples
            epoch_acc = running_corrects / processed_samples
            print(f"Época [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_acc:.4f}")
            
            torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch+1}.pt")
        else:
            print(f"Época [{epoch + 1}/{num_epochs}] falló por completo.")


def main():
    device = torch.device("cpu")
    print(f"Dispositivo forzado: {device}")

    data_dir = "datos/entrenamiento/fotos/"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = SafeImageFolder(root_dir=data_dir, transform=train_transform)

    if len(full_dataset.samples) == 0:
        print("\n❌ CRÍTICO: No se encontraron imágenes válidas.")
        return
        
    if len(full_dataset.classes) < 2:
        print("\n❌ CRÍTICO: Se detectaron menos de 2 clases.")
        return

    # APLICAMOS EL NUEVO SPLIT
    train_subset, test_subset = dividir_train_test_cross_pose(full_dataset)

    print(f"\n✅ Total Imágenes: {len(full_dataset)}")
    print(f"✅ Subset Train: {len(train_subset)}")
    print(f"✅ Subset Test (Prueba): {len(test_subset)}\n")

    train_loader = DataLoader(
        train_subset,
        batch_size=16, 
        shuffle=True,
        num_workers=0, 
    )

    num_classes = len(full_dataset.classes)
    model = ReIDModel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    entrenar_modelo(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
    )

    # Guardado del extractor final
    output_path = "models/mi_modelo_reid.pt"
    extractor = ReIDFeatureExtractor(model).to(device)
    torch.save(extractor.state_dict(), output_path)
    print(f"🎉 Extractor Re-ID (Libre de Data Leakage) guardado en: {output_path}")


if __name__ == "__main__":
    main()