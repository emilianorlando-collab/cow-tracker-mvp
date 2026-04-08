#!/usr/bin/env python3
"""
01_entrenar_reid.py

Script autocontenido para entrenar un modelo de Re-Identificación (Re-ID) de ganado
usando PyTorch + Torchvision, partiendo desde un dataset local organizado por carpetas
(ImageFolder).

Salida final importante:
- Guarda SOLO el extractor de embeddings (256-D) en: src/mi_modelo_reid.pt
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


class ReIDResNet18(nn.Module):
    """
    Modelo completo para entrenamiento supervisado tipo clasificación.

    Estructura conceptual:
    1) BackBone ResNet18 (extrae representación visual de alto nivel de 512-D).
    2) Cuello de botella (512 -> 256): comprime la representación para obtener
       un embedding compacto y útil para comparación en Re-ID.
    3) BatchNorm1d(256): estabiliza la escala del embedding durante entrenamiento.
    4) Capa clasificadora final (256 -> num_classes): usada SOLO para entrenar con
       CrossEntropy y forzar que embeddings separen identidades.

    Nota matemática breve del cuello de botella:
    Si h \in R^512 es la salida del backbone, el embedding e \in R^256 se obtiene como:
        e = W*h + b
    donde W \in R^(256x512), b \in R^256.
    Esta proyección lineal aprende a conservar información discriminativa en menos
    dimensiones, lo que reduce coste de búsqueda y almacenamiento en FAISS.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ResNet18 preentrenada en ImageNet para aprovechar features visuales robustas.
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Guardamos cuántas features entran a la capa final original (512 en ResNet18).
        in_features = base_model.fc.in_features

        # Reemplazamos la 'fc' por el bloque pedido:
        #   a) Linear 512 -> 256  (embedding puro)
        #   b) BatchNorm1d(256)
        #   c) Linear 256 -> num_classes
        base_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward para clasificación durante entrenamiento."""
        return self.model(x)


class ReIDFeatureExtractor(nn.Module):
    """
    Extractor puro de embeddings 256-D.

    Este módulo elimina la última capa clasificadora y conserva:
    - BackBone ResNet18 hasta pooling global
    - Proyección lineal 512 -> 256

    Su salida es el vector de características que luego puedes indexar en FAISS.
    """

    def __init__(self, trained_model: ReIDResNet18):
        super().__init__()

        # Copiamos todas las capas de ResNet18 excepto la 'fc'.
        self.backbone = nn.Sequential(*list(trained_model.model.children())[:-1])

        # Del bloque secuencial [Linear(512->256), BN, Linear(256->num_classes)],
        # tomamos SOLO la primera capa lineal para producir embedding puro.
        self.embedding = trained_model.model.fc[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Features espaciales -> pooling global (N, 512, 1, 1)
        x = self.backbone(x)

        # 2) Aplanar a (N, 512)
        x = torch.flatten(x, 1)

        # 3) Proyectar a embedding de 256-D
        x = self.embedding(x)

        return x


def crear_dataloader(data_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, int]:
    """
    Crea DataLoader de entrenamiento usando ImageFolder con augmentations pedidas.

    Transformaciones:
    - Resize(224,224): entrada estándar para ResNet.
    - RandomHorizontalFlip: robustez a orientación.
    - ColorJitter: robustez a cambios de iluminación/cámara.
    - ToTensor + Normalización ImageNet: alinea distribución esperada por backbone.
    """
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

    if len(dataset.classes) < 2:
        raise ValueError(
            "Se detectaron menos de 2 clases en el dataset. "
            "Verifica que 'datos/entrenamiento/' tenga al menos dos carpetas de vacas."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, len(dataset.classes)


def entrenar(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> None:
    """
    Bucle de entrenamiento principal.

    Métricas por época:
    - Loss promedio por muestra.
    - Accuracy de entrenamiento.
    """
    model.train()

    total_samples = len(dataloader.dataset)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # Acumuladores para métricas.
            batch_size_actual = images.size(0)
            running_loss += loss.item() * batch_size_actual

            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        print(
            f"Época [{epoch + 1}/{num_epochs}] - "
            f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento Re-ID de ganado con ResNet18")
    parser.add_argument("--data_dir", type=str, default="datos/entrenamiento/", help="Ruta del dataset")
    parser.add_argument("--num_epochs", type=int, default=15, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate para Adam")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers de DataLoader")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"No existe la ruta de dataset: {args.data_dir}. "
            "Asegúrate de tener carpetas por vaca dentro de esa ruta."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")

    dataloader, num_classes = crear_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Clases detectadas: {num_classes}")
    print(f"Imágenes detectadas: {len(dataloader.dataset)}")

    model = ReIDResNet18(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    entrenar(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
    )

    # Al final, aislamos el extractor de embeddings (sin capa clasificadora final).
    feature_extractor = ReIDFeatureExtractor(model).to(device)

    os.makedirs("src", exist_ok=True)
    output_path = "src/mi_modelo_reid.pt"

    # Guardamos SOLO los pesos del extractor 256-D.
    torch.save(feature_extractor.state_dict(), output_path)
    print(f"Extractor de embeddings guardado en: {output_path}")


if __name__ == "__main__":
    main()