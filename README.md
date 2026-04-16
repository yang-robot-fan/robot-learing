# robot-learing
import os
import time
import warnings
from typing import List, Tuple
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings("ignore")


class AudioDataset(Dataset):
    """
    将 ./data/cla1 和 ./data/cla2 下的音频转换为 log-mel 频谱图 (1, n_mels, time_steps)。
    """

    def __init__(
        self,
        data_path: str = "./data",
        sample_rate: int = 22050,
        duration: float = 3.0,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_frames: int = 128,
    ):
        super().__init__()
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames

        self.class_folders = ["cla1", "cla2"]
        self.audio_paths: List[str] = []
        self.labels: List[int] = []
        self.allowed_ext = [".wav", ".mp3", ".flac", ".m4a"]

        self._collect_files()

    def _collect_files(self) -> None:
        for class_idx, class_name in enumerate(self.class_folders):
            class_dir = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 文件夹 {class_dir} 不存在，已跳过。")
                continue
            files = [
                f
                for f in os.listdir(class_dir)
                if any(f.lower().endswith(ext) for ext in self.allowed_ext)
            ]
            for fname in files:
                self.audio_paths.append(os.path.join(class_dir, fname))
                self.labels.append(class_idx)

        if len(self.audio_paths) == 0:
            raise ValueError("未找到任何音频文件，请检查 ./data/cla1 和 ./data/cla2。")

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        mel = self._load_mel(audio_path)
        return mel, label

    def _load_mel(self, path: str) -> torch.Tensor:
        y, sr = librosa.load(path, sr=self.sample_rate, duration=self.duration)
        if len(y) == 0:
            y = np.zeros(int(self.sample_rate * self.duration), dtype=np.float32)

        # 若过短则重复填充
        target_len = int(self.sample_rate * self.duration)
        if len(y) < target_len:
            repeat = int(np.ceil(target_len / len(y)))
            y = np.tile(y, repeat)[:target_len]
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 统一时间维度
        if mel_db.shape[1] < self.max_frames:
            pad_width = self.max_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel_db.min())
        else:
            mel_db = mel_db[:, : self.max_frames]

        # 简单归一化到 0-1
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)

        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, max_frames)
        return mel_tensor


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_cbam: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.cbam = CBAM(out_channels) if use_cbam else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cbam:
            out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AudioCBAMResNetBiLSTM(nn.Module):
    def __init__(self, num_classes: int = 2, use_cbam: bool = True):
        super().__init__()
        self.use_cbam = use_cbam
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride, use_cbam=self.use_cbam)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, use_cbam=self.use_cbam))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)  # (batch, channels=512, time_steps=1)
        x, _ = self.lstm(x.permute(0, 2, 1))  # (batch, time_steps=1, features=512)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def save_model(model, class_names: List[str], model_path: str = "cnn_model") -> None:
    os.makedirs(model_path, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "class_names": class_names},
        os.path.join(model_path, "model_state.pt"),
    )
    with open(os.path.join(model_path, "class_names.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"模型已保存到: {model_path}")


def load_model(model_path: str = "cnn_model", use_cbam: bool = True, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(model_path, "model_state.pt"), map_location=device)
    class_names = checkpoint.get("class_names", ["cla1", "cla2"])
    num_classes = len(class_names)
    model = AudioCBAMResNetBiLSTM(num_classes=num_classes, use_cbam=use_cbam)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print(f"已从 {model_path} 加载模型")
    return model, class_names


def main():
    data_path = "./data"
    class_names = ["cla1", "cla2"]

    # Step 1: 加载数据
    print("[1/4] 加载音频数据并转换为 log-mel 频谱...")
    dataset = AudioDataset(data_path=data_path)
    labels = np.array(dataset.labels)
    print(f"总样本数: {len(dataset)}")
    for idx, name in enumerate(class_names):
        print(f"{name}: {(labels == idx).sum()} 样本")

    # 划分训练/测试
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0)

    # Step 2: 训练模型
    print("[2/4] 训练 CBAM ResNet + BiLSTM 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCBAMResNetBiLSTM(num_classes=len(class_names), use_cbam=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred = evaluate(model, test_loader, device)
        acc = accuracy_score(y_true, y_pred)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {train_loss:.4f} - "
            f"val_acc: {acc:.4f} - "
            f"time: {time.time() - start:.1f}s"
        )

    # Step 3: 评估并打印报告
    print("[3/4] 测试集评估...")
    y_true, y_pred = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    print("=== 测试集分类报告 ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"测试集准确率: {acc:.4f}")

    # Step 4: 保存模型
    print("[4/4] 保存模型...")
    save_model(model, class_names, model_path="cnn_model")

    # 演示加载与单样本预测
    loaded_model, loaded_class_names = load_model("cnn_model", use_cbam=True, device=device)
    sample_tensor, true_label = dataset[test_idx[0]]
    with torch.no_grad():
        logits = loaded_model(sample_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    print("\n加载模型测试预测：")
    print(f"预测类别: {loaded_class_names[pred_idx]} (置信度: {probs[pred_idx]:.3f})")
    print(f"真实类别: {loaded_class_names[true_label]}")


if __name__ == "__main__":
    main()
