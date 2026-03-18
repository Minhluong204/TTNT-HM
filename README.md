# Hướng dẫn chạy code 
## B1:Lấy dữ liệu dataset
### Truy cập vào https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification 
### Download file archive.zip
### Giải nén file này ta sẽ được file Garbage classification, nén file Garbage classification thành dạng file zip 
## B2: Huấn luyện AI 
### Vào google colab https://colab.research.google.com/
### Tạo 1 sổ tay mới --> chạy code
```
from google.colab import files
files.upload()
```
### sẽ ra được như này 

<img width="444" height="92" alt="image" src="https://github.com/user-attachments/assets/acf2e9ba-f4d1-41d2-b603-5f8f3be757c3" />

### sau đó ấn chọn tệp --> chọn file Garbage classification.zip ta vừa nén ở bước 1

### đợi sau khi tệp đã tải lên hoàn tất ta chạy code
```
!unzip -o "Garbage classification.zip"
```
### để colab mở đọc dữ liệu trong file
### tiếp theo ta sẽ Import các thư viện cần thiết
```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```
### Chuẩn bị dữ liệu và chia tập huấn luyện/kiểm tra
```
data_dir = "Garbage classification"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

--- Train/Validation/Test split ---
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
seed = 42

assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9

Chia theo lớp (stratified) để các tập có phân bố lớp tương tự nhau
from collections import defaultdict
import random
from torch.utils.data import Subset

targets = getattr(dataset, "targets", None)
if targets is None:
    targets = [label for _, label in dataset.samples]

indices_by_class = defaultdict(list)
for idx, label in enumerate(targets):
    indices_by_class[int(label)].append(idx)

rng = random.Random(seed)
train_indices, val_indices, test_indices = [], [], []
for label, idxs in indices_by_class.items():
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_indices.extend(idxs[:n_train])
    val_indices.extend(idxs[n_train : n_train + n_val])
    test_indices.extend(idxs[n_train + n_val :])

rng.shuffle(train_indices)
rng.shuffle(val_indices)
rng.shuffle(test_indices)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Classes:", dataset.classes)
print(f"Split sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```
### Tải và cấu hình mô hình ResNet18
```
# Tải ResNet18 pretrained (ImageNet) - tương thích nhiều phiên bản torchvision
try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception:
    model = models.resnet18(pretrained=True)

Số lớp đầu ra (mặc định 6 lớp rác)
num_classes = 6
try:
    num_classes = len(dataset.classes)
except Exception:
    pass

model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
### Hàm đánh giá mô hình (`evaluate`)
```
def evaluate(model, loader, criterion, device):
    """Tính loss và accuracy cho validation/test."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with ctx():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc
```
### Huấn luyện mô hình
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

import copy
best_val_acc = -1.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / max(1, train_total)
    train_acc = train_correct / max(1, train_total)

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}% | "
        f"Val loss: {val_loss:.4f} | Val acc: {val_acc*100:.2f}%"
    )

Load lại mô hình tốt nhất theo Validation trước khi đánh giá Test / lưu model
model.load_state_dict(best_model_wts)
print(f"Best Val acc: {best_val_acc*100:.2f}%")

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc*100:.2f}%")
```

### Đây là kết quả huấn luyện 
<img width="667" height="188" alt="image" src="https://github.com/user-attachments/assets/10aa815d-4d3b-4a96-82d2-bae07ac587ff" />

### Từ bảng kết quả trên có thể nhận thấy rằng mô hình đạt độ chính xác khá cao trên tập huấn luyện (93.60%), đồng thời duy trì được hiệu năng tốt trên tập validation (87.80%). Khi đánh giá trên tập test, độ chính xác đạt 82.55%, cho thấy mô hình có khả năng tổng quát hóa tương đối tốt trên dữ liệu chưa từng thấy.

### Lưu và tải mô hình đã huấn luyện
```
torch.save(model.state_dict(), "garbage_model.pth")
from google.colab import files
files.download("garbage_model.pth")
```
### sau khi chạy xong ta sẽ tải được file garbage_model.pth
### copy nó vào vs code để chạy bước tiếp theo
## B3: Chạy steamlit trên vs code
### Trong vs code hãy tạo file app.py và model.py
#### app.py
```
from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import load_model


CLASSES_VI: list[str] = [
    "Bìa carton",
    "Thủy tinh",
    "Kim loại",
    "Giấy",
    "Nhựa",
    "Rác khác",
]
IMAGE_SIZE: tuple[int, int] = (224, 224)
MODEL_PATH = Path(__file__).with_name("garbage_model.pth")


def _cache_resource(func: Callable[..., object]) -> Callable[..., object]:
    cache_resource = getattr(st, "cache_resource", None)
    if cache_resource is not None:
        try:
            return cache_resource(show_spinner=False)(func)
        except TypeError:
            return cache_resource(func)

    experimental_singleton = getattr(st, "experimental_singleton", None)
    if experimental_singleton is not None:
        return experimental_singleton(func)

    return func


@_cache_resource
def get_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình tại: {MODEL_PATH}")
    return load_model(str(MODEL_PATH))


@_cache_resource
def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def predict_image(
    image: Image.Image,
) -> tuple[int, float, list[dict[str, float]]]:
    model = get_model()
    transform = get_transform()

    input_tensor = transform(image).unsqueeze(0)
    has_inference_mode = hasattr(torch, "inference_mode")
    inference_ctx = (
        torch.inference_mode
        if has_inference_mode
        else torch.no_grad
    )

    with inference_ctx():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    predicted_idx = int(torch.argmax(probs).item())
    confidence_pct = float(probs[predicted_idx].item() * 100.0)

    rows: list[dict[str, float]] = []
    for idx, class_vi in enumerate(CLASSES_VI):
        rows.append(
            {
                "Loại": class_vi,
                "Xác suất (%)": round(float(probs[idx].item() * 100.0), 2),
            }
        )
    rows.sort(key=lambda r: r["Xác suất (%)"], reverse=True)

    return predicted_idx, confidence_pct, rows


st.set_page_config(
    page_title="Phân loại rác",
    layout="wide",
)

st.title("Ứng dụng phân loại rác")
st.caption(
    "Tải lên một ảnh để hệ thống dự đoán loại rác và hiển thị bảng xác suất."
)

with st.sidebar:
    st.header("Tải ảnh")
    uploaded_file = st.file_uploader(
        "Chọn ảnh (JPG/PNG/JPEG)",
        type=["jpg", "jpeg", "png"],
    )
    st.markdown("---")
    st.header("Hướng dẫn")
    st.markdown(
        """
- Chọn ảnh định dạng JPG/PNG/JPEG
- Ảnh càng rõ vật thể càng tốt
- Kết quả hiển thị kèm độ tin cậy (%)
""".strip()
    )


if uploaded_file is None:
    st.info("Hãy tải lên một ảnh ở thanh bên (sidebar) để bắt đầu.")
    st.stop()


try:
    image = Image.open(uploaded_file).convert("RGB")
except UnidentifiedImageError:
    st.error("Không đọc được ảnh. Vui lòng thử lại với file JPG/PNG hợp lệ.")
    st.stop()


left, right = st.columns([1, 1])

with left:
    st.subheader("Ảnh")
    st.image(image, caption="Ảnh đã tải lên", width="stretch")

with right:
    st.subheader("Kết quả")

    try:
        with st.spinner("Đang phân tích ảnh..."):
            predicted_idx, confidence_pct, rows = predict_image(image)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        st.error(
            "Có lỗi khi chạy mô hình. Vui lòng thử lại "
            "hoặc kiểm tra môi trường cài đặt."
        )
        st.stop()

    predicted_vi = CLASSES_VI[predicted_idx]
    st.success(f"Dự đoán: {predicted_vi}")
    st.metric("Độ tin cậy", f"{confidence_pct:.2f}%")

    st.subheader("Bảng so sánh xác suất")
    st.dataframe(rows, hide_index=True, width="stretch")
```
 #### model.py
 ```
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 6


def load_model(model_path):
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()

    return model
```
### mở terminal sau đó chạy 
```
cd garbage_app
..\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
### đợi vs code tải thư viện sau đó sẽ ra được giao diện như sau





