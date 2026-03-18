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
