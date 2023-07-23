from matplotlib import pyplot as plt
import torch
import torchvision

from PIL import Image
import streamlit as st
from torchvision import transforms as T

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def brain_tumor_page(img):
    weights = 'Weights/best.pt'

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=weights, force_reload=True)
    model.conf = 0.1

    model.eval()

    results = model(img)
    return results


info = ("Загрузив сюда изображение головного мозга в аксиальной проекции, можно увидеть вероятность опухоли!")

if st.button("Как можно использовать эту нейросеть?", type='secondary'):
    st.divider()
    st.markdown(
        f"<p style='font-size: 20px'><b>{info}</b></p>", unsafe_allow_html=True)
    st.divider()


uploaded_file = st.file_uploader(
    '# Загрузите изображение головного мозга в аксиальной проекции', type=["jpg", "jpeg", "png"])
col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption='Загруженное изображение', use_column_width=True)

    if st.button('Проверить', use_container_width=True, type='primary'):
        detected_image = brain_tumor_page(image)
        col2.image(detected_image.render(), caption='Результат работы идентификатора',
                   use_column_width=True)
