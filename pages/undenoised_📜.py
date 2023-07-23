import streamlit as st
from torchvision.transforms import ToTensor, Grayscale, ToPILImage
import torch
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.utils import save_image
# import torchutils as tu
from PIL import Image
# import torchutils


# Загрузка изображения и вывод результата через определенную нейросеть на странице "Undenoised"
st.set_page_config(
    page_title='Очистка текста',
    page_icon=":scroll:",
    layout='wide')


def undenoised_page(img):
    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()

            # Encoder
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4),
                nn.BatchNorm2d(32),
                nn.SELU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 8, kernel_size=2),
                nn.BatchNorm2d(8),
                nn.SELU()
            )

            self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True)

            self.unpool = nn.MaxUnpool2d(2, 2)

            self.conv1_t = nn.Sequential(
                nn.ConvTranspose2d(8, 32, kernel_size=2),
                nn.BatchNorm2d(32),
                nn.SELU()
            )
            self.conv2_t = nn.Sequential(
                nn.ConvTranspose2d(32, 1, kernel_size=4),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        def encode(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x, indicies = self.pool(x)
            return x, indicies

        def decode(self, x, indicies):
            x = self.unpool(x, indicies)
            x = self.conv1_t(x)
            x = self.conv2_t(x)
            return x

        def forward(self, x):
            latent, indicies = self.encode(x)
            out = self.decode(latent, indicies)
            return out

    model = ConvAutoencoder()
    model.load_state_dict(torch.load(
        'Weights/ADAMax_denoiz_weights.pth'))
    model.eval()

    img_tensor = T.ToTensor()(T.Grayscale()(img)).unsqueeze(0).float()
    denoised_img = model(img_tensor)
    # res_img = denoised_img.squeeze().detach().numpy()
    # print(denoised_img.shape)
    return T.Grayscale()(T.ToPILImage()(denoised_img.squeeze(0)))


info = ('''
        Очистка загрязненных документов!

        Теперь вы можете не переживать если вдруг пролили свой любимый кофе на важные документы!

        Мы это с легкостью исправим
        ''')

if st.button("Как можно использовать эту нейросеть?", type='secondary', key='info_button'):
    st.divider()
    st.markdown(
        info, unsafe_allow_html=True)
    st.divider()


uploaded_file = st.file_uploader(
    '# Загрузи сюда картинку c текстом, которую нужно отчистить:', type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption='Загруженное изображение',
               use_column_width=True)

    if st.button('Очистить', use_container_width=True, type='primary'):
        denoised_image = undenoised_page(image)
        col2.image(denoised_image, caption='Очищенное изображение',
                   use_column_width=True)
