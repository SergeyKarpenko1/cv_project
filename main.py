import streamlit as st

import matplotlib.pyplot as plt
import torch
import torchvision


st.set_page_config(
    page_title='Проект по Computer Vision',
    layout='wide'
)
# st.sidebar.header("Home page")
c1, c2 = st.columns(2)
c2.image('Unknown-11')
c1.markdown("""
# Проект по Computer Vision
Cостоит из 2 частей:
### 1.Детекция опухолей головного мозга с помощью **YOLO v5**
### 2.Очищение документов от шумов с помощью **Автоэнкодера**
""")


# def main():
#     pass


# if __name__ == "__main__":
#     main()
