### Проект по Computer Vision.

В рамках данного проекта было реализовано универсальное приложение на основе платформы Streamlit, предназначенное для обработки и анализа изображений.

Эти возможности позволяют пользователям выполнять разнообразные операции с изображениями, повышая эффективность и точность анализа.

Приложение включает в себя две функции, такие как:
1. детекция опухолей головного мозга в аксиальной проекции YOLOv5 о.

   - модель: YOLOv5 бучалась с целью научиться обнаруживать и классифицировать соответствующие объекты на изображениях. Обучение включало в себя      процесс оптимизации параметров модели с использованием градиентного спуска и функции потерь

   - данные для обучения: датасет [Object Detection Dataset](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets)

2. очищение документов от шумов с помощью автоэнкодера

    - модель: сверточная нейронная сеть, была построена архитектура автоэнкодера, состоящая из энкодера, декодера
  
    - данные для обучения: датасет [Denoising Dirty Documents](https://www.kaggle.com/c/denoising-dirty-documents/data)
  
Эти возможности позволяют пользователям выполнять разнообразные операции с изображениями, повышая эффективность и точность анализа.

**Результат:** 
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cvproject-rxwz8g0iw6m.streamlit.app/brain_tumor_%F0%9F%A7%A0)

### Computer Vision Project.

This project involves the implementation of a versatile application using the Streamlit platform, designed for image processing and analysis.

These functionalities enable users to perform various image operations, enhancing the efficiency and accuracy of analysis.

The application includes two main functions:

1. Brain Tumor Detection in Axial Projections
   - Model: YOLOv5
   - Training Data: [Object Detection Dataset](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets)

2. Document Denoising using an Autoencoder
   - Model: Convolutional Neural Network
   - Training Data: [Denoising Dirty Documents](https://www.kaggle.com/c/denoising-dirty-documents/data)

These capabilities empower users to perform diverse image operations, improving the effectiveness and precision of analysis.

Results: Streamlit. [Faster Image Master](https://cvproject-rxwz8g0iw6m.streamlit.app/brain_tumor_%F0%9F%A7%A0)
