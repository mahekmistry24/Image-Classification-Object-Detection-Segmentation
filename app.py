import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import torch
import zipfile
import tempfile
import os

# Class labels for Avengers classification
CLASS_LABELS = ["black widow", "captain america", "doctor strange", "ironman", "hulk", "loki", "spider-man", "thanos"]
MODEL_NAMES = ["custom_cnn_model", "inception_model", "vgg16_model", "resnet_model", "xception_model", "mobilenet_model"]

# Load the Flood Segmentation Model
@st.cache_resource
def load_segmentation_model():
    model_path = "flood_segmentation_model.h5"
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape[1:3]  # (height, width)
    return model, input_shape

# Load Avengers Classification Models
@st.cache_resource
def load_classification_model(model_name):
    model_path = f"./{model_name}.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

# Load Furniture Object Detection Model (YOLOv5)
@st.cache_resource
def load_object_detection_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Image preprocessing function
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = image.convert("RGB")
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)
    return image

# Predict with Classification Model
def classify_image(image, model, class_labels):
    processed_image = preprocess_image(image, (224, 224))
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

# Predict with Flood Segmentation Model
def segment_image(image, model, input_shape):
    processed_image = preprocess_image(image, target_size=input_shape)
    prediction = model.predict(processed_image)[0]  # Remove batch dimension
    return np.squeeze(prediction, axis=-1) if prediction.shape[-1] == 1 else prediction

# Detect objects with YOLOv5
def detect_objects(image, model):
    results = model(image)
    return results

# Perform Benchmarking
def benchmark_models(image_paths, true_labels):
    results = []
    for model_name in MODEL_NAMES:
        model = load_classification_model(model_name)
        y_pred = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).resize((224, 224)).convert("RGB")
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array, verbose=0)
                y_pred.append(np.argmax(pred))
            except Exception as e:
                st.error(f"Error processing image {img_path}: {e}")
                y_pred.append(-1)

        valid_indices = [i for i, y in enumerate(y_pred) if y != -1]
        y_true_valid = [true_labels[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]

        if y_true_valid:
            acc = accuracy_score(y_true_valid, y_pred_valid)
            prec = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=1)
            rec = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=1)
            f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=1)
            results.append([model_name, acc, prec, rec, f1])

    return results

# Streamlit App
def main():
    st.title("Multi-Function Image Analysis App")
    st.sidebar.title("Choose a Function")
    app_mode = st.sidebar.selectbox("Select Mode:", ["Avengers Classification", "Flood Segmentation", "Furniture Detection", "Benchmark Models"])

    if app_mode == "Avengers Classification":
        st.header("Avengers Image Classification")
        models = {model_name: load_classification_model(model_name) for model_name in MODEL_NAMES}

        uploaded_image = st.file_uploader("Upload an Image for Classification", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            for model_name, model in models.items():
                predicted_class = classify_image(image, model, CLASS_LABELS)
                st.subheader(f"{model_name} Prediction: {predicted_class}")

    elif app_mode == "Flood Segmentation":
        st.header("Flood Image Segmentation")
        model, input_shape = load_segmentation_model()

        uploaded_image = st.file_uploader("Upload an Image for Segmentation", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            prediction = segment_image(image, model, input_shape)
            st.subheader("Predicted Segmentation Mask:")
            st.image(prediction, caption="Segmentation Output", clamp=True, use_column_width=True)

    elif app_mode == "Furniture Detection":
        st.header("Furniture Object Detection")
        model = load_object_detection_model()

        uploaded_image = st.file_uploader("Upload an Image for Object Detection", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            results = detect_objects(np.array(image), model)
            st.subheader("Detection Results:")
            results.print()  # Display results in text
            results.save()  # Save results locally if needed
            st.image(results.imgs[0], caption="Detected Objects", use_column_width=True)

    elif app_mode == "Benchmark Models":
        st.header("Model Performance Benchmarking")
        uploaded_file = st.file_uploader("Upload test dataset (CSV or ZIP)", type=["csv", "zip"])

        if uploaded_file:
            image_paths = []
            true_labels = []

            if uploaded_file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Test dataset preview:", df.head())

                    if "image_path" in df.columns and "true_label" in df.columns:
                        image_paths = df["image_path"].tolist()
                        true_labels = df["true_label"].tolist()
                    else:
                        st.error("CSV must contain 'image_path' and 'true_label' columns.")

                except Exception as e:
                    st.error(f"Error processing CSV: {e}")

            elif uploaded_file.name.endswith(".zip"):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)

                        image_paths = [
                            os.path.join(temp_dir, file)
                            for file in os.listdir(temp_dir)
                            if file.lower().endswith((".jpg", ".jpeg", ".png"))
                        ]

                        try:
                            true_labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
                            st.warning("Labels inferred from folder names in the ZIP file.")
                        except:
                            st.warning("No labels provided in CSV or inferable from ZIP structure. Metrics cannot be calculated.")

                        st.write(f"Extracted {len(image_paths)} images from the ZIP file.")

                except Exception as e:
                    st.error(f"Failed to process ZIP file: {e}")

            if image_paths:
                benchmark_results = benchmark_models(image_paths, true_labels)
                if benchmark_results:
                    results_df = pd.DataFrame(benchmark_results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
                    st.write(results_df)

                    st.subheader("Comparison Graphs")
                    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    for metric in metrics:
                        ax.plot(results_df["Model"], results_df[metric], marker="o", label=metric)

                    ax.set_xlabel("Model", fontsize=12)
                    ax.set_ylabel("Score", fontsize=12)
                    ax.set_title("Model Performance Comparison", fontsize=14)
                    ax.tick_params(axis='x', rotation=45, ha='right', fontsize=10)
                    ax.legend(fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
