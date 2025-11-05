import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import zipfile
import tempfile

# List of models
MODEL_NAMES = [
    "custom_cnn_model","inception_model",
    "vgg16_model","nasnet_model","resnet_model",
    "xception_model","mobilenet_model"
]

# Class labels (Update based on your dataset)
CLASS_LABELS = ["black widow", "captain america", "doctor strange", "ironman", "hulk", "loki", "spider-man", "thanos"]

# Function to load the model
@st.cache_resource
def load_model(model_name):
    model_path = f"C:/Users/Manvi Bhala/Desktop/CVA/{model_name}.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

# Sidebar: Main Navigation
st.sidebar.title("Model Analysis")
page = st.sidebar.selectbox("Choose a page:", ["Architecture", "Benchmarks", "Inference"])

if page == "Architecture":
    st.title("Model Architectures")
    image_height = 300  # Adjust as needed

    # Use st.columns to create two columns per row
    cols = st.columns(2)  # Create 2 columns

    for i, model_name in enumerate(MODEL_NAMES):
        arch_path = f"C:/Users/Manvi Bhala/Desktop/CVA/{model_name}.png"
        if os.path.exists(arch_path):
            image = Image.open(arch_path)
            resized_image = image.resize((int(image.width * (image_height / image.height)), image_height))
            
            # Place each image in the appropriate column
            with cols[i % 2]:  # Cycle between 0 and 1 for the two columns
                st.header(model_name)
                st.image(resized_image, caption=f"{model_name} Architecture", use_column_width=False)
        else:
            # Display error in the correct column
            with cols[i % 2]:
                st.error(f"Architecture image not found for {model_name}.")

        # Add a spacer after every two images, unless it's the last one
        if i % 2 == 1 and i < len(MODEL_NAMES) - 1:  # Add spacer after every 2 images
            st.markdown("<br>", unsafe_allow_html=True)

elif page == "Benchmarks":
    st.title("Model Performance Comparison")
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
                        if len(set(true_labels)) != len(CLASS_LABELS):
                            st.warning("ZIP file labels inferred from folder names. Verify they match your CLASS_LABELS.")
                    except:
                        st.warning("No labels provided in CSV or inferable from ZIP structure. Metrics cannot be calculated.")
                        true_labels = None

                    st.write(f"Extracted {len(image_paths)} images from the ZIP file.")

            except Exception as e:
                st.error(f"Failed to process ZIP file: {e}")

        if image_paths:
            results = []
            for model_name in MODEL_NAMES:
                model = load_model(model_name)
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

                if true_labels:
                    valid_indices = [i for i, y in enumerate(y_pred) if y != -1]
                    y_true_valid = [true_labels[i] for i in valid_indices]
                    y_pred_valid = [y_pred[i] for i in valid_indices]

                    if y_true_valid:
                        acc = accuracy_score(y_true_valid, y_pred_valid)
                        prec = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=1)
                        rec = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=1)
                        f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=1)
                        results.append([model_name, acc, prec, rec, f1])
                    else:
                        st.warning(f"No valid predictions for {model_name}. Skipping metrics calculation.")

                else:
                    st.warning("True labels not provided. Metrics cannot be calculated.")

            if results:
                results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
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
            else:
                st.warning("No model results to display.")

        else:
            st.error("No valid images were found for processing.")

elif page == "Inference":
    st.title("Model Inference")
    uploaded_image = st.file_uploader("Upload an image for classification", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            display_width = 150  
            display_height = 150
            resized_image = image.resize((display_width, display_height))  # Correctly resize the image

            st.image(resized_image, caption="Uploaded Image", use_column_width=False)  # Display the resized image

            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            all_predictions = {}

            for model_name in MODEL_NAMES:
                model = load_model(model_name)
                prediction = model.predict(img_array, verbose=0)

                if prediction.size == 0:
                    st.error(f"Model {model_name} did not return any predictions.")
                elif prediction.shape[1] != len(CLASS_LABELS):
                    st.error(f"Mismatch for {model_name}: Model predicts {prediction.shape[1]} classes, but {len(CLASS_LABELS)} class labels are defined.")
                else:
                    predicted_index = np.argmax(prediction)
                    predicted_class = CLASS_LABELS[predicted_index]
                    all_predictions[model_name] = predicted_class

            for model_name, predicted_class in all_predictions.items():
                st.subheader(f"{model_name} Prediction: {predicted_class}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

