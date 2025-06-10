import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("kidney_stone_detection_model_final.keras")

# Define prediction function
def predict(image):
    image = image.resize((150, 150))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    predictions = model.predict(img_array)[0]
    classes = ["Normal", "Stone"]
    return {classes[i]: float(predictions[i]) for i in range(2)}

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload CT Image"),
    outputs=gr.Label(num_top_classes=2),
    title="Kidney Stone Detection",
    description="Upload a kidney CT scan image to predict if it contains a stone."
)

# Launch interface
interface.launch()
