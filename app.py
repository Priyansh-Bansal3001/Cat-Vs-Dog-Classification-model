import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("dog_vs_cat_model.h5")


# Define prediction function
def predict_image(img: Image.Image):
    img = img.resize((256, 256))
    img_array = np.array(img)

    # Check if it has 3 color channels
    if img_array.shape[-1] != 3:
        return "Invalid image (must have 3 channels)"

    img_array = img_array.reshape(1, 256, 256, 3) / 255.0
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return f"🐶 Dog (Confidence: {prediction:.2f})"
    else:
        return f"🐱 Cat (Confidence: {1 - prediction:.2f})"


# Build the Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Dog vs Cat Classifier 🐶🐱",
    description="Upload an image of a dog or a cat to classify it."
)

# Launch the app
interface.launch()
