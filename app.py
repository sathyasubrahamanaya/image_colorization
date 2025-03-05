# app.py
import streamlit as st
from PIL import Image
import model  # Import the module defined in model.py

# Configure the page
st.set_page_config(page_title="Image Colorization", layout="wide")
st.title("Image Colorization App")
st.write("Upload a grayscale image and see it transformed into a colorful image using our U-Net based GAN model.")

@st.cache(allow_output_mutation=True)
def load_model():
    """
    Loads and caches the colorization model.
    """
    return model.load_colorization_model()

# Load the model
try:
    colorization_model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error("Error opening image.")
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Colorize"):
            with st.spinner("Colorizing..."):
                try:
                    output_rgb = model.colorize_image(colorization_model, image)
                    # Convert output (NumPy array in range [0, 1]) to a PIL Image for display
                    output_image = Image.fromarray((output_rgb * 255).astype("uint8"))
                    st.image(output_image, caption="Colorized Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error during colorization: {e}")
