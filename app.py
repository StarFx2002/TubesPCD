import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import os
import matplotlib.pyplot as plt
import cv2
import base64

st.title("Pengaturan Gambar dengan Operasi Titik")
st.write("Aplikasi ini memungkinkan untuk mengubah gambar menggunakan operasi titik seperti brightness, contrast, gamma correction, inversi, dan thresholding.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

def rotate_image(img_array, angle_degrees):
    angle = np.deg2rad(angle_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    height, width = img_array.shape[:2]

    output = np.ones_like(img_array) * 255  # white background

    cx, cy = width // 2, height // 2  # center of image

    for y in range(height):
        for x in range(width):
            # Coordinates relative to center
            xr = x - cx
            yr = y - cy

            # Apply inverse rotation
            xs = int(cx + cos_a * xr + sin_a * yr)
            ys = int(cy - sin_a * xr + cos_a * yr)

            if 0 <= xs < width and 0 <= ys < height:
                output[y, x] = img_array[ys, xs]
            else:
                output[y, x] = 255  # white background

    return output

def box_blur(img_array, radius):
    if radius == 0:
        return img_array
    kernel_size = 2 * radius + 1
    padded = np.pad(img_array, ((radius, radius), (radius, radius), (0,0)), mode='edge')
    blurred = np.zeros_like(img_array, dtype=np.float32)
    
    # Iterate over kernel window and accumulate
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            blurred += padded[dy:dy+img_array.shape[0], dx:dx+img_array.shape[1], :]
    blurred /= (kernel_size * kernel_size)
    
    return blurred.astype(np.uint8)

def clamp(array):
    return np.clip(array, 0, 255).astype(np.uint8)

if uploaded_file:
    # Extract original name and extension
    original_name, ext = os.path.splitext(uploaded_file.name)
    edited_filename = f"{original_name}_edited.png"  # Save edited as PNG

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="Gambar Asli", use_container_width=True)

    st.subheader("Pengaturan Operasi Titik")

    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast (Skala Multiplier)", 0.5, 3.0, 1.0)
    gamma = st.slider("Gamma", 0.1, 3.0, 1.0)
    apply_negative = st.checkbox("Inversi (Negatif)")
    apply_threshold = st.checkbox("Thresholding")
    threshold_value = st.slider("Nilai Ambang", 0, 255, 128) if apply_threshold else None

    # Manual point operations
    # Brightness and contrast
    adjusted = img_array.astype(np.float32) * contrast + brightness
    adjusted = clamp(adjusted)

    # Negative
    if apply_negative:
        adjusted = 255 - adjusted

    # Gamma correction
    gamma_corrected = clamp(255.0 * (adjusted / 255.0) ** gamma)

    # Thresholding (optional grayscale first)
    if apply_threshold:
        grayscale = np.mean(gamma_corrected, axis=2)
        binary = np.where(grayscale >= threshold_value, 255, 0).astype(np.uint8)
        gamma_corrected = np.stack([binary] * 3, axis=-1)

    #Flip the image
    flip_horizontal = st.checkbox("Flip Horizontal")
    flip_vertical = st.checkbox("Flip Vertical")
    if flip_horizontal:
        gamma_corrected = np.fliplr(gamma_corrected)
    if flip_vertical:
        gamma_corrected = np.flipud(gamma_corrected)

    # Box Blur
    blur_radius = st.slider("Box Blur Radius", 0, 15, 0)
    gamma_corrected = box_blur(gamma_corrected, blur_radius)

    st.image(gamma_corrected, caption="Hasil Transformasi", use_container_width=True)

    # Plot histogram using OpenCV and matplotlib
    fig, ax = plt.subplots(figsize=(3, 2))
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([gamma_corrected], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title("Histogram Warna")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Count")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    st.markdown(
    f"""
    <style>
    .sticky-histogram {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: white;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
        z-index: 9999;
    }}
    </style>
    <div class="sticky-histogram">
        <img src="data:image/png;base64,{img_base64}" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

    # --- Crop ---
    st.subheader("Crop Image")
    width, height = image.size

    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("Left Crop", 0, width - 1, 0)
        y = st.slider("Top Crop", 0, height - 1, 0)
    with col2:
        max_crop_width = width
        max_crop_height = height
        crop_width = st.slider("Right Crop", 1, max_crop_width, max_crop_width)
        crop_height = st.slider("Bottom Crop", 1, max_crop_height, max_crop_height)

    # Crop the image array
    cropped_array = gamma_corrected[y:y+crop_height, x:x+crop_width, :]

    # Create blank white image the size of original
    blank_canvas = Image.new("RGB", (width, height), (255, 255, 255))
    # Convert cropped_array back to PIL Image
    cropped_img = Image.fromarray(cropped_array)
    # Paste cropped image onto blank canvas at top-left corner (0,0)
    blank_canvas.paste(cropped_img, (0, 0))
    # Display fixed-size image (original size) with cropped content shown in the corner
    MAX_DISPLAY_WIDTH = 700
    display_width = min(width, MAX_DISPLAY_WIDTH)
    st.image(blank_canvas, caption="Gambar setelah Crop", use_container_width=False, width=display_width)

    # Rotate
    angle = st.slider("Rotate Image (degrees)", -180, 180, 0)
    cropped_array = rotate_image(cropped_array, angle)

    st.image(cropped_array, caption="Hasil Rotasi", use_container_width=True)

    # Export / Download
    result_image = Image.fromarray(cropped_array)
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="💾 Save Picture",
        data=buffer,
        file_name=edited_filename,
        mime="image/png"
    )