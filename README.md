# Image Feature Panel

A lightweight image feature visualization tool built with Streamlit. It extracts frequency-domain, signal-layer, and physical-layer features to help users judge whether an image is straight out of camera or AI-generated/post-processed composite.

  🔗 Live Demo: [image-feature-panel.streamlit.app](https://image-feature-panel.streamlit.app)

![Layer 5](https://github.com/user-attachments/assets/7fb651ba-7681-44b1-9297-d05132b16a4f)
<img width="3000" height="2250" alt="forensic_report-4" src="https://github.com/user-attachments/assets/47f23383-6511-423e-9dfe-076d2fffec21" />

## 📦 Feature List

* **Metadata Inspection**: Automatically extract EXIF information and hidden AI generation parameters (supports Stable Diffusion, Midjourney, NovelAI, etc.).
* **Frequency-Domain Analysis (FFT)**: 2D spectrum and radial energy decay curves for detecting grid effects and abnormal high-frequency signals.

* **ELA (Error Level Analysis)**: Detect recompression differences and splicing artifacts.
* **Noise Residuals**: Separate image content to reveal the underlying sensor noise distribution.


* **Lighting Gradient**: Visualize lighting direction to check lighting/shadow consistency.
* **Chromatic Aberration**: Detect purple fringing/dispersion at lens edges.
* **Saturation**: Saturation heatmap for shadow regions.


* **Interactive Microscope**: Click-to-zoom to inspect real-time FFT features of specific regions.
* **Report Export**: One-click export of a PNG panel containing all feature maps.

## 🚀 Quick Start

### 1. Install dependencies

Ensure Python 3.8+ is installed, then run:

```bash
pip install -r requirements.txt

```

### 2. Run the app

```bash
streamlit run app.py

```

### 3. Usage

Once the browser opens at `http://localhost:8501`, upload an image in the left sidebar to start analysis.

![Layer 4](https://github.com/user-attachments/assets/7e3e9dc9-2e01-4f6f-a342-daa60c205d52)
![Layer 3](https://github.com/user-attachments/assets/a9eaa61b-1621-42b6-a85f-e82909d9b0f2)
![Layer 2](https://github.com/user-attachments/assets/742beadf-1ba5-4667-be78-df9268af2aa4)
![Layer 1](https://github.com/user-attachments/assets/38b60d42-fc9f-4894-a3fb-80594a1486b9)

## 📊 Feature Reference Table

| Feature Dimension | 📸 Real Photo Tendencies | 🤖 AI-Generated/Forged Tendencies |
| --- | --- | --- |
| **FFT Spectrum** | Energy smoothly decays from the center outward | Isolated bright spots (off-center), regular grids, or checkerboard textures |
| **Noise Residuals** | Even grain (photon noise) | Overly smooth (waxy) surfaces, or striped noise in details |
| **ELA** | Noise distribution is relatively uniform | Local regions (e.g., faces) have drastically different noise from other areas |
| **Lighting Direction** | Smooth lighting gradients on surfaces | Lighting direction is chaotic, colors appear as random speckles |
| **Chromatic Aberration** | Natural dispersion at edge highlights | No aberration at all (all black) or random color speckles across the image |
| **Metadata** | Contains aperture, shutter, ISO, and other camera info | Includes `parameters`, `steps`, or no EXIF data |

## 🛠️ Tech Stack

* **UI**: Streamlit
* **Computation**: Numpy, Scipy
* **Image Processing**: Pillow (PIL), OpenCV (algorithm logic)
* **Plotting**: Matplotlib

## 📄 License

MIT License
