import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gradio as gr
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ExifTags


# Core algorithms (copied from Streamlit app for reuse)
def img_to_float01(pil_img):
    return np.asarray(pil_img).astype(np.float32) / 255.0


def rgb_to_gray(rgb01):
    return (0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2])


def normalize_to_display(img_data):
    d_min = img_data.min()
    d_max = img_data.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(img_data)
    return (img_data - d_min) / (d_max - d_min)


def compute_fft(gray_img):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)
    return normalize_to_display(mag)


def compute_radial_profile(fft_mag):
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), fft_mag.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    return radial_profile[: min(cx, cy)]


def compute_ela(pil_img, quality=90):
    pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    diff = ImageChops.difference(pil_img, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(diff).enhance(scale)
    return ela


def compute_noise_residual(rgb01, blur_radius=1.5):
    pil_source = Image.fromarray((rgb01 * 255).astype(np.uint8))
    blurred = pil_source.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    b01 = np.asarray(blurred).astype(np.float32) / 255.0
    diff = np.abs(rgb01 - b01)
    diff_gray = rgb_to_gray(diff)
    p99 = np.percentile(diff_gray, 99.5)
    return np.clip(diff_gray / (p99 + 1e-6), 0, 1)


def compute_chromatic_aberration(rgb01):
    r = rgb01[..., 0]
    b = rgb01[..., 2]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = kx.T
    try:
        import scipy.signal

        gx = scipy.signal.convolve2d(r, kx, mode="same", boundary="symm")
        gy = scipy.signal.convolve2d(r, ky, mode="same", boundary="symm")
        mag_r = np.sqrt(gx**2 + gy**2)
        gx_b = scipy.signal.convolve2d(b, kx, mode="same", boundary="symm")
        gy_b = scipy.signal.convolve2d(b, ky, mode="same", boundary="symm")
        mag_b = np.sqrt(gx_b**2 + gy_b**2)
    except ImportError:
        mag_r = r
        mag_b = b
    diff = np.abs(mag_r - mag_b)
    return normalize_to_display(diff)


def compute_saturation_map(rgb01):
    cmax = rgb01.max(axis=2)
    cmin = rgb01.min(axis=2)
    delta = cmax - cmin
    s = np.zeros_like(cmax)
    mask = cmax > 0
    s[mask] = delta[mask] / cmax[mask]
    return s


def compute_illumination_map(rgb01):
    gray = rgb_to_gray(rgb01)
    kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    ky = kx.T
    try:
        import scipy.signal

        gx = scipy.signal.convolve2d(gray, kx, mode="same", boundary="symm")
        gy = scipy.signal.convolve2d(gray, ky, mode="same", boundary="symm")
    except ImportError:
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
    angle = np.arctan2(gy, gx)
    magnitude = np.sqrt(gx**2 + gy**2)
    hue = (angle + np.pi) / (2 * np.pi)
    p95 = np.percentile(magnitude, 95)
    value = np.clip(magnitude / (p95 + 1e-6), 0, 1)
    saturation = np.ones_like(hue)
    hsv = np.dstack((hue, saturation, value))
    rgb_map = mcolors.hsv_to_rgb(hsv)
    return rgb_map


def get_ai_generation_info(pil_img):
    info_dict = {}
    if pil_img.info:
        if "parameters" in pil_img.info:
            info_dict["Stable Diffusion Parameters"] = pil_img.info["parameters"]
        if "Comment" in pil_img.info:
            try:
                comment_json = json.loads(pil_img.info["Comment"])
                info_dict["NovelAI Generation"] = comment_json
            except Exception:
                info_dict["Comment"] = pil_img.info["Comment"]
        for k in ["Software", "Description", "Source", "workflow"]:
            if k in pil_img.info:
                info_dict[k] = pil_img.info[k]
    try:
        exif = pil_img.getexif()
        if exif and 37510 in exif:
            val = exif[37510]
            if isinstance(val, bytes):
                try:
                    val = val.decode("ascii", errors="ignore")
                except Exception:
                    pass
            info_dict["UserComment"] = val
    except Exception:
        pass
    return info_dict


# Helpers to format outputs for Gradio
def to_rgb_image(arr, cmap="inferno"):
    norm = normalize_to_display(arr)
    cmap_fn = plt.get_cmap(cmap)
    rgb = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def plot_radial(radial_profile):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(radial_profile, color="#ff4b4b", linewidth=2)
    ax.set_title("Frequency Decay Curve")
    ax.set_xlabel("Frequency (Low -> High)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def saturation_heatmap(sat_map):
    fig, ax = plt.subplots()
    im = ax.imshow(sat_map, cmap="jet")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def build_metadata_summary(pil_img):
    lines = []
    ai_info = get_ai_generation_info(pil_img)
    if ai_info:
        lines.append("[!] Suspected AI parameters found:")
        for k in ai_info:
            lines.append(f"- {k}")
    else:
        lines.append("[ ] No explicit AI parameters found.")

    lines.append("\nEXIF:")
    try:
        exif = pil_img.getexif()
        if exif:
            count = 0
            for tag, val in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if isinstance(val, str) and len(val) < 40:
                    lines.append(f"{decoded}: {val}")
                    count += 1
                    if count > 10:
                        break
        else:
            lines.append("No EXIF data detected.")
    except Exception:
        lines.append("Failed to read EXIF.")
    return "\n".join(lines)


def analyze(image, ela_quality, blur_radius):
    if image is None:
        return [None] * 9

    pil_img = image.convert("RGB")
    w_orig, h_orig = pil_img.size
    max_analysis = 1200
    scale_analysis = min(1.0, max_analysis / max(w_orig, h_orig))
    pil_small = pil_img.resize(
        (int(w_orig * scale_analysis), int(h_orig * scale_analysis)),
        Image.Resampling.LANCZOS,
    )

    img_np = img_to_float01(pil_small)
    img_gray = rgb_to_gray(img_np)
    fft_res = compute_fft(img_gray)
    ela_res = compute_ela(pil_small, ela_quality)
    noise_res = compute_noise_residual(img_np, blur_radius)

    fft_img = to_rgb_image(fft_res, cmap="inferno")
    radial_img = plot_radial(compute_radial_profile(fft_res))
    noise_img = Image.fromarray((normalize_to_display(noise_res) * 255).astype(np.uint8))
    ca_img = to_rgb_image(compute_chromatic_aberration(img_np), cmap="magma")
    illum_img = Image.fromarray(
        (normalize_to_display(compute_illumination_map(img_np)) * 255).astype(np.uint8)
    )
    sat_img = saturation_heatmap(compute_saturation_map(img_np))
    meta_text = build_metadata_summary(pil_img)

    return [
        pil_img,
        meta_text,
        fft_img,
        radial_img,
        ela_res,
        noise_img,
        ca_img,
        illum_img,
        sat_img,
    ]


def build_interface():
    with gr.Blocks(title="AI Image Forensics (Gradio)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## AI Image Forensics (Gradio)\n"
            "Upload an image to inspect metadata, frequency fingerprints, noise, and lighting cues. "
            "Set parameters then click **Analyze** to generate all views."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                ela_quality = gr.Slider(
                    minimum=50,
                    maximum=99,
                    step=1,
                    value=90,
                    label="ELA Compression Quality",
                )
                blur_radius = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                    value=1.5,
                    label="Noise Separation Radius",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Overview"):
                    orig_out = gr.Image(label="Original", type="pil")
                    meta_out = gr.Textbox(
                        label="Metadata Summary",
                        lines=12,
                        interactive=False,
                    )
                with gr.Tab("Frequency"):
                    fft_out = gr.Image(label="2D FFT Spectrum", type="pil")
                    radial_out = gr.Image(label="Frequency Decay Curve", type="pil")
                with gr.Tab("Signal"):
                    ela_out = gr.Image(label="ELA", type="pil")
                    noise_out = gr.Image(label="Noise Residual", type="pil")
                with gr.Tab("Physics"):
                    ca_out = gr.Image(label="Chromatic Aberration", type="pil")
                    illum_out = gr.Image(label="Illumination Gradient", type="pil")
                    sat_out = gr.Image(label="Saturation Heatmap", type="pil")

        analyze_btn.click(
            analyze,
            inputs=[image_input, ela_quality, blur_radius],
            outputs=[
                orig_out,
                meta_out,
                fft_out,
                radial_out,
                ela_out,
                noise_out,
                ca_out,
                illum_out,
                sat_out,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0")
