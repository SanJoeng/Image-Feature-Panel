import matplotlib
import streamlit as st
import numpy as np
import io
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ExifTags

# ==========================================
# 0. Dependency Check
# ==========================================
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Please install the click interaction package first: pip install streamlit-image-coordinates")
    st.stop()

# ==========================================
# 1. Core Algorithms: Feature Extraction
# ==========================================

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

# --- A. Frequency Analysis ---
def compute_fft(gray_img):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)
    return normalize_to_display(mag)

def compute_radial_profile(fft_mag):
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
    tbin = np.bincount(r.ravel(), fft_mag.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    return radial_profile[:min(cx, cy)]

# --- B. Signal Analysis ---
def compute_ela(pil_img, quality=90):
    pil_img = pil_img.convert('RGB')
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    diff = ImageChops.difference(pil_img, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
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

# --- C. Physical Analysis ---
def compute_chromatic_aberration(rgb01):
    r = rgb01[..., 0]
    b = rgb01[..., 2]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = kx.T
    try:
        import scipy.signal
        gx = scipy.signal.convolve2d(r, kx, mode='same', boundary='symm')
        gy = scipy.signal.convolve2d(r, ky, mode='same', boundary='symm')
        mag_r = np.sqrt(gx**2 + gy**2)
        gx_b = scipy.signal.convolve2d(b, kx, mode='same', boundary='symm')
        gy_b = scipy.signal.convolve2d(b, ky, mode='same', boundary='symm')
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
        gx = scipy.signal.convolve2d(gray, kx, mode='same', boundary='symm')
        gy = scipy.signal.convolve2d(gray, ky, mode='same', boundary='symm')
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

# --- D. Metadata Mining ---
def get_ai_generation_info(pil_img):
    info_dict = {}
    if pil_img.info:
        if 'parameters' in pil_img.info:
            info_dict['Stable Diffusion Parameters'] = pil_img.info['parameters']
        if 'Comment' in pil_img.info:
            try:
                comment_json = json.loads(pil_img.info['Comment'])
                info_dict['NovelAI Generation'] = comment_json
            except:
                info_dict['Comment'] = pil_img.info['Comment']
        for k in ['Software', 'Description', 'Source', 'workflow']:
            if k in pil_img.info:
                info_dict[k] = pil_img.info[k]
    try:
        exif = pil_img.getexif()
        if exif and 37510 in exif:
            val = exif[37510]
            if isinstance(val, bytes):
                try: val = val.decode('ascii', errors='ignore')
                except: pass
            info_dict['UserComment'] = val
    except: pass
    return info_dict

# --- E. Dashboard Export (English Titles) ---
def generate_dashboard_figure(pil_img, img_np, img_gray, ela_img, quality, blur):
    fig = plt.figure(figsize=(20, 15), facecolor='white')
    
    # 1. Original
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(pil_img)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. FFT
    ax2 = fig.add_subplot(3, 3, 2)
    fft_img = compute_fft(img_gray)
    ax2.imshow(fft_img, cmap='inferno')
    ax2.set_title("FFT Spectrum (Log)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Radial
    ax3 = fig.add_subplot(3, 3, 3)
    rad_prof = compute_radial_profile(fft_img)
    ax3.plot(rad_prof, color='red', lw=2)
    ax3.set_title("Frequency Decay Curve", fontsize=14, fontweight='bold') # Fixed Title
    ax3.set_xlabel("Frequency (Low -> High)")
    ax3.set_ylabel("Power")
    ax3.grid(True, alpha=0.3)
    
    # 4. ELA
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(ela_img)
    ax4.set_title(f"ELA (Quality={quality})", fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # 5. Noise
    ax5 = fig.add_subplot(3, 3, 5)
    noise_img = compute_noise_residual(img_np, blur_radius=blur)
    ax5.imshow(noise_img, cmap='gray')
    ax5.set_title(f"Noise Residual (r={blur})", fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # 6. Chromatic Aberration
    ax6 = fig.add_subplot(3, 3, 6)
    ca_img = compute_chromatic_aberration(img_np)
    ax6.imshow(ca_img, cmap='magma')
    ax6.set_title("Chromatic Aberration Map", fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # 7. Saturation
    ax7 = fig.add_subplot(3, 3, 7)
    sat_img = compute_saturation_map(img_np)
    ax7.imshow(sat_img, cmap='jet')
    ax7.set_title("Saturation Heatmap", fontsize=14, fontweight='bold')
    ax7.axis('off')
    
    # 8. Illumination
    ax8 = fig.add_subplot(3, 3, 8)
    illum_map = compute_illumination_map(img_np)
    ax8.imshow(illum_map)
    ax8.set_title("Illumination Gradient", fontsize=14, fontweight='bold')
    ax8.axis('off')
    
    # 9. Meta Info
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    info_str = "METADATA SUMMARY\n----------------\n"
    ai_info = get_ai_generation_info(pil_img)
    if ai_info:
        info_str += "[!] AI PARAMS FOUND:\n"
        for k in ai_info.keys():
            info_str += f" - {k}\n"
    else:
        info_str += "[ ] No explicit AI params\n"
    
    info_str += "\nEXIF DATA:\n"
    try:
        exif = pil_img.getexif()
        if exif:
            count = 0
            for tag, val in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if isinstance(val, str) and len(val) < 30:
                    info_str += f"{decoded}: {val}\n"
                    count += 1
                    if count > 10: break
    except: pass
    ax9.text(0.05, 0.95, info_str, fontsize=10, fontfamily='monospace', va='top')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# ==========================================
# 2. UI Main Program
# ==========================================

st.set_page_config(layout="wide", page_title="AI Image Forensics")

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Analysis Console")
    uploaded_file = st.file_uploader("📂 Upload Image", type=['jpg','jpeg','png','webp','tiff'])
    st.divider()
    st.subheader("Parameter Tuning")
    ela_quality = st.slider("ELA Compression Quality", 50, 99, 90)
    blur_radius = st.slider("Noise Separation Radius", 0.5, 5.0, 1.5)
    st.info("Tip: Use the Microscope tab to inspect details.")

st.title("🕵️‍♂️ AI Image Forensics")

if not uploaded_file:
    st.warning("👈 Please upload an image from the left panel to start.")
    st.stop()

# --- Preprocessing ---
pil_img = Image.open(uploaded_file).convert('RGB')
w_orig, h_orig = pil_img.size

max_analysis = 1200
scale_analysis = min(1.0, max_analysis / max(w_orig, h_orig))
pil_small = pil_img.resize((int(w_orig * scale_analysis), int(h_orig * scale_analysis)), Image.Resampling.LANCZOS)

img_np = img_to_float01(pil_small)
img_gray = rgb_to_gray(img_np)
fft_res = compute_fft(img_gray)
ela_res = compute_ela(pil_small, ela_quality)
noise_res = compute_noise_residual(img_np, blur_radius)

# --- Export ---
col_title, col_export = st.columns([5, 1])
with col_export:
    if st.button("📸 Export Report (PNG)"):
        with st.spinner("Generating dashboard..."):
            dash_bytes = generate_dashboard_figure(pil_small, img_np, img_gray, ela_res, ela_quality, blur_radius)
            st.download_button("⬇️ Download", dash_bytes, "forensic_report.png", "image/png")

# --- Tabs ---
tab_meta, tab_freq, tab_signal, tab_physics, tab_micro = st.tabs([
    "📂 Metadata", "📈 Frequency (FFT)", "📶 Signal (ELA/Noise)", "🌈 Physics (Light/CA)", "🔬 Microscope"
])

# 1. Metadata
with tab_meta:
    c1, c2 = st.columns([3, 2])
    c1.image(pil_img, caption=f"Resolution: {w_orig}x{h_orig}", use_container_width=True)
    with c2:
        st.subheader("🕵️‍♂️ Hidden Parameters")
        ai_info = get_ai_generation_info(pil_img)
        if ai_info:
            st.error("🚨 Suspected AI generation metadata found! (Smoking Gun)")
            for k, v in ai_info.items():
                with st.expander(f"📌 {k}", expanded=True):
                    st.code(v, language='text')
            st.caption("🔍 If you see Prompts/Seed/Steps here, it's almost certain the image is AI-generated.")
        else:
            st.success("✅ No explicit AI parameters found in the file header.")
            st.caption("Note: This does not prove the image is genuine. Metadata can be stripped or re-saved.")
        
        st.divider()
        st.subheader("📷 Standard EXIF")
        exif_data = {}
        try:
            info = pil_img.getexif()
            if info:
                for tag, value in info.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    if isinstance(value, bytes): continue
                    if isinstance(value, str): value = value[:50]
                    exif_data[decoded] = value
                st.dataframe(exif_data, use_container_width=True, height=400)
            else:
                st.warning("⚠️ No EXIF data found")
                st.caption("Camera originals usually include aperture/shutter/ISO. Empty EXIF increases suspicion.")
        except: st.error("Failed to read EXIF")

# 2. Frequency Domain
with tab_freq:
    st.info("""
    **📊 Reading Guide:**
    * **ℹ️ About the bright cross**:
        * Almost every image (real or AI) shows a bright cross at the center.
        * **This is normal** spectral leakage, **not** an AI cue. Focus on areas outside the cross.
    * **✅ Genuine photos**:
        * Energy decays smoothly and randomly from center to edges, like fog.
        * No abrupt bright spots or geometric patterns.
    * **🚨 AI artifacts (Smoking Gun)**:
        1. **Isolated bright dots** far from the center on dark background.
        2. **Regular grids** or checkerboard-like point patterns over the spectrum.
        *Reason: CNN upsampling leaves periodic fingerprints.*
    """)
    c1, c2 = st.columns(2)
    c1.image(fft_res, clamp=True, use_container_width=True, caption="2D FFT Spectrum (Log Scale)")
    
    fig_rad, ax = plt.subplots(figsize=(6,3))
    ax.plot(compute_radial_profile(fft_res), color='#ff4b4b', linewidth=2)
    ax.set_title("Frequency Decay Curve") # Fixed Title
    ax.set_xlabel("Frequency (Low -> High)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)
    c2.pyplot(fig_rad)
    c2.caption("🔍 Interpretation: A smooth downward curve is normal. A late upward bump (right side) signals abnormal high-frequency noise.")

with tab_signal:
    st.info("""
    **📊 Reading Guide:**
    * **ELA (Error Level Analysis)**:
        * **✅ Genuine**: Noise evenly distributed like thin sand. Brighter at complex textures (e.g., leaves) is normal.
        * **🚨 Splicing/Edits**: If face noise/brightness differs sharply from background (e.g., red noise vs blue noise), likely pasted.
    * **Noise Residual**:
        * **✅ Genuine**: Fine photon noise even at low ISO.
        * **🚨 AI**: Plastic/waxy smoothness or odd striped noise in details; lacks randomness.
    """)
    c1, c2 = st.columns(2)
    c1.image(ela_res, use_container_width=True, caption=f"ELA (Quality={ela_quality})")
    c2.image(noise_res, clamp=True, channels='GRAY', use_container_width=True, caption=f"Noise Residual (r={blur_radius})")

# 4. Physics
with tab_physics:
    st.info("""
    **📊 Reading Guide:**
    * **🌈 Illumination Gradient**: Colors encode light direction. On smooth surfaces (faces, spheres), colors should transition smoothly. Random colors mean broken lighting logic.
    * **🟣 Chromatic Aberration**: Real lenses show purple/green edges at highlights. AI often shows none (all black) or random speckles everywhere.
    * **🔥 Saturation**: Check shadows. Real shadows are low saturation. Bright saturated red/blue speckles in dark areas are typical diffusion artifacts.
    """)
    c1, c2, c3 = st.columns(3)
    
    # Illumination Map
    illum_map = compute_illumination_map(img_np)
    c1.image(illum_map, use_container_width=True, caption="Illumination Gradient (Dir)")
    
    # Chromatic Aberration
    c2.image(compute_chromatic_aberration(img_np), clamp=True, use_container_width=True, caption="Chromatic Aberration")
    
    # Saturation
    sat_map = compute_saturation_map(img_np)
    fig_sat, ax = plt.subplots()
    im = ax.imshow(sat_map, cmap='jet')
    ax.axis('off')
    c3.pyplot(fig_sat)
    c3.caption("Saturation Heatmap (Red=High)")

# 5. Microscope
with tab_micro:
    st.markdown("#### 🔬 Interactive Microscope")
    st.caption("👈 Click any point on the Navigation panel to inspect that region in high resolution.")

    col_nav, col_zoom = st.columns([1, 2])

    with col_nav:
        st.subheader("1. Navigation (click to locate)")
        
        crop_size = st.slider(
            "🔎 View Window (px)",
            min_value=50, max_value=1200, value=400, step=50,
            help="Larger value = wider view (lower magnification); smaller = higher magnification."
        )

        # Use thumbnail mode to preserve aspect ratio within 350x350 box
        # Copy for display
        pil_nav = pil_img.copy()
        
        # Generate thumbnail (in-place, keep ratio)
        pil_nav.thumbnail((350, 350), Image.Resampling.LANCZOS)
        
        nav_w, nav_h = pil_nav.size
        
        coords = streamlit_image_coordinates(
            pil_nav,
            key="zoom_click"
        )

        if coords:
            scale_x = w_orig / nav_w
            scale_y = h_orig / nav_h
            
            center_x = int(coords['x'] * scale_x)
            center_y = int(coords['y'] * scale_y)
        else:
            center_x = w_orig // 2
            center_y = h_orig // 2

        st.info(f"Original coordinates: ({center_x}, {center_y})")

    with col_zoom:
        st.subheader("2. Detail (hi-res crop)")

        half_size = crop_size // 2
        x0 = max(0, center_x - half_size)
        y0 = max(0, center_y - half_size)
        x1 = min(w_orig, center_x + half_size)
        y1 = min(h_orig, center_y + half_size)

        crop_img = pil_img.crop((x0, y0, x1, y1))
        
        st.image(crop_img, use_container_width=True)

        # Local FFT analysis
        with st.expander("View FFT of this region (exclude background influence)", expanded=True):
            crop_np = img_to_float01(crop_img)
            crop_gray_small = rgb_to_gray(crop_np)
            crop_fft = compute_fft(crop_gray_small)
            
            st.image(
                crop_fft,
                clamp=True,
                caption="Local FFT spectrum",
                use_container_width=True
            )
