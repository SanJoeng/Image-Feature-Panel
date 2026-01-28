import matplotlib
import streamlit as st
import numpy as np
import io
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ExifTags

# ==========================================
# 0. 依赖检查
# ==========================================
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("请先安装点击交互库: pip install streamlit-image-coordinates")
    st.stop()

# ==========================================
# 1. 核心算法: 特征提取
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

# --- A. 频域分析 ---
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

# --- B. 信号分析 ---
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

# --- C. 物理分析 ---
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

# --- D. 元数据深度挖掘 ---
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

# --- E. Dashboard 导出绘制 (English Titles) ---
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
# 2. UI 主程序
# ==========================================

st.set_page_config(layout="wide", page_title="AI 影像取证台")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🎛️ 分析控制台")
    uploaded_file = st.file_uploader("📂 导入图片", type=['jpg','jpeg','png','webp','tiff'])
    st.divider()
    st.subheader("参数微调")
    ela_quality = st.slider("ELA 压缩质量", 50, 99, 90)
    blur_radius = st.slider("噪点分离半径", 0.5, 5.0, 1.5)
    st.info("提示：Tab 5 可使用显微镜功能")

st.title("🕵️‍♂️ AI 影像取证")

if not uploaded_file:
    st.warning("👈 请先在左侧导入图片以开始工作流。")
    st.stop()

# --- 数据预处理 ---
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

# --- 导出 ---
col_title, col_export = st.columns([5, 1])
with col_export:
    if st.button("📸 导出报告(PNG)"):
        with st.spinner("生成 Dashboard..."):
            dash_bytes = generate_dashboard_figure(pil_small, img_np, img_gray, ela_res, ela_quality, blur_radius)
            st.download_button("⬇️ 下载", dash_bytes, "forensic_report.png", "image/png")

# --- Tabs ---
tab_meta, tab_freq, tab_signal, tab_physics, tab_micro = st.tabs([
    "📂 元数据侦测", "📈 频域 (FFT)", "📶 信号 (ELA/Noise)", "🌈 物理 (光照/色差)", "🔬 显微镜"
])

# 1. 元数据
with tab_meta:
    c1, c2 = st.columns([3, 2])
    c1.image(pil_img, caption=f"分辨率: {w_orig}x{h_orig}", use_container_width=True)
    with c2:
        st.subheader("🕵️‍♂️ 隐藏参数")
        ai_info = get_ai_generation_info(pil_img)
        if ai_info:
            st.error("🚨 发现疑似 AI 生成配置信息！(Smoking Gun)")
            for k, v in ai_info.items():
                with st.expander(f"📌 {k}", expanded=True):
                    st.code(v, language='text')
            st.caption("🔍 解读：如果这里出现了 Prompts, Seed 或 Steps，这几乎是 100% 的 AI 直出图证据。这是最直接的判定方式。")
        else:
            st.success("✅ 未在文件头中发现明文 AI 参数")
            st.caption("注意：这不代表不是 AI。可能是生成后经过了 PS 转存、微信发送或专门的 Metadata 清洗。")
        
        st.divider()
        st.subheader("📷 标准 EXIF")
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
                st.warning("⚠️ 无 EXIF 数据")
                st.caption("真实相机拍摄的原始照片通常会包含光圈、快门、ISO 等信息。如果 EXIF  полностью为空，可疑度增加。")
        except: st.error("无法读取 EXIF")

# 2. 频域
with tab_freq:
    st.info("""
    **📊 判读指南：**
    * **ℹ️ 关于“十字亮线”**：
        * 注意：你会在几乎所有图片（无论是实拍还是 AI）的中心看到明亮的十字线。
        * **这是正常的数学现象**（边缘频谱泄露），**不是**判断 AI 的依据，请忽略它，观察十字线以外的区域。
    * **✅ 真图特征**：
        * 十字线以外的区域，能量像云雾一样从中心向四周**平滑、随机地衰减**。
        * 没有突兀的亮点或几何规律。
    * **🚨 AI 伪影 (Smoking Gun)**：
        1.  **异常星点 (Artifact Dots)**：在远离中心的暗色背景中，出现**孤立的、不对称的明亮白点**（这是最强的 AI 特征）。
        2.  **规则网格 (Grids)**：仔细观察云雾背景，若隐若现地覆盖着像“棋盘”或“方格纸”一样的点阵结构。
        *原理：这是卷积神经网络 (CNN) 在上采样 (Upsampling) 生成图像时留下的周期性指纹。*
    """)
    c1, c2 = st.columns(2)
    c1.image(fft_res, clamp=True, use_container_width=True, caption="2D FFT 频谱 (Log Scale)")
    
    fig_rad, ax = plt.subplots(figsize=(6,3))
    ax.plot(compute_radial_profile(fft_res), color='#ff4b4b', linewidth=2)
    ax.set_title("Frequency Decay Curve") # Fixed Title
    ax.set_xlabel("Frequency (Low -> High)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)
    c2.pyplot(fig_rad)
    c2.caption("🔍 解读：正常曲线应平滑下降。如果在尾部（右侧高频区）突然上翘，说明存在非自然的高频噪声。")

# 3. 信号
with tab_signal:
    st.info("""
    **📊 判读指南：**
    * **ELA (误差水平分析)**：
        * **✅ 真图**：全图噪点分布均匀，像一层薄薄的沙子。复杂纹理（如树叶）处更亮是正常的。
        * **🚨 拼接/P图**：如果人脸区域的颜色/亮度与背景**截然不同**（例如背景是红噪点，人脸是蓝噪点），说明是后期贴上去的。
    * **Noise (噪声残差)**：
        * **✅ 真图**：即使是 ISO 100 的照片，放大看也会有细腻的**光子噪声**（颗粒感）。
        * **🚨 AI 生成**：往往像“塑料”或“蜡像”一样光滑，或者在头发等细节处出现奇怪的条纹状噪点，缺乏随机性。
    """)
    c1, c2 = st.columns(2)
    c1.image(ela_res, use_container_width=True, caption=f"ELA (Quality={ela_quality})")
    c2.image(noise_res, clamp=True, channels='GRAY', use_container_width=True, caption=f"Noise Residual (r={blur_radius})")

# 4. 物理
with tab_physics:
    st.info("""
    **📊 判读指南：**
    * **🌈 光照梯度 (Illumination)**：**颜色代表光照方向**。在平滑的曲面（如人脸、球体）上，颜色应该**平滑过渡**。如果颜色杂乱无章（五颜六色），说明 AI 搞乱了光影逻辑。
    * **🟣 色差 (Chromatic Aberration)**：真实镜头在画面边缘的高光交界处会有**紫边/绿边**。AI 生成图往往要么**完全没有色差**（全黑，过于完美），要么全图随机乱飞。
    * **🔥 饱和度 (Saturation)**：检查阴影区域。物理世界的阴影应该是低饱和度的。如果你在黑影里看到了**高饱和度的红色/蓝色杂斑**，这是 Diffusion 模型的典型缺陷。
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

# 5. 显微镜
with tab_micro:
    st.markdown("#### 🔬 交互式显微镜")
    st.caption("👈 **操作方法**：在左侧【导航图】上点击任意位置，右侧会显示该区域的高清原图细节。")

    col_nav, col_zoom = st.columns([1, 2])

    with col_nav:
        st.subheader("1. 导航 (点击定位)")
        
        # 视窗大小 (View Size)
        crop_size = st.slider(
            "🔎 视窗范围 (像素)",
            min_value=50, max_value=1200, value=400, step=50,
            help="数值越大，视野越广（倍率越低）；数值越小，倍率越高。"
        )

        # === 修复核心：使用 thumbnail 缩略图模式 ===
        # 不再强制 resize 到固定宽度，而是限制在 350x350 的框内
        # 这样无论是长图还是宽图，都能完整显示，不会被截断
        
        # 1. 复制一个用于显示的副本
        pil_nav = pil_img.copy()
        
        # 2. 生成缩略图 (原地修改 pil_nav，保持比例)
        # 350 是侧边栏/分栏通常的安全宽度
        pil_nav.thumbnail((350, 350), Image.Resampling.LANCZOS)
        
        # 3. 获取缩略图的实际尺寸
        nav_w, nav_h = pil_nav.size
        
        # 4. 显示导航图 (注意：这里不要传 width 参数，让组件自己适应图片)
        coords = streamlit_image_coordinates(
            pil_nav,
            key="zoom_click"
        )

        # 5. 坐标映射逻辑 (根据缩略图和原图的比例反算)
        if coords:
            # 算出缩放比例
            scale_x = w_orig / nav_w
            scale_y = h_orig / nav_h
            
            # 反算回原图坐标
            center_x = int(coords['x'] * scale_x)
            center_y = int(coords['y'] * scale_y)
        else:
            center_x = w_orig // 2
            center_y = h_orig // 2

        st.info(f"原图坐标: ({center_x}, {center_y})")

    with col_zoom:
        st.subheader("2. 细节 (高清原图)")

        # 边界保护 (防止超出图片范围)
        half_size = crop_size // 2
        x0 = max(0, center_x - half_size)
        y0 = max(0, center_y - half_size)
        x1 = min(w_orig, center_x + half_size)
        y1 = min(h_orig, center_y + half_size)

        # 裁剪原图
        crop_img = pil_img.crop((x0, y0, x1, y1))
        
        # 显示裁剪图 (使用 use_container_width 撑满右侧区域)
        st.image(crop_img, use_container_width=True)

        # 局部 FFT 分析
        with st.expander("查看该区域的 FFT 特征 (排除背景干扰)", expanded=True):
            # 实时计算
            crop_np = img_to_float01(crop_img)
            crop_gray_small = rgb_to_gray(crop_np)
            crop_fft = compute_fft(crop_gray_small)
            
            # 显示 FFT (use_container_width=True 保证图不会忽大忽小)
            st.image(
                crop_fft,
                clamp=True,
                caption="局部 FFT 频谱",
                use_container_width=True
            )
