# Image Feature Panel

这是一个轻量级的图像特征可视化工具，基于 Streamlit 开发。它通过提取图像的频域、信号层和物理层特征，辅助用户判断图片是相机直出还是 AI 生成/后期合成。

  🔗 在线演示 (Live Demo): [image-feature-panel.streamlit.app](https://image-feature-panel.streamlit.app)

![图层 5](https://github.com/user-attachments/assets/7fb651ba-7681-44b1-9297-d05132b16a4f)
<img width="3000" height="2250" alt="forensic_report-4" src="https://github.com/user-attachments/assets/47f23383-6511-423e-9dfe-076d2fffec21" />

## 📦 功能列表

* **元数据检测**: 自动提取 EXIF 信息及隐藏的 AI 生成参数 (支持 Stable Diffusion, Midjourney, NovelAI 等)。
* **频域分析 (FFT)**: 2D 频谱图与径向能量衰减曲线，用于检测网格效应和异常高频信号。

* **ELA (误差水平分析)**: 检测重压缩差异和拼接痕迹。
* **噪声残差**: 分离图像内容，查看底层传感器噪声分布。


* **光照梯度**: 可视化光照方向，检查光影逻辑一致性。
* **色差**: 检测镜头边缘的紫边/色散现象。
* **饱和度**: 阴影区域的饱和度热力图。


* **交互式显微镜**: 支持局部点击放大，实时查看特定区域的 FFT 特征。
* **报告导出**: 一键生成包含所有特征图的 PNG 面板。

## 🚀 快速开始

### 1. 安装依赖

确保已安装 Python 3.8+，然后运行：

```bash
pip install -r requirements.txt

```

### 2. 运行应用

```bash
streamlit run app.py

```

### 3. 使用

浏览器自动打开 `http://localhost:8501` 后，在左侧侧边栏上传图片即可开始分析。

![图层 4](https://github.com/user-attachments/assets/7e3e9dc9-2e01-4f6f-a342-daa60c205d52)
![图层 3](https://github.com/user-attachments/assets/a9eaa61b-1621-42b6-a85f-e82909d9b0f2)
![图层 2](https://github.com/user-attachments/assets/742beadf-1ba5-4667-be78-df9268af2aa4)
![图层 1](https://github.com/user-attachments/assets/38b60d42-fc9f-4894-a3fb-80594a1486b9)

## 📊 特征参考简表

| 特征维度 | 📸 真实照片倾向 | 🤖 AI生成/伪造倾向 |
| --- | --- | --- |
| **FFT 频谱** | 能量从中心向四周平滑衰减 | 出现孤立亮斑（非中心）、规则网格或棋盘纹理 |
| **噪声残差** | 均匀的颗粒感（光子噪声） | 表面过于光滑（蜡像感），或细节处有条纹状噪点 |
| **ELA** | 噪点分布相对均匀 | 局部区域（如人脸）与其他区域噪点差异巨大 |
| **光照方向** | 曲面光照颜色平滑过渡 | 光照方向杂乱，颜色呈随机噪点状 |
| **色差** | 边缘高光处有自然色散 | 完全无色差（全黑）或全图随机色斑 |
| **元数据** | 含光圈、快门、ISO 等相机信息 | 包含 `parameters`, `steps` 等生成信息或无 EXIF |

## 🛠️ 技术栈

* **UI**: Streamlit
* **计算**: Numpy, Scipy
* **图像处理**: Pillow (PIL), OpenCV (算法逻辑)
* **绘图**: Matplotlib

## 📄 License

MIT License
