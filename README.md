# AI Image Generator

A text-to-image generation system built with Stable Diffusion and Streamlit that converts text prompts into high-quality images.

## Features

- Generate images from text prompts using state-of-the-art Stable Diffusion models
- Adjustable parameters for fine-tuning image generation
- Supports both GPU and CPU (with automatic fallback)
- Simple and intuitive web interface
- Save and download generated images with metadata
- Responsive design that works on desktop and mobile

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA support for faster generation

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-image-generator.git
   cd ai-image-generator
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`

## 🖥️ Usage

1. Enter a descriptive prompt in the text area (e.g., "A beautiful landscape with mountains and a lake at sunset")
2. (Optional) Add negative prompts to exclude unwanted elements
3. Adjust the generation parameters:
   - Number of images to generate (1-4)
   - Guidance scale (controls how closely the image follows the prompt)
   - Number of inference steps (more steps = higher quality but slower)
4. Click "Generate Images" and wait for the magic to happen!
5. Download your favorite generated images using the download buttons

## 🛠️ Technical Details

### Model

This application uses the following models:
- `runwayml/stable-diffusion-v1-5` (default for CPU)
- `stabilityai/stable-diffusion-2-1` (used when GPU is available)

### Hardware Requirements

- **Minimum (CPU):**
  - 4+ GB RAM
  - 10+ GB free disk space (for model weights)
  - Modern x86-64 CPU with AVX2 support

- **Recommended (GPU):**
  - NVIDIA GPU with 8GB+ VRAM
  - CUDA 11.7 or higher
  - cuDNN 8.5 or higher

### Performance

Generation times vary based on hardware:
- GPU (NVIDIA RTX 3080): ~5-10 seconds per image
- CPU (modern 8-core): ~2-5 minutes per image

## 📂 Project Structure

```
ai-image-generator/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── static/               # Static files (images, styles)
│   └── images/           # Generated images and metadata
└── utils/                # Utility functions (if any)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stable Diffusion](https://stability.ai/stable-diffusion) by Stability AI and Runway ML
- [🤗 Diffusers](https://github.com/huggingface/diffusers) library
- [Streamlit](https://streamlit.io/) for the web interface

## ⚠️ Disclaimer

This tool is intended for responsible and ethical use. Please be mindful of the content you generate and ensure it complies with all applicable laws and ethical guidelines. The developers are not responsible for any misuse of this tool.
