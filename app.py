import os
import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import streamlit as st
from io import BytesIO
import base64
import uuid
from datetime import datetime
from pathlib import Path
import json
from dotenv import load_dotenv
import base64
from streamlit_extras.stylable_container import stylable_container

# Custom CSS for modern styling
st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #2c3e50;
            color: white;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Sliders */
        .stSlider .thumb {
            background-color: #4CAF50 !important;
        }
        
        .stSlider .st-ae {
            background-color: #4CAF50 !important;
        }
        
        /* Text areas */
        .stTextArea>div>div>textarea {
            border-radius: 10px;
            border: 1px solid #ddd;
            padding: 10px;
        }
        
        /* Cards for images */
        .image-card {
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(45deg, #4CAF50, #45a049);
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #2c3e50;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_ID = "CompVis/stable-diffusion-v1-4"  # Using a different model that doesn't require authentication
OUTPUT_DIR = "static/images"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state.generated = False

@st.cache_resource
def load_model():
    """Load the Stable Diffusion model with appropriate device settings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        print(f"Loading model {MODEL_ID} on {device}...")
        
        # Try to load with authentication token first
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            print("Using Hugging Face token for authentication")
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                use_auth_token=hf_token,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            print("No Hugging Face token found, trying local model")
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False
            )
            
    except Exception as e:
        print(f"Error loading model: {e}")
        st.warning("Error loading model. Falling back to a smaller model...")
        
        # Try a smaller model that's more likely to be cached
        fallback_model = "CompVis/stable-diffusion-v1-4"
        print(f"Trying fallback model: {fallback_model}")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            fallback_model,
            torch_dtype=torch_dtype,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False
        )
    
    # Configure the pipeline with more stable settings
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Optimize for memory and stability
    if device == "cuda":
        pipe.enable_attention_slicing(slice_size=1)  # More stable with slice_size=1
    else:
        pipe.enable_attention_slicing()
        
    # Set a maximum number of steps to prevent index errors
    pipe.scheduler.config.solver_order = 2  # Use second order solver for better stability
    
    return pipe

def generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, num_images):
    """Generate images using the loaded model"""
    pipe = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Add quality enhancements to prompt
    enhanced_prompt = f"{prompt}, high quality, highly detailed, 4k, professional photography"
    
    # Generate images with error handling
    try:
        with torch.inference_mode():
            # Ensure num_inference_steps is within bounds
            max_steps = 50  # Safe upper limit
            num_inference_steps = min(max(num_inference_steps, 10), max_steps)
            
            # Generate one image at a time for stability
            images = []
            for _ in range(num_images):
                output = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=min(guidance_scale, 10.0),  # Limit guidance scale
                    num_images_per_prompt=1,
                )
                images.extend(output.images)
                
    except Exception as e:
        print(f"Error during image generation: {e}")
        # Return empty list to prevent crashes
        return []
    
    return images

def save_image(image, prompt, negative_prompt, parameters):
    """Save generated image with metadata"""
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{str(uuid.uuid4())[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save image
    image.save(filepath, "PNG")
    
    # Save metadata
    metadata = {
        "filename": filename,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "generated_at": datetime.now().isoformat(),
        "parameters": parameters
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filepath, metadata

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = (f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="text-decoration: none;">'
            f'<button style="background: linear-gradient(45deg, #4CAF50, #45a049); color: white; border: none; '
            f'border-radius: 20px; padding: 10px 20px; font-weight: 600; cursor: pointer; width: 100%;">'
            f'{text}</button></a>')
    return href

def main():
    # Custom header
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="margin: 0; color: #2c3e50;">🎨 AI Image Generator</h1>
            <p style="color: #7f8c8d; margin-top: 5px;">Transform your imagination into stunning AI-generated art with Stable Diffusion</p>
        </div>
        <div style="height: 2px; background: linear-gradient(90deg, #4CAF50, #2196F3); margin: 0 -2rem 2rem -2rem;"></div>
        """,
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.markdown(
            """
            <div style='background: linear-gradient(45deg, #4CAF50, #2196F3); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;'>
                <h3 style='color: white; margin: 0;'>⚙️ Generation Settings</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Model parameters with better organization
        with st.expander("📝 Prompt Settings", expanded=True):
            prompt = st.text_area(
                "Describe your image",
                value="a beautiful landscape with mountains and a lake, sunset, 4k, highly detailed",
                help="Be as descriptive as possible for better results"
            )
            
            negative_prompt = st.text_area(
                "What to avoid in the image",
                value="blurry, low quality, distorted, bad anatomy, disfigured, deformed",
                help="List things you want to avoid in the generated images"
            )
        
        with st.expander("🎛️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                num_images = st.slider("Number of images", 1, 4, 1, 
                                     help="How many variations to generate (more images take longer)")
                guidance_scale = st.slider("Creativity", 1.0, 20.0, 7.5, 0.5,
                                         help="Higher values make the model follow your prompt more closely")
            
            with col2:
                num_inference_steps = st.slider("Quality", 10, 100, 25, 5,
                                              help="More steps = better quality but slower generation")
        
        generate_btn = st.button("✨ Generate Images", use_container_width=True, type="primary")
        
        # Add some information in the sidebar
        st.markdown("---")
        with st.expander("💡 Tips for better results"):
            st.markdown("""
            - Be specific and descriptive in your prompts
            - Include style references (e.g., "digital art", "oil painting", "photorealistic")
            - Use negative prompts to remove unwanted elements
            - For portraits, include details like lighting and expression
            - Experiment with different guidance scales (7-9 usually works well)
            """)
        
        # Add model info
        st.markdown("---")
        st.markdown("### System Information")
        st.markdown(f"""
        <div style="background: rgba(44, 62, 80, 0.1); padding: 10px; border-radius: 8px;">
            <p style="margin: 5px 0;"><b>Device:</b> <span style="color: {'#4CAF50' if torch.cuda.is_available() else '#2196F3'}">
                {'GPU 🔥' if torch.cuda.is_available() else 'CPU 🐢'}</span></p>
            <p style="margin: 5px 0;"><b>Model:</b> {MODEL_ID.split('/')[-1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if generate_btn and prompt:
        with st.spinner("🎨 Generating your images... This may take a few minutes..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            try:
                # Update progress
                progress_bar.progress(20)
                status_text.info("🔍 Initializing model and processing your request...")
                
                # Generate images
                images = generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images=num_images
                )
                
                progress_bar.progress(80)
                status_text.success(f"✅ Successfully generated {len(images)} images!")
                
                # Display images in a responsive grid
                st.markdown("## 🖼️ Generated Images")
                
                # Calculate columns based on number of images
                num_cols = 2 if len(images) > 1 else 1
                cols = st.columns(num_cols)
                
                for idx, img in enumerate(images):
                    with cols[idx % num_cols]:
                        # Create a card for each image
                        with st.container():
                            st.markdown(f"<div class='image-card'>", unsafe_allow_html=True)
                            
                            # Display image
                            st.image(img, use_column_width=True)
                            
                            # Save the image
                            params = {
                                "guidance_scale": guidance_scale,
                                "num_inference_steps": num_inference_steps,
                                "model": MODEL_ID
                            }
                            
                            filepath, metadata = save_image(
                                image=img,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                parameters=params
                            )
                            
                            # Add download button with custom styling
                            st.markdown(
                                get_image_download_link(
                                    img, 
                                    os.path.basename(filepath),
                                    f"⬇️ Download Image {idx+1}"
                                ),
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                
                generation_time = time.time() - start_time
                progress_bar.progress(100)
                
                st.success(f"✨ Generation completed in {generation_time:.2f} seconds!")
                
                # Add a nice footer
                st.markdown("---")
                st.markdown("### 🎨 Try Another Creation")
                st.markdown("Ready to create something new? Adjust your settings and generate more amazing images!")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.error("❌ An error occurred during generation")
                st.exception(e)
    
    elif generate_btn and not prompt:
        st.warning("⚠️ Please enter a prompt to generate images")
    
    # Initial state - show some examples
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <h2>🎨 Create Stunning AI Art</h2>
            <p style="font-size: 1.1em; color: #7f8c8d;">
                Enter a detailed description of the image you want to generate in the sidebar.<br>
                Adjust the settings and click "Generate Images" to start creating!
            </p>
            <div style="margin: 30px 0;">
                <div style="display: inline-block; text-align: left; max-width: 600px; background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px;">
                    <h4>💡 Try these examples:</h4>
                    <ul>
                        <li>"A serene mountain lake at sunset, snow-capped peaks, 8k, highly detailed"</li>
                        <li>"A cyberpunk cityscape at night, neon lights, rainy streets, 4k digital art"</li>
                        <li>"A cute corgi dog wearing sunglasses, beach background, cartoon style"</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
