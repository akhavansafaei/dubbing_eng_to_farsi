#!/usr/bin/env python3
"""
Video Dubbing Setup Script
Automatically installs dependencies and downloads required models for both Windows and Linux
"""

import os
import sys
import subprocess
import platform
import urllib.request
import shutil
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split() if not shell else cmd
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    print(f"Detected system: {system}")
    
    if system == "linux":
        print("Installing Linux system dependencies...")
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg",
            "sudo apt-get install -y libcudnn8 libcudnn8-dev",
            "sudo apt-get install -y git"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            if not run_command(cmd, shell=True):
                print(f"Warning: Failed to run {cmd}")
    
    elif system == "windows":
        print("Windows detected. Please ensure you have:")
        print("1. FFmpeg installed and added to PATH")
        print("2. Git installed")
        print("3. Visual Studio Build Tools (for some Python packages)")
        print("\nYou can download FFmpeg from: https://ffmpeg.org/download.html")
        print("You can download Git from: https://git-scm.com/download/win")
        
        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            print("⚠️  WARNING: FFmpeg not found in PATH!")
            print("Please install FFmpeg and add it to your system PATH")
        else:
            print("✅ FFmpeg found in PATH")
            
        if not shutil.which("git"):
            print("⚠️  WARNING: Git not found in PATH!")
            print("Please install Git")
        else:
            print("✅ Git found in PATH")
    
    else:
        print(f"Unsupported system: {system}")
        print("This script supports Windows and Linux only")
        return False
    
    return True

def install_python_dependencies():
    """Install Python dependencies via pip"""
    print("Installing Python dependencies...")
    
    # Core dependencies
    dependencies = [
        "numpy<2",  # Specific version constraint
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "openai-whisper",
        "sherpa-onnx",
        "soundfile",
        "moviepy",
        "pysrt",
        "pydub",
        "yt-dlp",
        "ffmpeg-python",
        "pyannote.audio==3.1.1",
        "google-generativeai",
        "tqdm"
    ]
    
    # Install PyTorch with CUDA support first
    print("Installing PyTorch with CUDA support...")
    torch_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch==2.1.0", "torchaudio==2.1.0", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    if not run_command(torch_cmd):
        print("Failed to install PyTorch with CUDA. Trying CPU version...")
        cpu_cmd = [sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchaudio==2.1.0"]
        if not run_command(cpu_cmd):
            print("Failed to install PyTorch. Please install manually.")
            return False
    
    # Install other dependencies
    for dep in dependencies:
        if "torch" in dep:  # Skip torch packages as they're already installed
            continue
            
        print(f"Installing {dep}...")
        cmd = [sys.executable, "-m", "pip", "install", dep]
        if not run_command(cmd):
            print(f"Failed to install {dep}")
            return False
    
    print("✅ All Python dependencies installed successfully")
    return True

def download_amir_model():
    """Download the Persian TTS model"""
    model_dir = "./vits-piper-fa_IR-amir-medium"
    
    if os.path.exists(model_dir):
        print(f"Model directory {model_dir} already exists. Checking contents...")
        required_files = [
            "fa_IR-amir-medium.onnx",
            "tokens.txt",
            "espeak-ng-data"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                missing_files.append(file)
        
        if not missing_files:
            print("✅ Persian TTS model already downloaded and complete")
            return True
        else:
            print(f"Missing files: {missing_files}")
            print("Re-downloading model...")
            shutil.rmtree(model_dir)
    
    print("Downloading Persian TTS model...")
    
    # Try using git first
    git_cmd = ["git", "clone", "https://huggingface.co/csukuangfj/vits-piper-fa_IR-amir-medium"]
    
    if run_command(git_cmd):
        print("✅ Model downloaded successfully using git")
        return True
    
    # If git fails, try manual download
    print("Git clone failed. Attempting manual download...")
    
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Download required files manually
        base_url = "https://huggingface.co/csukuangfj/vits-piper-fa_IR-amir-medium/resolve/main/"
        files_to_download = [
            "fa_IR-amir-medium.onnx",
            "tokens.txt",
            "README.md"
        ]
        
        for filename in files_to_download:
            url = base_url + filename
            local_path = os.path.join(model_dir, filename)
            
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                return False
        
        # Create espeak-ng-data directory (may be empty but needed)
        espeak_dir = os.path.join(model_dir, "espeak-ng-data")
        os.makedirs(espeak_dir, exist_ok=True)
        
        print("✅ Model downloaded successfully via manual download")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def create_sample_keys_file():
    """Create a sample keys.txt file"""
    keys_file = "keys.txt"
    
    if os.path.exists(keys_file):
        print(f"✅ {keys_file} already exists")
        return True
    
    sample_content = """AIzaSyCr7FaCu2kV_sGGadx9QjP_vlfjN0Nu8b4
AIzaSyCNhF0n1xkqj0xSIJaH8SqwwvJmws-zUjs
AIzaSyAPx47e7-dhHxMgfSHW5Kcu8zbIixl6RG4
AIzaSyAb-rl_aXXXYDM_RbbkzWPu7LWmNYiWyuE
AIzaSyC7xnpqOYd6lmFVGX5stPpBk5VnuuVr2X4
AIzaSyCkIy2EkvSV-GcGa_uo2KxQ_4mKhOerEN0
AIzaSyDsAFujZKFKtCk1TXWIcfTBdCbo91-VoRc
AIzaSyC4ggTf98D-52XJfew5Mjw6ocqBsRtyaxY
AIzaSyA594JS4rL3jw0dkr7dkpghfLG6Yzcaka0
AIzaSyAiirJIqTLoy7Vkj0jBwl2dX-XZ1jVcDcY
AIzaSyD6HGo0JogjzewwwvtEifCCcTk0mM8iBYY
AIzaSyAMZOsK33UcSXyzELhZXfImf2lPetEUKqQ
"""
    
    try:
        with open(keys_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print(f"✅ Created sample {keys_file}")
        print("⚠️  Please edit keys.txt and add your actual Google Gemini API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create {keys_file}: {e}")
        return False

def verify_installation():
    """Verify that all components are properly installed"""
    print("\n" + "="*50)
    print("VERIFYING INSTALLATION")
    print("="*50)
    
    # Check Python packages
    packages_to_check = [
        "whisper", "google.generativeai", "sherpa_onnx", 
        "soundfile", "moviepy", "pysrt", "torch", "torchaudio"
    ]
    
    missing_packages = []
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    # Check system tools
    system_tools = ["ffmpeg"]
    if platform.system().lower() != "windows":
        system_tools.append("git")
    
    missing_tools = []
    for tool in system_tools:
        if shutil.which(tool):
            print(f"✅ {tool}")
        else:
            print(f"❌ {tool}")
            missing_tools.append(tool)
    
    # Check model files
    model_dir = "./vits-piper-fa_IR-amir-medium"
    required_files = [
        "fa_IR-amir-medium.onnx",
        "tokens.txt",
        "espeak-ng-data"
    ]
    
    missing_model_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"✅ Model file: {file}")
        else:
            print(f"❌ Model file: {file}")
            missing_model_files.append(file)
    
    # Check keys file
    if os.path.exists("keys.txt"):
        with open("keys.txt", 'r') as f:
            content = f.read()
            if "AIza" in content and not content.strip().startswith("#"):
                print("✅ keys.txt (with API keys)")
            else:
                print("⚠️  keys.txt (needs API keys)")
    else:
        print("❌ keys.txt")
    
    # Summary
    all_good = (not missing_packages and not missing_tools and not missing_model_files)
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 INSTALLATION COMPLETE!")
        print("All dependencies and models are ready.")
        if "keys.txt (needs API keys)" in []:
            print("Don't forget to add your Google Gemini API keys to keys.txt")
    else:
        print("⚠️  INSTALLATION INCOMPLETE")
        if missing_packages:
            print(f"Missing Python packages: {', '.join(missing_packages)}")
        if missing_tools:
            print(f"Missing system tools: {', '.join(missing_tools)}")
        if missing_model_files:
            print(f"Missing model files: {', '.join(missing_model_files)}")
    
    print("="*50)
    return all_good

def main():
    """Main setup function"""
    print("🚀 Video Dubbing Setup Script")
    print("Setting up dependencies for English to Persian video dubbing")
    print("="*60)
    
    # Step 1: Install system dependencies
    print("\n📋 Step 1: Installing system dependencies...")
    if not install_system_dependencies():
        print("❌ Failed to install system dependencies")
        return False
    
    # Step 2: Install Python dependencies
    print("\n🐍 Step 2: Installing Python dependencies...")
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        return False
    
    # Step 3: Download Persian TTS model
    print("\n🎤 Step 3: Downloading Persian TTS model...")
    if not download_amir_model():
        print("❌ Failed to download Persian TTS model")
        return False
    
    # Step 4: Create sample keys file
    print("\n🔑 Step 4: Creating sample keys file...")
    create_sample_keys_file()
    
    # Step 5: Verify installation
    print("\n🔍 Step 5: Verifying installation...")
    success = verify_installation()
    
    if success:
        print("\n🎬 Ready to use! Run the video dubbing script with:")
        print("python video_dubber.py input_video.mp4 output_video.mp4")
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")
    
    return success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)