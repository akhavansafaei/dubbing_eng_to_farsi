#!/bin/bash
# Quick Installation Script for Video Dubbing Pipeline

echo "🚀 Video Dubbing Pipeline - Quick Setup"
echo "========================================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    OS="unknown"
fi

echo "Detected OS: $OS"

# Linux Installation
if [ "$OS" = "linux" ]; then
    echo "📋 Installing system dependencies for Linux..."
    
    # Update package manager
    sudo apt-get update
    
    # Install system dependencies
    sudo apt-get install -y ffmpeg git python3-pip
    
    # Optional: Install CUDA dependencies (for GPU acceleration)
    read -p "Install CUDA dependencies for GPU acceleration? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get install -y libcudnn8 libcudnn8-dev
    fi
fi

# Windows Installation (via PowerShell commands)
if [ "$OS" = "windows" ]; then
    echo "📋 Windows Setup Instructions:"
    echo "1. Install FFmpeg: https://ffmpeg.org/download.html"
    echo "2. Install Git: https://git-scm.com/download/win"
    echo "3. Add both to your system PATH"
    echo "4. Install Python 3.8+ from python.org"
    echo ""
    echo "After installing the above, continue with Python packages below."
fi

echo ""
echo "🐍 Installing Python dependencies..."

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
python -m pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing core dependencies..."
python -m pip install "numpy<2"
python -m pip install openai-whisper
python -m pip install sherpa-onnx soundfile
python -m pip install moviepy pysrt pydub
python -m pip install ffmpeg-python yt-dlp
python -m pip install "pyannote.audio==3.1.1"
python -m pip install google-generativeai
python -m pip install tqdm

echo ""
echo "🎤 Downloading Persian TTS model..."

# Download the Persian TTS model
if [ -d "./vits-piper-fa_IR-amir-medium" ]; then
    echo "Model directory already exists, skipping download..."
else
    git clone https://huggingface.co/csukuangfj/vits-piper-fa_IR-amir-medium
    echo "✅ Persian TTS model downloaded"
fi

echo ""
echo "🔑 Creating sample keys.txt file..."

# Create sample keys file if it doesn't exist
if [ ! -f "keys.txt" ]; then
    cat > keys.txt << EOF
# Add your Google Gemini API keys here (one per line)
# You can get API keys from: https://ai.google.dev/
# Example:
# AIzaSyABC123def456GHI789jkl012MNO345pqr678
# AIzaSyDEF456ghi789JKL012mno345PQR678stu901
EOF
    echo "✅ Created sample keys.txt"
    echo "⚠️  Please edit keys.txt and add your actual Google Gemini API keys"
else
    echo "✅ keys.txt already exists"
fi

echo ""
echo "🎉 Setup completed!"
echo "Don't forget to:"
echo "1. Add your Google Gemini API keys to keys.txt"
echo "2. Test the installation by running: python video_dubber.py --help"
echo ""
echo "Usage: python video_dubber.py input_video.mp4 output_video.mp4"