@echo off
REM Windows Installation Script for Video Dubbing Pipeline
echo 🚀 Video Dubbing Pipeline - Windows Setup
echo =========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ from python.org
    echo After installing Python, re-run this script.
    pause
    exit /b 1
)

REM Check if FFmpeg is installed
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ❌ FFmpeg not found in PATH!
    echo Please download FFmpeg from https://ffmpeg.org/download.html
    echo Extract it and add the bin folder to your system PATH
    echo After installing FFmpeg, re-run this script.
    pause
    exit /b 1
)

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git not found in PATH!
    echo Please install Git from https://git-scm.com/download/win
    echo After installing Git, re-run this script.
    pause
    exit /b 1
)

echo ✅ Python, FFmpeg, and Git are available
echo.

echo 🐍 Installing Python dependencies...
echo.

REM Upgrade pip
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
python -m pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

REM Install core dependencies
echo Installing core dependencies...
python -m pip install "numpy<2"
python -m pip install openai-whisper
python -m pip install sherpa-onnx soundfile
python -m pip install moviepy pysrt pydub
python -m pip install ffmpeg-python yt-dlp
python -m pip install "pyannote.audio==3.1.1"
python -m pip install google-generativeai
python -m pip install tqdm

echo.
echo 🎤 Downloading Persian TTS model...

REM Download the Persian TTS model
if exist "vits-piper-fa_IR-amir-medium" (
    echo Model directory already exists, skipping download...
) else (
    git clone https://huggingface.co/csukuangfj/vits-piper-fa_IR-amir-medium
    echo ✅ Persian TTS model downloaded
)

echo.
echo 🔑 Creating sample keys.txt file...

REM Create sample keys file if it doesn't exist
if not exist "keys.txt" (
    echo # Add your Google Gemini API keys here ^(one per line^) > keys.txt
    echo # You can get API keys from: https://ai.google.dev/ >> keys.txt
    echo # Example: >> keys.txt
    echo # AIzaSyABC123def456GHI789jkl012MNO345pqr678 >> keys.txt
    echo # AIzaSyDEF456ghi789JKL012mno345PQR678stu901 >> keys.txt
    echo ✅ Created sample keys.txt
    echo ⚠️  Please edit keys.txt and add your actual Google Gemini API keys
) else (
    echo ✅ keys.txt already exists
)

echo.
echo 🎉 Setup completed!
echo Don't forget to:
echo 1. Add your Google Gemini API keys to keys.txt
echo 2. Test the installation by running: python video_dubber.py --help
echo.
echo Usage: python video_dubber.py input_video.mp4 output_video.mp4
echo.
pause