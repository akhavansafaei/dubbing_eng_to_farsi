# Video Dubbing: English to Farsi (Persian)

🎬 **AI-Powered Video Dubbing Pipeline** — Automatically dub English videos into Persian (Farsi) using state-of-the-art AI technologies for transcription, translation, and text-to-speech synthesis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Version History](#version-history)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This repository contains a complete AI-powered video dubbing pipeline that transforms English-language videos into Persian (Farsi) dubbed versions. The system handles the entire workflow from speech recognition to final video assembly, providing professional-quality localization.

**Complete Workflow:**
```
English Video → Transcription → Translation → Refinement → TTS → Timing Sync → Dubbed Video
```

### Key Capabilities

- **Automatic Speech Recognition**: Extracts English speech using OpenAI Whisper
- **AI Translation**: Translates content to Persian using Google Gemini AI
- **Translation Refinement**: Improves fluency and cultural appropriateness
- **Persian TTS**: Generates natural-sounding Persian audio using Sherpa-ONNX
- **Intelligent Timing**: Synchronizes audio with video timeline
- **Video Assembly**: Creates final dubbed video with FFmpeg

---

## ✨ Features

### Core Features

- ✅ **Full Pipeline Automation**: Complete end-to-end processing
- ✅ **High-Quality Transcription**: Whisper-based speech recognition
- ✅ **Context-Aware Translation**: Uses conversation context for coherent translations
- ✅ **Natural TTS**: Persian voice synthesis with timing adaptation
- ✅ **Batch Processing**: Process multiple videos sequentially
- ✅ **Progress Tracking**: Real-time progress indicators for each step
- ✅ **Error Handling**: Robust error recovery and API quota management
- ✅ **Modular Design**: Use individual components or full pipeline

### Advanced Features

- 🔄 **API Quota Management**: Automatic rotation between multiple API keys
- 📊 **Detailed Reporting**: CSV reports for timing adjustments
- 🎛️ **Configurable Settings**: Customize TTS voice, speed, and quality
- 🧹 **Cleanup Options**: Manage temporary files and intermediate outputs
- 📁 **SRT Export**: Save subtitles in standard SRT format
- 🎚️ **Audio Mixing**: Preserve background audio with configurable levels

---

## 🏗️ Architecture

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Video Dubbing Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

Step 1: TRANSCRIPTION                    Step 2: TRANSLATION
┌──────────────────────┐                ┌──────────────────────┐
│   English Video      │                │  English Subtitles   │
│   ┌──────────┐       │   Whisper      │  ┌──────────┐        │   Gemini AI
│   │  [MP4]   │───────┼───────────────▶│  │  [SRT]   │────────┼────────────┐
│   └──────────┘       │   (base.en)    │  └──────────┘        │            │
└──────────────────────┘                └──────────────────────┘            │
                                                                              │
Step 3: REFINEMENT                       Step 4: TTS GENERATION              │
┌──────────────────────┐                ┌──────────────────────┐            │
│  Persian Subtitles   │   Gemini AI    │  Refined Persian     │  Sherpa    │
│  ┌──────────┐        │◀───────────────│  ┌──────────┐        │◀───────────┘
│  │  [SRT]   │────────┼───────────────▶│  │  [SRT]   │────────┼────────────┐
│  └──────────┘        │  (refinement)  │  └──────────┘        │   ONNX     │
└──────────────────────┘                └──────────────────────┘            │
                                                                              │
Step 5: TIMING SYNC                      Step 6: VIDEO ASSEMBLY              │
┌──────────────────────┐                ┌──────────────────────┐            │
│  Persian Audio       │   Duration     │  Adjusted Timing     │  FFmpeg    │
│  ┌──────────┐        │◀───────────────│  ┌──────────┐        │◀───────────┘
│  │  [WAV]   │────────┼───────────────▶│  │  [SRT]   │────────┼────────────┐
│  └──────────┘        │   Analysis     │  └──────────┘        │            │
└──────────────────────┘                └──────────────────────┘            │
                                                                              │
                                        ┌──────────────────────┐            │
                                        │  Final Dubbed Video  │            │
                                        │  ┌──────────┐        │◀───────────┘
                                        │  │  [MP4]   │        │
                                        │  └──────────┘        │
                                        └──────────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Transcription** | OpenAI Whisper / Faster-Whisper | Convert speech to text (English) |
| **Translation** | Google Gemini AI (gemini-1.5-flash) | Translate English to Persian |
| **Refinement** | Google Gemini AI | Improve translation quality |
| **TTS Engine** | Sherpa-ONNX (Piper VITS) | Generate Persian speech |
| **Video Processing** | FFmpeg | Audio/video manipulation |
| **Subtitle Processing** | pysrt | SRT file handling |

---

## 📁 Project Structure

```
dubbing_eng_to_farsi/
│
├── final/                          # ⭐ Production Version (Recommended)
│   ├── final_v3.py                 # Main unified dubbing script
│   ├── readme.md                   # Detailed usage documentation
│   ├── instruction.md              # Step-by-step examples
│   ├── requirements.txt            # Python dependencies
│   ├── setup.py                    # Setup script
│   └── install.sh/bat              # Installation helpers
│
├── 1/                              # Phase 1: Initial Implementation
│   └── phase 1/
│       ├── final_phase_one.py      # Phase 1 script (Gemini API)
│       ├── readme.md               # Phase 1 documentation
│       ├── instruction.md          # Phase 1 instructions
│       └── requirements.txt        # Phase 1 dependencies
│
├── 2/                              # Phase 2: Modular Components
│   └── py_files/content_py_files/content/
│       ├── transcribe.py           # Whisper transcription module
│       ├── text_translator.py      # Translation module
│       ├── srt_refiner.py          # Translation refinement
│       ├── srt_to_audio.py         # TTS generation
│       ├── srt_adjuster.py         # Timing adjustment
│       ├── video_assembler.py      # Video assembly
│       ├── audio_separator.py      # Audio extraction/separation
│       ├── fa_srt_gen.py           # Persian SRT generation
│       ├── video_audio_extractor.py # Audio/video extraction
│       └── youtube_downloader.py   # YouTube download utility
│
├── 1.5/                            # Version 1.5: Experimental
│   └── local.py                    # Local processing version
│
├── yd/                             # YouTube Downloader Utilities
│   ├── yd.py                       # YouTube downloader
│   └── test.py                     # Test scripts
│
├── final deliveries/               # Archived Deliveries
│   └── codes/
│       └── phase 1 (gemimi, free and limited api)/
│
├── dub (1).pdf                     # Project documentation
├── requirements_extracted.txt      # Consolidated requirements
└── README.md                       # This file
```

### Recommended Entry Points

- **For Production Use**: `final/final_v3.py` — Complete, tested pipeline
- **For Learning**: `2/py_files/` — Individual modular components
- **For Reference**: `1/phase 1/` — Initial implementation

---

## 🚀 Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.8 or higher
- FFmpeg installed and in PATH
- 4GB+ RAM
- 2GB+ free disk space
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/akhavansafaei/dubbing_eng_to_farsi.git
cd dubbing_eng_to_farsi

# 2. Navigate to production version
cd final

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (if not already installed)
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

### Configuration

```bash
# 1. Create API keys file
cat > keys.txt << EOF
your_google_api_key_1
your_google_api_key_2
your_google_api_key_3
EOF

# 2. Download Persian TTS model
# Extract to: ./vits-piper-fa_IR-amir-medium/
# Required files:
#   - fa_IR-amir-medium.onnx
#   - tokens.txt
#   - espeak-ng-data/ (directory)
```

### Basic Usage

```bash
# Dub a single video
python final_v3.py input_video.mp4 output_dubbed.mp4

# Example
python final_v3.py lecture.mp4 lecture_persian.mp4
```

---

## 💻 Installation

### Method 1: Automated Setup (Recommended)

```bash
# Navigate to final directory
cd final/

# Run setup script
# Linux/macOS:
chmod +x install.sh
./install.sh

# Windows:
install.bat
```

### Method 2: Manual Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# 2. Install core dependencies
pip install --upgrade pip
pip install numpy<2  # Important: numpy version constraint

# 3. Install PyTorch (with CUDA support if available)
# For CUDA 11.8:
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchaudio

# 4. Install remaining dependencies
pip install openai-whisper google-generativeai sherpa-onnx soundfile tqdm
pip install moviepy pysrt pydub ffmpeg-python yt-dlp
pip install pyannote.audio==3.1.1 transformers huggingface_hub

# 5. Optional: Faster Whisper (recommended for better performance)
pip install faster-whisper
```

### Dependency Overview

| Category | Packages |
|----------|----------|
| **Core AI** | `torch`, `transformers`, `google-generativeai` |
| **Speech Processing** | `openai-whisper`, `sherpa-onnx`, `soundfile`, `librosa` |
| **Video Processing** | `moviepy`, `ffmpeg-python`, `pysrt`, `pydub` |
| **Utilities** | `tqdm`, `numpy`, `pandas` |
| **Optional** | `faster-whisper`, `bitsandbytes`, `accelerate` |

---

## 📖 Usage

### Command Line Interface

```bash
# Basic usage
python final_v3.py <input_video> <output_video>

# Examples
python final_v3.py video.mp4 video_persian.mp4
python final_v3.py /path/to/input.mp4 /path/to/output.mp4
```

### Python API

```python
from final_v3 import VideoDubber

# Initialize dubber
dubber = VideoDubber("input.mp4", "output.mp4")

# Run full pipeline
success = dubber.run_full_pipeline()

# Or run individual steps
dubber.transcribe_video()           # Step 1: Transcription
dubber.translate_to_persian()       # Step 2: Translation
dubber.refine_persian_srt()         # Step 3: Refinement
dubber.generate_tts_audio()         # Step 4: TTS Generation
dubber.adjust_srt_timing()          # Step 5: Timing Adjustment
dubber.create_final_dubbed_video()  # Step 6: Video Assembly
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple videos

for video in *.mp4; do
    output="dubbed_${video}"
    python final_v3.py "$video" "$output"
done
```

### Using Modular Components (Phase 2)

```bash
# Navigate to modular components
cd 2/py_files/content_py_files/content/

# Run individual components
python transcribe.py video.mp4          # Transcription only
python text_translator.py input.srt     # Translation only
python srt_to_audio.py input.srt        # TTS generation only
```

---

## 🛠️ Technologies

### AI & Machine Learning

- **[OpenAI Whisper](https://github.com/openai/whisper)**: Speech recognition and transcription
  - Model: `base.en` (English-optimized)
  - Alternative: `faster-whisper` for improved performance

- **[Google Gemini AI](https://ai.google.dev/)**: Neural machine translation
  - Model: `gemini-1.5-flash`
  - Features: Context-aware translation with refinement

- **[Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)**: Text-to-speech synthesis
  - Model: `vits-piper-fa_IR-amir-medium`
  - Voice: Persian (Farsi) - Amir voice

### Video & Audio Processing

- **[FFmpeg](https://ffmpeg.org/)**: Video encoding, audio mixing, format conversion
- **[MoviePy](https://zulko.github.io/moviepy/)**: Video editing and manipulation
- **[Librosa](https://librosa.org/)**: Audio analysis and processing
- **[PyDub](https://github.com/jiaaro/pydub)**: Audio manipulation

### Supporting Libraries

- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[Transformers](https://huggingface.co/transformers/)**: Hugging Face model support
- **[pysrt](https://github.com/byroot/pysrt)**: SRT subtitle parsing
- **[tqdm](https://github.com/tqdm/tqdm)**: Progress bars

---

## 📚 Version History

### Version 3.0 (Current - `final/`)
**Status**: Production Ready ✅

- Complete unified pipeline in single script
- Improved error handling and API quota management
- Enhanced progress tracking
- Optimized performance
- Comprehensive documentation

**Use Case**: Production deployments, batch processing

### Phase 2 (`2/`)
**Status**: Modular Components

- Separated components for flexibility
- Individual scripts for each pipeline step
- YouTube downloader integration
- Audio separation capabilities

**Use Case**: Custom workflows, component reuse

### Phase 1 (`1/phase 1/`)
**Status**: Initial Implementation

- First working version using Gemini API
- Free tier API support
- Basic pipeline functionality

**Use Case**: Reference implementation, learning

### Version 1.5 (`1.5/`)
**Status**: Experimental

- Local processing experiments
- Alternative approaches

**Use Case**: Research and development

---

## ⚡ Performance

### Processing Times

| Video Length | Transcription | Translation | TTS | Total Time |
|--------------|---------------|-------------|-----|------------|
| 5 minutes    | ~1 min        | ~3-5 min    | ~2-3 min | 15-25 min |
| 30 minutes   | ~6 min        | ~20-30 min  | ~15-20 min | 1.5-2.5 hrs |
| 1 hour       | ~12 min       | ~40-60 min  | ~30-40 min | 3-4 hrs |

**Note**: Times vary based on hardware, API response times, and content complexity.

### Resource Requirements

| Resource | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| **CPU** | Dual-core | Quad-core | 8+ cores |
| **RAM** | 4GB | 8GB | 16GB+ |
| **Storage** | 2GB free | 10GB free | 50GB+ SSD |
| **GPU** | None | NVIDIA GPU | RTX 3060+ |
| **Network** | 1 Mbps | 10 Mbps | 100 Mbps |

### Optimization Tips

1. **Use `faster-whisper`** instead of `openai-whisper` for 2-4x speed improvement
2. **Multiple API keys** prevent quota delays during translation
3. **SSD storage** significantly improves I/O performance
4. **GPU acceleration** speeds up Whisper transcription
5. **Close background apps** to free memory during processing

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No API keys found in keys.txt"
```bash
# Solution: Create keys.txt with Google API keys
echo "your_api_key_here" > keys.txt

# Get API keys from:
# https://makersuite.google.com/app/apikey
```

#### 2. "TTS model directory not found"
```bash
# Solution: Download and extract TTS model
# Place files in: ./vits-piper-fa_IR-amir-medium/
# Required structure:
#   fa_IR-amir-medium.onnx
#   tokens.txt
#   espeak-ng-data/
```

#### 3. "FFmpeg not found"
```bash
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Add FFmpeg to PATH after installation
```

#### 4. Memory Errors
```bash
# Solution 1: Use faster-whisper
pip install faster-whisper

# Solution 2: Use smaller Whisper model
# Edit script to use 'tiny.en' or 'base.en' instead of 'medium'

# Solution 3: Process shorter video segments
ffmpeg -i long_video.mp4 -t 1800 -c copy segment_1.mp4
```

#### 5. API Quota Exceeded
```bash
# Solution: Add multiple API keys to keys.txt
cat > keys.txt << EOF
api_key_1
api_key_2
api_key_3
EOF
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run script with debug output
python final_v3.py input.mp4 output.mp4
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Areas for Improvement

- [ ] Support for additional languages (Arabic, Turkish, etc.)
- [ ] GUI interface (Electron, PyQt, Gradio)
- [ ] Real-time dubbing capabilities
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Docker containerization
- [ ] API service wrapper
- [ ] Subtitle editor integration
- [ ] Quality assessment metrics

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/dubbing_eng_to_farsi.git
cd dubbing_eng_to_farsi

# Create feature branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .
flake8 .

# Commit and push
git add .
git commit -m "Add: Your feature description"
git push origin feature/your-feature-name
```

---

## 📄 License

This project is provided **as-is for educational and personal use**.

### Usage Terms

- ✅ Educational purposes
- ✅ Personal projects
- ✅ Research and development
- ⚠️ Commercial use: Please contact for licensing
- ❌ Redistribution without attribution

### Third-Party Licenses

Please respect the licenses of all dependencies:
- Google Gemini API: [Terms of Service](https://ai.google.dev/terms)
- OpenAI Whisper: MIT License
- FFmpeg: LGPL/GPL (depending on build)
- Sherpa-ONNX: Apache License 2.0

### Content Rights

**Important**: Always respect copyright and content rights when dubbing videos. This tool should only be used for:
- Content you own
- Content you have permission to modify
- Fair use scenarios (educational, commentary, etc.)

---

## 🆘 Support

### Documentation

- **Detailed Usage**: See `final/readme.md`
- **Examples**: See `final/instruction.md`
- **Component Docs**: Individual READMEs in each directory

### Getting Help

1. Check existing documentation in respective directories
2. Review troubleshooting section above
3. Test with shorter video clips first
4. Verify all dependencies are installed correctly
5. Check console output for specific error messages

### Resources

- [Whisper Documentation](https://github.com/openai/whisper)
- [Gemini API Guide](https://ai.google.dev/docs)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Sherpa-ONNX Guide](https://k2-fsa.github.io/sherpa/onnx/)

---

## 🎯 Use Cases

### Educational Content
- Lecture translation for multilingual students
- Training video localization
- Course material adaptation

### Content Creators
- YouTube video dubbing
- Podcast localization
- Social media content translation

### Business Applications
- Corporate training videos
- Product demonstration videos
- Marketing content localization

### Research & Development
- Speech recognition research
- Translation quality analysis
- TTS voice evaluation

---

## 🌟 Acknowledgments

This project leverages several amazing open-source projects:

- OpenAI for Whisper speech recognition
- Google for Gemini AI translation capabilities
- The Sherpa-ONNX team for Persian TTS models
- FFmpeg developers for video processing tools
- The entire open-source AI/ML community

---

## 📊 Project Statistics

- **Languages**: Python
- **Total Lines of Code**: ~5,000+
- **Python Files**: 19
- **Dependencies**: 25+ packages
- **Repository Size**: ~1.3 MB (excluding models)

---

## 🔮 Future Roadmap

### Short Term
- [ ] Docker containerization
- [ ] Web-based interface
- [ ] Subtitle editor
- [ ] Quality metrics

### Long Term
- [ ] Real-time dubbing
- [ ] Multi-language support
- [ ] Cloud API service
- [ ] Mobile app integration

---

**Last Updated**: November 2025
**Version**: 3.0
**Maintainer**: [akhavansafaei](https://github.com/akhavansafaei)
**Repository**: [dubbing_eng_to_farsi](https://github.com/akhavansafaei/dubbing_eng_to_farsi)

---

Made with ❤️ for the Persian-speaking community
