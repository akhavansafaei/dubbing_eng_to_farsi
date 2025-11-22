# Unified Video Dubbing Script

A complete pipeline for dubbing English videos to Persian (Farsi) using AI-powered transcription, translation, and text-to-speech synthesis.

## 🎯 Overview

This script performs a complete video dubbing workflow:

1. **Transcription**: Extracts English speech using Whisper
2. **Translation**: Translates to Persian using Google Gemini AI
3. **Refinement**: Improves translation fluency and coherence
4. **TTS Generation**: Creates Persian audio using Sherpa-ONNX TTS
5. **Timing Adjustment**: Synchronizes audio with video timing
6. **Video Assembly**: Produces final dubbed video with FFmpeg

## 📋 Requirements

### System Requirements
- Python 3.8+
- FFmpeg (installed and accessible via command line)
- At least 4GB RAM
- 2GB+ free disk space for processing

### Python Packages
```bash
pip install openai-whisper google-generativeai sherpa-onnx soundfile tqdm
```

**Alternative Whisper Installation** (for better performance):
```bash
pip install faster-whisper
```

### Required Files and Setup

#### 1. Google API Keys (`keys.txt`)
Create a file named `keys.txt` in the script directory with your Google API keys:
```
your_google_api_key_1
your_google_api_key_2
your_google_api_key_3
```
- One key per line
- Multiple keys recommended for quota management
- Get keys from [Google AI Studio](https://makersuite.google.com/app/apikey)

#### 2. Persian TTS Model
Download the Persian TTS model and extract to `./vits-piper-fa_IR-amir-medium/`:

**Required model files:**
- `fa_IR-amir-medium.onnx`
- `tokens.txt`
- `espeak-ng-data/` (directory)

**Model Directory Structure:**
```
./vits-piper-fa_IR-amir-medium/
├── fa_IR-amir-medium.onnx
├── tokens.txt
└── espeak-ng-data/
    ├── lang/
    ├── voices/
    └── ...
```

## 🚀 Usage

### Basic Usage
```bash
python video_dubber.py input_video.mp4 output_video.mp4
```

### Examples

#### Example 1: Simple Dubbing
```bash
python video_dubber.py "lecture.mp4" "lecture_persian.mp4"
```

#### Example 2: Processing Multiple Videos
```bash
# Process all MP4 files in a directory
for video in *.mp4; do
    python video_dubber.py "$video" "dubbed_$video"
done
```

#### Example 3: With Custom Paths
```bash
python video_dubber.py "/path/to/input/video.mp4" "/path/to/output/dubbed_video.mp4"
```

## 📁 Output Files

The script creates a working directory `./dubbing_work_[video_name]/` containing:

```
dubbing_work_video_name/
├── transcripts/
│   └── video_name.srt              # Original English subtitles
├── audio/
│   ├── line_1.wav                  # Generated Persian audio files
│   ├── line_2.wav
│   └── ...
├── video_name_persian.srt          # Initial Persian translation
├── video_name_refined.srt          # Refined Persian translation
├── video_name_adjusted.srt         # Timing-adjusted subtitles
└── video_name_adjustment_report.csv # Timing adjustment details
```

## 🔧 Configuration Options

### API Configuration
- **Multiple API Keys**: Add multiple Google API keys to `keys.txt` for automatic quota management
- **Model Selection**: Script uses `gemini-1.5-flash` by default

### TTS Configuration
- **Voice Speed**: Maximum voice speedup ratio is 1.35x (configurable in code)
- **Audio Quality**: 16-bit PCM WAV output at model's native sample rate
- **Model Path**: Default `./vits-piper-fa_IR-amir-medium/`

### Processing Options
- **Whisper Model**: Uses `base.en` model (English-optimized)
- **Translation Context**: Uses conversation summary for better coherence
- **Timing Buffer**: Maintains subtitle readability timing standards

## 🛠️ Advanced Usage

### Step-by-Step Processing
You can run individual steps by modifying the script or using it as a library:

```python
from video_dubber import VideoDubber

dubber = VideoDubber("input.mp4", "output.mp4")

# Run individual steps
dubber.transcribe_video()
dubber.translate_to_persian()
dubber.refine_persian_srt()
dubber.generate_tts_audio()
dubber.adjust_srt_timing()
dubber.create_final_dubbed_video()
```

### Manual Review Points
The script includes user approval checkpoints:

1. **After Step 3**: Review refined Persian translation
2. **Before Step 4**: Approve TTS generation
3. **After completion**: Choose cleanup options

### Customization Options

#### Modify Translation Prompts
Edit the prompt creation methods in the `VideoDubber` class:
- `create_translation_prompt()`
- `create_refinement_prompt()`

#### Adjust TTS Settings
Modify TTS parameters in `initialize_tts()`:
- Voice speed: `speed=1.0`
- Voice ID: `sid=0`
- Threading: `num_threads=2`

#### FFmpeg Audio Mixing
Customize audio mixing in `create_final_dubbed_video()`:
- Background audio volume: `weights=0.1 1.0`
- Audio bitrate: `-b:a 128k`

## 🐛 Troubleshooting

### Common Issues

#### 1. "No API keys found in keys.txt"
- **Solution**: Create `keys.txt` with valid Google API keys
- **Check**: File exists in script directory and contains keys

#### 2. "TTS model directory not found"
- **Solution**: Download and extract Persian TTS model
- **Check**: Model files exist in `./vits-piper-fa_IR-amir-medium/`

#### 3. "FFmpeg not found"
- **Solution**: Install FFmpeg and add to system PATH
- **Windows**: Download from https://ffmpeg.org/
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

#### 4. Memory Issues
- **Solution**: Use `faster-whisper` instead of `openai-whisper`
- **Alternative**: Process shorter video segments

#### 5. API Quota Exceeded
- **Solution**: Add multiple API keys to `keys.txt`
- **Automatic**: Script automatically switches keys

### Error Logs
Check console output for detailed error messages. Common patterns:

- **Transcription errors**: Usually model loading or video format issues
- **Translation errors**: API key or network connectivity problems
- **TTS errors**: Model file corruption or text encoding issues
- **FFmpeg errors**: Audio format or codec compatibility

## 📊 Performance Guidelines

### Processing Times (Approximate)
- **5-minute video**: 15-25 minutes total processing
- **30-minute video**: 1.5-2.5 hours total processing
- **Steps breakdown**:
  - Transcription: 20% of video length
  - Translation: 30-60 seconds per subtitle line
  - TTS: 5-10 seconds per line
  - Video assembly: 10-20% of video length

### Resource Usage
- **CPU**: Moderate usage during transcription and TTS
- **Memory**: 2-4GB peak during processing
- **Disk**: 3-5x video file size for temporary files
- **Network**: API calls for translation (~1KB per subtitle)

### Optimization Tips
1. **Use faster-whisper** for better performance
2. **Multiple API keys** prevent quota delays
3. **SSD storage** improves file I/O
4. **Close other applications** to free memory

## 🔒 Privacy and Security

- **Local Processing**: Transcription and TTS run locally
- **API Usage**: Only text translation sent to Google API
- **Data Retention**: Google API doesn't store translation data beyond processing
- **Cleanup**: Use cleanup option to remove temporary files

## 📝 File Format Support

### Input Video Formats
- MP4, AVI, MOV, MKV, WMV
- Any format supported by FFmpeg

### Audio Requirements
- Clear speech audio
- Minimal background noise recommended
- English language content

### Output Format
- MP4 with H.264 video
- AAC audio codec
- Preserves original video quality

## 🤝 Contributing

### Code Structure
The script is organized into logical steps:
- Each major function handles one pipeline step
- Error handling and user interaction included
- Modular design allows easy customization

### Potential Improvements
- Support for additional languages
- GUI interface
- Batch processing automation
- Cloud deployment options
- Real-time processing

## 📄 License

This script is provided as-is for educational and personal use. Please respect:
- Google API Terms of Service
- Video content copyrights
- Model licensing terms

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Verify all requirements are installed
3. Review console error messages
4. Test with shorter video clips first

---

**Last Updated**: June 2025  
**Version**: 1.0  
**Compatibility**: Python 3.8+, FFmpeg 4.0+