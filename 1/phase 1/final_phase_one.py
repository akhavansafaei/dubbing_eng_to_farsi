#!/usr/bin/env python3
"""
Unified Video Dubbing Script
Complete pipeline for dubbing videos from English to Persian

This script performs the following steps:
1. Transcribes video using Whisper (English)
2. Translates transcription to Persian using Gemma 3 4B
3. Refines Persian translation for better fluency
4. Generates Persian TTS audio for each subtitle line
5. Adjusts timing based on audio duration
6. Creates final dubbed video with Persian audio overlay

Usage: python video_dubber.py input_video.mp4 output_video.mp4

Requirements:
- openai-whisper or faster-whisper
- transformers
- torch
- huggingface_hub
- sherpa-onnx
- soundfile
- ffmpeg
- TTS model directory: ./vits-piper-fa_IR-amir-medium
"""

import os
import sys
import time
import json
import csv
import re
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import torch
import platform
import signal
import threading
import gc

# Import required libraries with fallbacks
try:
    import whisper
    USE_FASTER_WHISPER = False
except ImportError:
    try:
        from faster_whisper import WhisperModel
        USE_FASTER_WHISPER = True
        print("Using faster-whisper implementation")
    except ImportError:
        print("Error: Please install either 'openai-whisper' or 'faster-whisper'")
        print("Install command: pip install openai-whisper")
        sys.exit(1)

try:
    from transformers import AutoTokenizer, Gemma3ForCausalLM
    from huggingface_hub import login
except ImportError:
    print("Error: Please install transformers and huggingface_hub")
    print("Install command: pip install transformers huggingface_hub")
    sys.exit(1)

try:
    import sherpa_onnx
    import soundfile as sf
except ImportError:
    print("Error: Please install sherpa-onnx and soundfile")
    print("Install command: pip install sherpa-onnx soundfile")
    sys.exit(1)


class TimeoutException(Exception):
    """Exception raised when an operation times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Operation timed out")


class VideoDubber:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.base_name = Path(video_path).stem
        self.work_dir = f"./dubbing_work_{self.base_name}"
        
        # Initialize Gemma model configuration with better memory management
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.summary = ""
        
        # Memory management settings
        self.max_retries = 3
        self.model_load_timeout = 300  # 5 minutes timeout
        
        # TTS configuration
        self.model_dir = "./vits-piper-fa_IR-amir-medium"
        self.tts_engine = None
        
        # File paths
        self.transcript_dir = os.path.join(self.work_dir, "transcripts")
        self.audio_dir = os.path.join(self.work_dir, "audio")
        self.srt_original = os.path.join(self.transcript_dir, f"{self.base_name}.srt")
        self.srt_persian = os.path.join(self.work_dir, f"{self.base_name}_persian.srt")
        self.srt_refined = os.path.join(self.work_dir, f"{self.base_name}_refined.srt")
        self.srt_adjusted = os.path.join(self.work_dir, f"{self.base_name}_adjusted.srt")
        self.adjustment_report = os.path.join(self.work_dir, f"{self.base_name}_adjustment_report.csv")
        
        # Create work directories
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

    def check_gpu_memory(self):
        """Check available GPU memory"""
        if torch.cuda.is_available():
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
            gpu_memory_free = gpu_memory_total - gpu_memory_cached
            
            print(f"GPU Memory - Total: {gpu_memory_total:.2f}GB")
            print(f"GPU Memory - Allocated: {gpu_memory_allocated:.2f}GB")
            print(f"GPU Memory - Cached: {gpu_memory_cached:.2f}GB")
            print(f"GPU Memory - Free: {gpu_memory_free:.2f}GB")
            
            # Check if we have enough memory (need at least 8GB for Gemma 3 4B)
            if gpu_memory_free < 8.0:
                print("WARNING: Insufficient GPU memory. Consider using CPU or a smaller model.")
                return False
        return True

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def initialize_model_with_timeout(self):
        """Initialize model with timeout protection"""
        def model_loader():
            try:
                model_id = "google/gemma-3-4b-it"
                
                # Check memory before loading
                if not self.check_gpu_memory():
                    print("Switching to CPU due to insufficient GPU memory")
                    self.device = "cpu"
                
                print(f"Loading model on {self.device}...")
                
                # Load with appropriate settings
                if self.device == "cuda":
                    self.model = Gemma3ForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,  # Use half precision to save memory
                        device_map="auto",
                        low_cpu_mem_usage=True
                    ).eval()
                else:
                    self.model = Gemma3ForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    ).to(self.device).eval()
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                return True
                
            except Exception as e:
                print(f"Model loading error: {e}")
                return False
        
        # Run model loading in a separate thread with timeout
        result = [False]
        exception = [None]
        
        def target():
            try:
                result[0] = model_loader()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.model_load_timeout)
        
        if thread.is_alive():
            print("Model loading timed out. Please try again or use a smaller model.")
            return False
        
        if exception[0]:
            print(f"Model loading failed with exception: {exception[0]}")
            return False
            
        return result[0]

    def initialize_model(self):
        """Initialize Gemma 3 4B model with better error handling"""
        print("Initializing Gemma 3 4B model...")
        
        # Clear any existing GPU memory
        self.clear_gpu_memory()
        
        # Authenticate with Hugging Face
        try:
            login('')
        except Exception as e:
            print(f"Hugging Face authentication failed: {e}")
            return False
        
        # Track device details
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using GPU: {gpu_name} with {gpu_memory_total:.2f} GB memory")
        else:
            print("Using CPU")
        
        # Try loading model with retries
        for attempt in range(self.max_retries):
            print(f"Model loading attempt {attempt + 1}/{self.max_retries}")
            
            start_time = time.time()
            success = self.initialize_model_with_timeout()
            load_time = time.time() - start_time
            
            if success:
                print(f"Model loaded successfully in {load_time:.2f} seconds")
                return True
            else:
                print(f"Attempt {attempt + 1} failed")
                if attempt < self.max_retries - 1:
                    print("Retrying...")
                    self.clear_gpu_memory()
                    time.sleep(5)  # Wait before retry
        
        print("Failed to load model after all attempts")
        return False

    def get_llm_response(self, prompt, max_retries=3):
        """Get response from Gemma 3 4B model with better error handling"""
        if self.model is None or self.tokenizer is None:
            print("Model not initialized. Initializing now...")
            if not self.initialize_model():
                return "[Translation Failed - Model initialization failed]"
        
        for attempt in range(max_retries):
            try:
                # Construct messages for chat template
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                # Tokenize input with length checking
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    return_dict=True, 
                    return_tensors="pt",
                    max_length=2048,  # Limit input length
                    truncation=True
                ).to(self.device)
                
                # Generate response with timeout protection
                with torch.inference_mode():
                    # Set a reasonable timeout for generation
                    start_time = time.time()
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,  # Reduced for faster generation
                        do_sample=True, 
                        top_p=0.95, 
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True
                    )
                    generation_time = time.time() - start_time
                    
                    if generation_time > 30:  # If generation takes too long
                        print(f"Warning: Generation took {generation_time:.2f} seconds")
                
                # Decode the output
                response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Extract only the generated part (remove the input prompt)
                input_text = self.tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)[0]
                if response.startswith(input_text):
                    generated_text = response[len(input_text):].strip()
                else:
                    generated_text = response.strip()
                
                # Clear memory after successful generation
                del outputs, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return generated_text
                
            except torch.cuda.OutOfMemoryError:
                print(f"GPU out of memory on attempt {attempt + 1}")
                self.clear_gpu_memory()
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return "[Translation Failed - Out of Memory]"
                    
            except Exception as e:
                print(f"Error generating response (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return "[Translation Failed]"
        
        return "[Translation Failed - Max retries exceeded]"

    # PART 1: Video Transcription
    
    def format_time_srt(self, seconds):
        """Format time for SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def detect_speech_start(self, audio_path, energy_threshold=0.01, min_speech_duration=0.5):
        """
        Detect when actual speech starts in the audio
        Returns the timestamp (in seconds) of the first speech segment
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate frame energy
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate RMS energy for each frame
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert frame indices to time
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            # Find frames above energy threshold
            speech_frames = rms > energy_threshold
            
            # Find continuous speech segments
            speech_starts = []
            in_speech = False
            speech_start_time = 0
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and not in_speech:
                    # Start of speech segment
                    speech_start_time = times[i]
                    in_speech = True
                elif not is_speech and in_speech:
                    # End of speech segment
                    speech_duration = times[i] - speech_start_time
                    if speech_duration >= min_speech_duration:
                        speech_starts.append(speech_start_time)
                    in_speech = False
            
            # Handle case where speech continues to end of file
            if in_speech:
                speech_duration = times[-1] - speech_start_time
                if speech_duration >= min_speech_duration:
                    speech_starts.append(speech_start_time)
            
            # Return the first speech start time, or 0 if no speech detected
            return speech_starts[0] if speech_starts else 0.0
            
        except Exception as e:
            print(f"Warning: Could not detect speech start: {e}")
            return 0.0

    def extract_audio_for_analysis(self, video_path):
        """Extract audio from video for speech detection"""
        temp_audio = os.path.join(self.work_dir, "temp_audio_analysis.wav")
        
        try:
            # Extract first 60 seconds of audio for analysis (to save time)
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-t", "60",  # First 60 seconds only
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM format for librosa
                "-ar", "22050",  # Sample rate
                "-ac", "1",  # Mono
                temp_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Could not extract audio for analysis: {result.stderr}")
                return None
                
            return temp_audio
            
        except Exception as e:
            print(f"Warning: Audio extraction failed: {e}")
            return None

    def save_srt(self, result, output_path, start_offset=0.0):
        """Save transcription in SRT format with optional start time offset"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                # Apply offset to both start and end times
                adjusted_start = max(0, segment['start'] - start_offset)
                adjusted_end = max(0, segment['end'] - start_offset)
                
                # Skip segments that would have negative or zero duration after adjustment
                if adjusted_end <= adjusted_start:
                    continue
                    
                start_time = self.format_time_srt(adjusted_start)
                end_time = self.format_time_srt(adjusted_end)
                text = segment['text'].strip()
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

    def transcribe_video(self):
        """Transcribe video using Whisper and adjust timing to start from first speech"""
        print(f"Step 1: Transcribing video: {self.video_path}")
        
        if not os.path.exists(self.video_path):
            print(f"Error: Input file '{self.video_path}' not found")
            return False

        print(f"Loading Whisper base.en model...")
        try:
            if USE_FASTER_WHISPER:
                model = WhisperModel("base.en", device="cpu")
                print("Faster-Whisper model loaded successfully")
            else:
                model = whisper.load_model("base.en")
                print("OpenAI-Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        start_time = time.time()
        try:
            # First, perform transcription
            if USE_FASTER_WHISPER:
                segments, info = model.transcribe(self.video_path, language="en")
                result = {
                    'text': '',
                    'segments': [],
                    'language': info.language
                }
                
                for segment in segments:
                    segment_dict = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    }
                    result['segments'].append(segment_dict)
                    result['text'] += segment.text
            else:
                result = model.transcribe(self.video_path, language="en")
            
            end_time = time.time()
            print(f"Transcription completed in {end_time - start_time:.2f} seconds")
            
            if not result['segments']:
                print("No speech segments found in the video")
                return False
            
            # Detect when actual speech starts
            print("Detecting speech start time...")
            temp_audio = self.extract_audio_for_analysis(self.video_path)
            
            speech_start_offset = 0.0
            if temp_audio and os.path.exists(temp_audio):
                speech_start_offset = self.detect_speech_start(temp_audio)
                print(f"Detected speech starts at: {speech_start_offset:.2f} seconds")
                
                # Clean up temporary audio file
                try:
                    os.remove(temp_audio)
                except:
                    pass
            else:
                # Fallback: use the start time of the first transcribed segment
                speech_start_offset = result['segments'][0]['start']
                print(f"Using first segment start time: {speech_start_offset:.2f} seconds")
            
            # Save SRT with adjusted timing
            self.save_srt(result, self.srt_original, start_offset=speech_start_offset)
            print(f"SRT file saved: {self.srt_original}")
            print(f"Timing adjusted to start from: {speech_start_offset:.2f} seconds")
            
            if result['segments']:
                original_duration = result['segments'][-1]['end']
                adjusted_duration = original_duration - speech_start_offset
                print(f"Original duration: {original_duration:.2f} seconds")
                print(f"Adjusted duration: {adjusted_duration:.2f} seconds")
                print(f"Number of segments: {len(result['segments'])}")
            
            return True
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return False

    # PART 2: Translation to Persian
    def parse_srt_file(self, file_path):
        """Parse SRT file into segments"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        blocks = re.split(r'\n\s*\n', content.strip())
        segments = []
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3 and '-->' in lines[1]:
                try:
                    segments.append({
                        'sequence': lines[0],
                        'start': lines[1].split(' --> ')[0].strip(),
                        'end': lines[1].split(' --> ')[1].strip(),
                        'text': '\n'.join(lines[2:]).strip()
                    })
                except (ValueError, IndexError):
                    print(f"Skipping malformed SRT block:\n{block}")
        return segments

    def create_summary_prompt(self, full_text):
        """Create prompt for transcript summary"""
        return f"""
        Please provide a concise, comprehensive summary of the following English transcript.
        This summary will be used to provide overall context for a translation task.

        Transcript:
        {full_text}
        """

    def create_translation_prompt(self, current_text, previous_english, next_english, summary):
        """Create translation prompt with context"""
        surrounding_context = []
        if previous_english:
            surrounding_context.append(f"Previous English Line:\n{previous_english}")
        if next_english:
            surrounding_context.append(f"Next English Line:\n{next_english}")
        surrounding_context_section = "\n\n".join(surrounding_context) if surrounding_context else "No surrounding lines."

        return f"""
        You are an expert translator for English to Persian (Farsi) subtitles. Your task is to translate the "CURRENT TEXT TO TRANSLATE" while maintaining consistency with the provided context.

        ---
        CONTEXT 1: OVERALL SUMMARY
        {summary}
        ---
        CONTEXT 2: SURROUNDING ENGLISH LINES
        {surrounding_context_section}
        ---

        CURRENT TEXT TO TRANSLATE:
        {current_text}

        Instructions:
        1. Translate the "CURRENT TEXT TO TRANSLATE" into natural, fluent Persian.
        2. Keep the translation concise and easy to read for subtitles.
        3. Return ONLY the Persian translation. Do not include any other text, labels, or explanations.
        """

    def generate_summary(self, segments):
        """Generate summary of transcript for context"""
        print("Generating transcript summary...")
        full_text = ' '.join([segment['text'] for segment in segments])
        
        # Limit summary text to avoid memory issues
        summary_text = full_text[:8000] if len(full_text) > 8000 else full_text
        summary_prompt = self.create_summary_prompt(summary_text)
        
        self.summary = self.get_llm_response(summary_prompt)
        if not self.summary or "failed" in self.summary.lower():
            print("Warning: Could not generate a summary. Proceeding without it.")
            self.summary = "No summary available."
        else:
            print(f"Summary generated: {self.summary[:120]}...")
        return self.summary

    def translate_segments(self, segments):
        """Translate segments to Persian with better error handling"""
        if not segments:
            return []

        translated_segments = []
        print(f"Translating {len(segments)} segments...")
        
        failed_translations = 0
        
        for i in tqdm(range(len(segments)), desc="Translating"):
            current_segment = segments[i]

            previous_english_text = segments[i-1]['text'] if i > 0 else ""
            next_english_text = segments[i+1]['text'] if i < len(segments)-1 else ""

            translation_prompt = self.create_translation_prompt(
                current_segment['text'],
                previous_english_text,
                next_english_text,
                self.summary
            )
            
            persian_text = self.get_llm_response(translation_prompt)
            
            # Check if translation failed
            if "Translation Failed" in persian_text:
                failed_translations += 1
                print(f"Translation failed for segment {i+1}: {current_segment['text'][:50]}...")
                # Use original text as fallback
                persian_text = f"[FAILED] {current_segment['text']}"

            translated_segments.append({**current_segment, 'text': persian_text})
            
            # Add a small delay to prevent overwhelming the GPU
            time.sleep(0.5)
            
            # Clear memory every 5 translations
            if i % 5 == 0:
                self.clear_gpu_memory()
        
        if failed_translations > 0:
            print(f"Warning: {failed_translations} translations failed out of {len(segments)}")
        
        return translated_segments

    def write_srt_file(self, segments, output_path):
        """Write segments to SRT file"""
        with open(output_path, 'w', encoding='utf-8') as file:
            for segment in segments:
                file.write(f"{segment['sequence']}\n")
                file.write(f"{segment['start']} --> {segment['end']}\n")
                file.write(f"{segment['text']}\n\n")

    def translate_to_persian(self):
        """Step 2: Translate English SRT to Persian with better error handling"""
        print(f"Step 2: Translating to Persian...")
        
        segments = self.parse_srt_file(self.srt_original)
        if not segments:
            print("No valid subtitle segments found.")
            return False
        
        print(f"Found {len(segments)} subtitle segments.")
        
        # Initialize model here to check if it works before processing
        if not self.initialize_model():
            print("Error: Failed to initialize translation model")
            return False
        
        try:
            self.generate_summary(segments)
            translated_segments = self.translate_segments(segments)
            self.write_srt_file(translated_segments, self.srt_persian)
            
            print(f"Persian translation saved: {self.srt_persian}")
            return True
            
        except KeyboardInterrupt:
            print("\nTranslation interrupted by user")
            return False
        except Exception as e:
            print(f"Error during translation: {e}")
            return False
        finally:
            # Clean up model to free memory
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            self.clear_gpu_memory()

    # PART 3: Advanced Refinement with Two-Stage Processing
    def create_refinement_prompt(self, srt_line, original_prev, original_next, refined_prev):
        """Create refinement prompt for Stage 1 processing"""
        local_context_parts = []
        if original_prev:
            local_context_parts.append(f"Original Previous Line:\n{original_prev}")
        if original_next:
            local_context_parts.append(f"Original Next Line:\n{original_next}")
        local_context_section = "\n\n".join(local_context_parts) if local_context_parts else "No surrounding lines in the original SRT."

        refined_history_section = refined_prev if refined_prev else "This is the first line to be refined."

        return f"""You are a Persian language expert specializing in subtitle editing for improved fluency. Refine the following subtitle line based on the provided context:

    ---
    CONTEXT 1: LOCAL (The original surrounding subtitles)
    {local_context_section}
    ---
    CONTEXT 2: HISTORY (The immediately preceding line you just refined)
    Previously Refined Line: {refined_history_section}
    ---

    LINE TO REFINE:
    {srt_line}
    ---

    Instructions:
    1. Rephrase the "LINE TO REFINE" to make it natural, grammatically correct, and cohesive with the "Previously Refined Line," without omitting any information. The length should remain almost identical to the original.
    2. Ensure the refined line flows well with the surrounding context.
    3. **CRITICAL**: Do not significantly alter the duration or character count—maintain subtitle timing and readability.
    4. Return **only** the refined Persian line, without any additional commentary.
    5. If a sentence fragment belongs to the next subtitle line, move it accordingly to preserve semantic completeness and grammatical accuracy.
    """

    def create_overlap_refinement_prompt(self, ten_lines):
        """Create prompt for Stage 2 overlap refinement of 10 lines"""
        lines_text = "\n".join([f"LINE_{i+1}: {line}" for i, line in enumerate(ten_lines)])
    
        return f"""شما یک ویرایشگر حرفه‌ای زیرنویس فارسی هستید. در هر مرحله ۱۰ خط از یک متن ترجمه‌شده‌ی زیرنویس به شما داده می‌شود. این متن قرار است به صورت صوتی و توسط مدل تبدیل متن به گفتار (TTS) خوانده شود.

    **مهم: دقیقاً همان تعداد خط ({len(ten_lines)} خط) را که دریافت کرده‌اید برگردانید.**

    متن را با هدف خوانایی طبیعی و گویایی روان بازنویسی و اصلاح کنید.
    لحن جمله‌ها باید طبیعی، محاوره‌ای ملایم، و مناسب گفتار باشد.
    ساختار جمله‌ها را طوری تنظیم کنید که در گفتار واضح و راحت ادا شوند.
    اشتباهات ناشی از تقسیم‌بندی زیرنویس‌ها را اصلاح کنید.
    اگر عبارتی خوب و طبیعی است، تغییری در آن ندهید.
    ویرایش‌های شما باید حداقلی اما مؤثر باشند.

    **فرمت پاسخ:**
    دقیقاً به همین فرمت پاسخ دهید:
    LINE_1: [متن اصلاح شده خط ۱]
    LINE_2: [متن اصلاح شده خط ۲]
    LINE_3: [متن اصلاح شده خط ۳]
    ...
    LINE_{len(ten_lines)}: [متن اصلاح شده خط {len(ten_lines)}]

    زیرنویس‌های دریافتی:
    {lines_text}

    **نکات مهم:**
    - دقیقاً همان تعداد خط را برگردانید که دریافت کردید
    - از فرمت LINE_X استفاده کنید
    - هیچ توضیح اضافی یا کامنت ندهید
    - اگر خطی خالی است، آن را خالی برگردانید"""

    def parse_overlap_response(self, response_text, expected_count):
        """
        Parse the LLM response for overlap refinement with multiple fallback strategies.
        Returns a list of refined lines or None if parsing fails completely.
        """
        if not response_text or response_text.strip() == "[Refinement Failed]":
            return None
            
        response_text = response_text.strip()
        refined_lines = []
        
        # Strategy 1: Parse LINE_X: format
        line_pattern = re.compile(r'^LINE_(\d+):\s*(.*)$', re.MULTILINE)
        matches = line_pattern.findall(response_text)
        
        if matches and len(matches) == expected_count:
            # Sort by line number and extract text
            matches.sort(key=lambda x: int(x[0]))
            refined_lines = [match[1].strip() for match in matches]
            return refined_lines
        
        # Strategy 2: Split by lines and remove numbering
        lines = response_text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove various numbering patterns
            line = re.sub(r'^LINE_\d+:\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^\d+\)\s*', '', line)
            line = re.sub(r'^\d+\s*[-–—]\s*', '', line)
            
            if line:
                processed_lines.append(line)
        
        if len(processed_lines) == expected_count:
            return processed_lines
        
        # Strategy 3: Try to extract exactly expected_count non-empty lines
        all_lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        # Remove common prefixes from all lines
        cleaned_lines = []
        for line in all_lines:
            # Remove numbering and prefixes
            clean_line = re.sub(r'^(LINE_\d+:|[۰-۹]+[\.:\-\)]|\d+[\.:\-\)]|[•\-\*])\s*', '', line)
            if clean_line:
                cleaned_lines.append(clean_line)
        
        if len(cleaned_lines) == expected_count:
            return cleaned_lines
        
        # Strategy 4: If we have more lines than expected, take the first expected_count
        if len(cleaned_lines) > expected_count:
            return cleaned_lines[:expected_count]
        
        # Strategy 5: If we have fewer lines, pad with empty strings
        if len(cleaned_lines) < expected_count:
            while len(cleaned_lines) < expected_count:
                cleaned_lines.append("")
            return cleaned_lines
        
        return None

    def initialize_stage2_model(self):
        """Initialize Stage 2 model (gemini-2.5 for better refinement)"""
        try:
            # Use a more advanced model for Stage 2 if available
            self.llm_stage2 = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
            return True
        except:
            # Fallback to the same model if pro version not available
            self.llm_stage2 = self.llm
            return True

    def get_llm_response_stage2(self, prompt, max_retries=5):
        """Get response from Stage 2 LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.llm_stage2.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                error_message = str(e).lower()
                print(f"\nStage 2 Error on attempt {attempt + 1}: {error_message}")
                if any(keyword in error_message for keyword in ["quota", "limit", "exceeded", "resource has been exhausted"]):
                    print("Quota likely exceeded, switching API key...")
                    self.switch_api_key()
                    # Also update stage 2 model with new key
                    self.initialize_stage2_model()
                    time.sleep(3)
                else:
                    time.sleep(attempt + 2)

                if attempt == max_retries - 1:
                    print(f"Stage 2 failed to get a response after {max_retries} attempts.")
                    return "[Refinement Failed]"
        return "[Refinement Failed]"

    def final_overlap_refinement(self, segments):
        """
        Perform final overlap refinement on the SRT segments using Stage 2 model.
        Processes 10 lines at a time with 5-line overlap.
        Each chunk reads the most up-to-date refined lines from previous processing.
        """
        print("\nStarting Stage 2: Final overlap refinement...")
    
        # Extract just the text from segments for processing
        texts = [segment['text'] for segment in segments]
        refined_texts = texts.copy()  # Start with Stage 1 refined texts
    
        # Process in overlapping chunks of 10 with 5-line steps
        start_idx = 0
        total_chunks = max(1, (len(texts) - 5) // 5 + 1) if len(texts) > 10 else 1
    
        with tqdm(total=total_chunks, desc="Stage 2: Overlap refinement") as pbar:
            while start_idx < len(texts):
                end_idx = min(start_idx + 10, len(texts))
                
                # CRITICAL: Use refined_texts (most up-to-date) instead of original texts
                current_chunk = refined_texts[start_idx:end_idx]
            
                # Skip if chunk is too small (less than 2 lines)
                if len(current_chunk) < 2:
                    print(f"Skipping chunk {start_idx}-{end_idx-1}: too small ({len(current_chunk)} lines)")
                    break
                
                chunk_size = len(current_chunk)
                print(f"\nProcessing chunk {start_idx}-{end_idx-1} ({chunk_size} lines)")
                
                # Get refined version of this chunk using Stage 2 model with retry logic
                max_chunk_retries = 5
                refined_lines = None
                
                for retry in range(max_chunk_retries):
                    prompt = self.create_overlap_refinement_prompt(current_chunk)
                    refined_chunk_text = self.get_llm_response_stage2(prompt)
                    
                    if refined_chunk_text and refined_chunk_text != "[Refinement Failed]":
                        # Parse the response back into individual lines
                        refined_lines = self.parse_overlap_response(refined_chunk_text, chunk_size)
                        
                        if refined_lines and len(refined_lines) == chunk_size:
                            print(f"✓ Successfully refined chunk {start_idx}-{end_idx-1}")
                            break
                        else:
                            actual_count = len(refined_lines) if refined_lines else 0
                            print(f"⚠ Retry {retry+1}/{max_chunk_retries}: Expected {chunk_size} lines, got {actual_count}")
                            if retry < max_chunk_retries - 1:
                                time.sleep(3)
                    else:
                        print(f"⚠ Retry {retry+1}/{max_chunk_retries}: LLM response failed")
                        if retry < max_chunk_retries - 1:
                            time.sleep(3)
                
                # Apply refinements if successful, otherwise keep Stage 1 result
                if refined_lines and len(refined_lines) == chunk_size:
                    for i, refined_line in enumerate(refined_lines):
                        actual_idx = start_idx + i
                        # Update refined_texts immediately for next chunk to use
                        refined_texts[actual_idx] = refined_line
                        
                    print(f"✓ Applied refinements to lines {start_idx}-{end_idx-1}")
                else:
                    print(f"⚠ Could not refine chunk {start_idx}-{end_idx-1} after {max_chunk_retries} retries.")
                    print(f"⚠ Keeping Stage 1 refined text for lines {start_idx}-{end_idx-1}")
            
                # Move to next chunk (5 lines forward for overlap)
                start_idx += 5
                pbar.update(1)
                time.sleep(0.5)  # Respect API rate limits
    
        # Update the segments with final refined texts
        final_segments = []
        for i, segment in enumerate(segments):
            final_segments.append({**segment, 'text': refined_texts[i]})
    
        return final_segments

    def refine_persian_srt(self):
        """Step 3: Advanced Two-Stage Persian SRT Refinement"""
        print(f"Step 3: Advanced refinement of Persian translation...")
        
        # Initialize Stage 2 model
        if not self.initialize_stage2_model():
            print("Warning: Could not initialize Stage 2 model, using Stage 1 model for both stages")
        
        segments = self.parse_srt_file(self.srt_persian)
        if not segments:
            print("No valid segments found in Persian SRT file.")
            return False
        
        print(f"Found {len(segments)} segments to refine.")
        
        # Stage 1: Sequential refinement with context
        print("Starting Stage 1: Sequential refinement with local context...")
        stage1_segments = []
        refined_history = []
        
        for i, segment in enumerate(tqdm(segments, desc="Stage 1: Sequential refinement")):
            if not segment['text'].strip():
                refined_text = ""
            else:
                # Get context from surrounding segments
                original_prev = segments[i-1]['text'] if i > 0 else None
                original_next = segments[i+1]['text'] if i < len(segments) - 1 else None
                refined_prev = refined_history[-1] if refined_history else None

                prompt = self.create_refinement_prompt(
                    segment['text'], original_prev, original_next, refined_prev
                )
                refined_text = self.get_llm_response(prompt)
                time.sleep(0.5)
            
            stage1_segments.append({**segment, 'text': refined_text})
            refined_history.append(refined_text)

        print("✅ Stage 1 completed: Sequential refinement")
        
        # Save intermediate result
        stage1_path = self.srt_refined.replace('.srt', '_stage1.srt')
        self.write_srt_file(stage1_segments, stage1_path)
        print(f"Stage 1 result saved: {stage1_path}")
        
        # Stage 2: Overlap refinement for better coherence
        final_segments = self.final_overlap_refinement(stage1_segments)
        
        # Save final refined result
        self.write_srt_file(final_segments, self.srt_refined)
        print(f"✅ Stage 2 completed: Final overlap refinement")
        print(f"Final refined Persian SRT saved: {self.srt_refined}")
        
        # Show refinement summary
        print("\n" + "="*50)
        print("REFINEMENT SUMMARY")
        print("="*50)
        print(f"Stage 1 (Sequential): {len(stage1_segments)} lines processed")
        print(f"Stage 2 (Overlap): Enhanced coherence across subtitle boundaries")
        print(f"Model used: {self.llm.model_name} (Stage 1) + {getattr(self.llm_stage2, 'model_name', 'Same as Stage 1')} (Stage 2)")
        print(f"Final output: {self.srt_refined}")
        print("="*50)
        
        return True

    # PART 4: TTS Audio Generation
    def initialize_tts(self):
        """Initialize TTS engine"""
        if not os.path.isdir(self.model_dir):
            print(f"Error: TTS model directory not found at '{self.model_dir}'.")
            print("Please ensure you have the Persian TTS model downloaded.")
            return False

        print("Initializing Text-to-Speech engine...")
        model_path = f"{self.model_dir}/fa_IR-amir-medium.onnx"
        tokens_path = f"{self.model_dir}/tokens.txt"
        espeak_data_path = f"{self.model_dir}/espeak-ng-data"

        if not all(os.path.exists(p) for p in [model_path, tokens_path, espeak_data_path]):
            print(f"Error: One or more model files are missing from '{self.model_dir}'.")
            return False

        try:
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=model_path,
                        lexicon="",
                        tokens=tokens_path,
                        data_dir=espeak_data_path
                    ),
                    provider="cpu",
                    num_threads=2,
                ),
                max_num_sentences=1,
            )
            self.tts_engine = sherpa_onnx.OfflineTts(tts_config)
            print("TTS engine initialized successfully.")
            return True
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            return False

    def text_to_speech_persian(self, text, output_filename):
        """Generate Persian TTS audio"""
        try:
            audio = self.tts_engine.generate(text, sid=0, speed=1.0)

            if len(audio.samples) == 0:
                print(f"Warning: TTS failed for text: '{text}'. No audio generated.")
                return False

            sf.write(
                output_filename,
                audio.samples,
                samplerate=audio.sample_rate,
                subtype="PCM_16",
            )
            return True
        except Exception as e:
            print(f"An error occurred during TTS generation for '{text}': {e}")
            return False

    def generate_tts_audio(self):
        """Step 4: Generate TTS audio for each line"""
        print(f"Step 4: Generating Persian TTS audio...")
        
        # Ask user to check the refined SRT file
        print(f"\nPlease check the refined Persian SRT file at: {self.srt_refined}")
        print("Review the translation quality and make sure it's acceptable.")
        
        while True:
            user_input = input("\nIs the Persian translation approved? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                break
            elif user_input in ['n', 'no']:
                print("Please review and manually edit the Persian SRT file if needed.")
                print("You can edit the file and run the script again.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
        if not self.initialize_tts():
            return False
        
        segments = self.parse_srt_file(self.srt_refined)
        if not segments:
            return False
        
        print(f"Found {len(segments)} subtitle segments to process.")
        
        for segment in tqdm(segments, desc="Generating Audio"):
            line_text = segment['text']
            sequence_num = segment['sequence']
            output_filename = os.path.join(self.audio_dir, f"line_{sequence_num}.wav")
            
            if os.path.exists(output_filename):
                continue
                
            self.text_to_speech_persian(line_text, output_filename)
            time.sleep(0.1)

        print("Audio generation completed!")
        return True

    # PART 5: Timing Adjustment
    def time_str_to_seconds(self, time_str):
        """Convert SRT time format to seconds"""
        h, m, s_ms = time_str.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    def seconds_to_time_str(self, seconds):
        """Convert seconds to SRT time format"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = round((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def find_audio_file(self, sequence_num):
        """Find audio file for sequence number"""
        base_name = f"line_{sequence_num}"
        wav_path = os.path.join(self.audio_dir, f"{base_name}.wav")
        if os.path.exists(wav_path):
            return wav_path
        
        mp3_path = os.path.join(self.audio_dir, f"{base_name}.mp3")
        if os.path.exists(mp3_path):
            return mp3_path
        
        return None

    def get_audio_duration(self, file_path):
        """Get audio file duration"""
        if not file_path or not os.path.exists(file_path):
            return 0.0
        try:
            with sf.SoundFile(file_path) as f:
                return len(f) / f.samplerate
        except Exception as e:
            print(f"Warning: Could not read duration from '{file_path}'. Error: {e}")
            return 0.0

    def write_csv_report(self, report_data, output_path):
        """Write adjustment report to CSV"""
        if not report_data:
            return
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=report_data[0].keys())
            writer.writeheader()
            writer.writerows(report_data)

    def adjust_srt_timing(self):
        """Step 5: Adjust SRT timing based on audio duration"""
        print(f"Step 5: Adjusting SRT timing based on audio duration...")
        
        segments = self.parse_srt_file(self.srt_refined)
        if not segments:
            return False

        print(f"Processing {len(segments)} subtitle segments...")

        MAX_VOICE_SPEED = 1.35
        new_segments = []
        adjustment_report = []
        cumulative_delay = 0.0

        for segment in tqdm(segments, desc="Adjusting Timings"):
            sequence_num = segment['sequence']
            audio_file_path = self.find_audio_file(sequence_num)

            if not audio_file_path:
                print(f"Warning: Skipping line {sequence_num} - no audio file found.")
                continue

            actual_audio_duration = self.get_audio_duration(audio_file_path)
            if actual_audio_duration == 0.0:
                print(f"Warning: Skipping line {sequence_num} due to invalid audio file.")
                continue
                
            # Apply cumulative delay from previous lines
            start_sec = self.time_str_to_seconds(segment['start']) + cumulative_delay
            end_sec = self.time_str_to_seconds(segment['end']) + cumulative_delay
            original_duration = end_sec - start_sec

            voice_speedup_ratio = 1.0
            final_duration = original_duration

            if actual_audio_duration > original_duration:
                voice_speedup_ratio = min(actual_audio_duration / original_duration, MAX_VOICE_SPEED)
                duration_after_speedup = actual_audio_duration / voice_speedup_ratio
                
                if duration_after_speedup <= original_duration:
                    final_duration = original_duration
                else:
                    final_duration = duration_after_speedup
                    additional_delay = final_duration - original_duration
                    cumulative_delay += additional_delay

            final_end_sec = start_sec + final_duration

            new_segments.append({
                'sequence': segment['sequence'],
                'start': self.seconds_to_time_str(start_sec),
                'end': self.seconds_to_time_str(final_end_sec),
                'text': segment['text']
            })

            file_extension = os.path.splitext(audio_file_path)[1].upper()

            adjustment_report.append({
                "line_sequence": sequence_num,
                "audio_file_type": file_extension,
                "original_duration_sec": f"{original_duration:.3f}",
                "audio_duration_sec": f"{actual_audio_duration:.3f}",
                "voice_speedup_ratio": f"{voice_speedup_ratio:.2f}x",
                "final_duration_sec": f"{final_duration:.3f}",
                "cumulative_delay_sec": f"{cumulative_delay:.3f}",
            })
            
        self.write_srt_file(new_segments, self.srt_adjusted)
        self.write_csv_report(adjustment_report, self.adjustment_report)
        
        print(f"Adjusted SRT file saved: {self.srt_adjusted}")
        print(f"Adjustment report saved: {self.adjustment_report}")
        return True

    # PART 6: Final Video Assembly
    def parse_srt_timestamp(self, timestamp):
        """Convert SRT timestamp to seconds"""
        time_pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
        match = re.match(time_pattern, timestamp)
        if not match:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
        
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds

    def parse_csv_adjustments(self):
        """Parse CSV adjustment report"""
        adjustments = {}
        
        with open(self.adjustment_report, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                line_sequence = int(row['line_sequence'])
                voice_speedup = float(row['voice_speedup_ratio'].replace('x', ''))
                
                adjustments[line_sequence] = {
                    'voice_speedup_ratio': voice_speedup
                }
        
        return adjustments

    def find_tts_audio_file(self, sequence):
        """Find TTS audio file for sequence"""
        patterns = [
            f"line_{sequence}.wav", f"line_{sequence}.mp3",
            f"line_{sequence:02d}.wav", f"line_{sequence:02d}.mp3",
            f"line_{sequence:03d}.wav", f"line_{sequence:03d}.mp3",
        ]
        
        for pattern in patterns:
            file_path = os.path.join(self.audio_dir, pattern)
            if os.path.exists(file_path):
                return file_path
        
        return None

    # def create_final_dubbed_video(self):
    #     """Step 6: Create final dubbed video"""
    #     print(f"Step 6: Creating final dubbed video...")
    def create_final_dubbed_video(self):
        """Step 6: Create final dubbed video"""
        print(f"Step 6: Creating final dubbed video...")
        
        # Parse the adjusted SRT file
        segments = self.parse_srt_file(self.srt_adjusted)
        if not segments:
            print("Error: No segments found in adjusted SRT file")
            return False
        
        # Parse CSV adjustments
        try:
            adjustments = self.parse_csv_adjustments()
        except Exception as e:
            print(f"Error parsing CSV adjustments: {e}")
            return False
        
        print(f"Processing {len(segments)} segments with {len(adjustments)} adjustments")
        
        # Build filter complex for audio mixing
        filter_complex_parts = []
        input_files = [self.video_path]  # Original video is input 0
        
        # Reduce original video audio to 5% (keep this as is)
        filter_complex_parts.append(f"[0:a]volume=0.05[orig_audio]")
        
        # Add TTS audio files as inputs
        tts_inputs = []
        missing_audio_files = []
        
        for segment in segments:
            seq = int(segment['sequence'])
            start_seconds = self.time_str_to_seconds(segment['start'])
            
            audio_file = self.find_tts_audio_file(seq)
            if audio_file:
                tts_inputs.append({
                    'file': audio_file,
                    'start_time': start_seconds,
                    'sequence': seq,
                    'input_index': len(input_files)
                })
                input_files.append(audio_file)
                print(f"Found audio file for segment {seq}: {os.path.basename(audio_file)}")
            else:
                missing_audio_files.append(seq)
                print(f"Warning: No audio file found for segment {seq}")
        
        if missing_audio_files:
            print(f"Missing audio files for segments: {missing_audio_files}")
        
        # Process each TTS audio
        audio_mix_inputs = ["[orig_audio]"]
        
        for i, tts_input in enumerate(tts_inputs):
            seq = tts_input['sequence']
            input_idx = tts_input['input_index']
            start_time = tts_input['start_time']
            
            # Apply speed adjustment if needed
            if seq in adjustments:
                speed_ratio = adjustments[seq]['voice_speedup_ratio']
                if abs(speed_ratio - 1.0) > 0.01:
                    # Apply speed adjustment AND normalize volume to prevent loss
                    filter_complex_parts.append(f"[{input_idx}:a]atempo={speed_ratio},volume=2.0[speed{i}]")
                    current_audio = f"[speed{i}]"
                else:
                    # Boost volume for non-speed-adjusted audio
                    filter_complex_parts.append(f"[{input_idx}:a]volume=2.0[vol{i}]")
                    current_audio = f"[vol{i}]"
            else:
                # Boost volume for audio without adjustments
                filter_complex_parts.append(f"[{input_idx}:a]volume=2.0[vol{i}]")
                current_audio = f"[vol{i}]"
            
            # Add delay to position audio at correct time
            delay_ms = int(start_time * 1000)
            filter_complex_parts.append(f"{current_audio}adelay={delay_ms}|{delay_ms}[tts{i}]")
            audio_mix_inputs.append(f"[tts{i}]")
        
        # Mix all audio inputs with proper weights
        if len(audio_mix_inputs) > 1:
            mix_inputs = "".join(audio_mix_inputs)
            # Use amix with normalize=0 to prevent automatic volume reduction
            # and set weights to give more prominence to TTS audio
            num_inputs = len(audio_mix_inputs)
            filter_complex_parts.append(f"{mix_inputs}amix=inputs={num_inputs}:dropout_transition=0:normalize=0[mixed_audio]")
            # Apply final volume boost to compensate for mixing
            filter_complex_parts.append(f"[mixed_audio]volume=1.5[final_audio]")
            output_map = "[final_audio]"
        else:
            output_map = "[orig_audio]"
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Add all input files
        for input_file in input_files:
            cmd.extend(['-i', input_file])
        
        # Add filter complex
        filter_complex = ";".join(filter_complex_parts)
        cmd.extend(['-filter_complex', filter_complex])
        
        # Map outputs
        cmd.extend([
            '-map', '0:v',      # Map video from original input
            '-map', output_map, # Map mixed audio
            '-c:v', 'copy',     # Copy video without re-encoding
            '-c:a', 'aac',      # Encode audio as AAC
            '-b:a', '192k',     # Increased audio bitrate for better quality
            '-ac', '2',         # Ensure stereo output
            self.output_path
        ])
        
        print("Running FFmpeg to create final video...")
        print(f"Filter complex: {filter_complex}")  # Debug output to see the actual filter
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            
            print(f"Final dubbed video created successfully: {self.output_path}")
            
            # Create processing report
            self.create_processing_report()
            return True
            
        except Exception as e:
            print(f"Error creating final video: {e}")
            return False   
    # def create_final_dubbed_video(self):
    #     """Step 6: Create final dubbed video"""
    #     print(f"Step 6: Creating final dubbed video...")
        
    #     # Parse the adjusted SRT file
    #     segments = self.parse_srt_file(self.srt_adjusted)
    #     if not segments:
    #         print("Error: No segments found in adjusted SRT file")
    #         return False
        
    #     # Parse CSV adjustments
    #     try:
    #         adjustments = self.parse_csv_adjustments()
    #     except Exception as e:
    #         print(f"Error parsing CSV adjustments: {e}")
    #         return False
        
    #     print(f"Processing {len(segments)} segments with {len(adjustments)} adjustments")
        
    #     # Build filter complex for audio mixing
    #     filter_complex_parts = []
    #     input_files = [self.video_path]  # Original video is input 0
        
    #     # Reduce original video audio to 5%
    #     filter_complex_parts.append(f"[0:a]volume=0.05[orig_audio]")
        
    #     # Add TTS audio files as inputs
    #     tts_inputs = []
    #     missing_audio_files = []
        
    #     for segment in segments:
    #         seq = int(segment['sequence'])
    #         start_seconds = self.time_str_to_seconds(segment['start'])
            
    #         audio_file = self.find_tts_audio_file(seq)
    #         if audio_file:
    #             tts_inputs.append({
    #                 'file': audio_file,
    #                 'start_time': start_seconds,
    #                 'sequence': seq,
    #                 'input_index': len(input_files)
    #             })
    #             input_files.append(audio_file)
    #             print(f"Found audio file for segment {seq}: {os.path.basename(audio_file)}")
    #         else:
    #             missing_audio_files.append(seq)
    #             print(f"Warning: No audio file found for segment {seq}")
        
    #     if missing_audio_files:
    #         print(f"Missing audio files for segments: {missing_audio_files}")
        
    #     # Process each TTS audio
    #     audio_mix_inputs = ["[orig_audio]"]
        
    #     for i, tts_input in enumerate(tts_inputs):
    #         seq = tts_input['sequence']
    #         input_idx = tts_input['input_index']
    #         start_time = tts_input['start_time']
            
    #         # Apply speed adjustment if needed
    #         if seq in adjustments:
    #             speed_ratio = adjustments[seq]['voice_speedup_ratio']
    #             if abs(speed_ratio - 1.0) > 0.01:
    #                 filter_complex_parts.append(f"[{input_idx}:a]atempo={speed_ratio}[speed{i}]")
    #                 current_audio = f"[speed{i}]"
    #             else:
    #                 current_audio = f"[{input_idx}:a]"
    #         else:
    #             current_audio = f"[{input_idx}:a]"
            
    #         # Add delay to position audio at correct time
    #         delay_ms = int(start_time * 1000)
    #         filter_complex_parts.append(f"{current_audio}adelay={delay_ms}|{delay_ms}[tts{i}]")
    #         audio_mix_inputs.append(f"[tts{i}]")
        
    #     # Mix all audio inputs
    #     if len(audio_mix_inputs) > 1:
    #         mix_inputs = "".join(audio_mix_inputs)
    #         filter_complex_parts.append(f"{mix_inputs}amix=inputs={len(audio_mix_inputs)}:dropout_transition=0[final_audio]")
    #         output_map = "[final_audio]"
    #     else:
    #         output_map = "[orig_audio]"
        
    #     # Build FFmpeg command
    #     cmd = ['ffmpeg', '-y']
        
    #     # Add all input files
    #     for input_file in input_files:
    #         cmd.extend(['-i', input_file])
        
    #     # Add filter complex
    #     filter_complex = ";".join(filter_complex_parts)
    #     cmd.extend(['-filter_complex', filter_complex])
        
    #     # Map outputs
    #     cmd.extend([
    #         '-map', '0:v',      # Map video from original input
    #         '-map', output_map, # Map mixed audio
    #         '-c:v', 'copy',     # Copy video without re-encoding
    #         '-c:a', 'aac',      # Encode audio as AAC
    #         '-b:a', '128k',     # Audio bitrate
    #         self.output_path
    #     ])
        
    #     print("Running FFmpeg to create final video...")
    #     try:
    #         result = subprocess.run(cmd, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             print(f"FFmpeg error: {result.stderr}")
    #             return False
            
    #         print(f"Final dubbed video created successfully: {self.output_path}")
            
    #         # Create processing report
    #         self.create_processing_report()
    #         return True
            
    #     except Exception as e:
    #         print(f"Error creating final video: {e}")
    #         return False
    def create_processing_report(self):
        """Create a processing report"""
        report_path = self.output_path.replace('.mp4', '_processing_report.json')
        
        # Count audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3']:
            audio_files.extend(Path(self.audio_dir).glob(ext))
        
        # Parse segments for report
        segments = self.parse_srt_file(self.srt_adjusted) if os.path.exists(self.srt_adjusted) else []
        adjustments = self.parse_csv_adjustments() if os.path.exists(self.adjustment_report) else {}
        
        report_data = {
            'input_video': self.video_path,
            'output_video': self.output_path,
            'work_directory': self.work_dir,
            'files_created': {
                'original_srt': self.srt_original,
                'persian_srt': self.srt_persian,
                'refined_srt': self.srt_refined,
                'adjusted_srt': self.srt_adjusted,
                'adjustment_report': self.adjustment_report
            },
            'segments_processed': len(segments),
            'adjustments_applied': len(adjustments),
            'processing_method': 'reduced_original_audio_with_tts_overlay',
            'original_audio_volume': '5%',
            'supported_audio_formats': ['wav', 'mp3'],
            'audio_files_found': len(audio_files),
            'tts_model': self.model_dir,
            'api_keys_used': len(self.api_keys)
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"Processing report saved to: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save processing report: {e}")
        
    # def create_final_dubbed_video(self):
    #         """Step 6: Create final dubbed video"""
    #         print(f"Step 6: Creating final dubbed video...")
            
    #         # Parse adjusted SRT file
    #         srt_segments = []
    #         with open(self.srt_adjusted, 'r', encoding='utf-8') as file:
    #             content = file.read()
            
    #         blocks = re.split(r'\n\s*\n', content.strip())
    #         for i, block in enumerate(blocks):
    #             if not block.strip():
    #                 continue
                    
    #             lines = block.strip().split('\n')
    #             if len(lines) < 3:
    #                 continue
                    
    #             try:
    #                 sequence = int(lines[0])
    #                 timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
    #                 if not timestamp_match:
    #                     continue
                        
    #                 start_time = timestamp_match.group(1)
    #                 end_time = timestamp_match.group(2)
    #                 text = '\n'.join(lines[2:]).strip()
                    
    #                 start_seconds = self.parse_srt_timestamp(start_time)
    #                 end_seconds = self.parse_srt_timestamp(end_time)
                    
    #                 srt_segments.append({
    #                     'sequence': sequence,
    #                     'start': start_time,
    #                     'end': end_time,
    #                     'start_seconds': start_seconds,
    #                     'end_seconds': end_seconds,
    #                     'duration_seconds': end_seconds - start_seconds,
    #                     'text': text
    #                 })
                        
    #             except (ValueError, IndexError) as e:
    #                 print(f"Warning: Skipping malformed SRT block: {e}")
    #                 continue
            
    #         if not srt_segments:
    #             print("Error: No valid SRT segments found for video creation")
    #             return False
            
    #         # Parse adjustments
    #         adjustments = self.parse_csv_adjustments()
            
    #         # Build filter complex for audio mixing
    #         filter_complex_parts = []
    #         audio_inputs = []
            
    #         # Add original video as input [0]
    #         audio_inputs = [self.video_path]  # Fixed: Store just the path, not the command part
    
    #         # Process each subtitle segment
    #         valid_segments = []
    #         for i, segment in enumerate(srt_segments):
    #             sequence = segment['sequence']
    #             start_seconds = segment['start_seconds']
    #             duration_seconds = segment['duration_seconds']
                
    #             # Find corresponding TTS audio file
    #             tts_audio_file = self.find_tts_audio_file(sequence)
    #             if not tts_audio_file:
    #                 print(f"Warning: No TTS audio file found for sequence {sequence}")
    #                 continue
                
    #             # Add TTS audio file as input
    #             input_index = len(audio_inputs)
    #             audio_inputs.append(f"-i \"{tts_audio_file}\"")
    #             valid_segments.append((len(audio_inputs) - 1, segment, sequence))  # Store input index, segment, and sequence
    
    #             # Get voice speedup ratio from adjustments
    #             voice_speedup = adjustments.get(sequence, {}).get('voice_speedup_ratio', 1.0)
            
    #             # Create filter for this audio segment
    #             if voice_speedup != 1.0:
    #                 # Apply speed adjustment and delay
    #                 filter_part = f"[{input_index}]atempo={voice_speedup}[speed{i}]; [speed{i}]adelay={int(start_seconds * 1000)}|{int(start_seconds * 1000)}[delayed{i}]"
    #             else:
    #                 # Just apply delay
    #                 filter_part = f"[{input_index}]adelay={int(start_seconds * 1000)}|{int(start_seconds * 1000)}[delayed{i}]"
                
    #             filter_complex_parts.append(filter_part)
    #         if not valid_segments:
    #             print("Error: No audio segments to process")
    #             return False
    #         # if not filter_complex_parts:
    #         #     print("Error: No audio segments to process")
    #         #     return False
    #         # Create filter for each audio segment
    #         for i, (input_index, segment, sequence) in enumerate(valid_segments):
    #             start_seconds = segment['start_seconds']
                
    #             # Get voice speedup ratio from adjustments
    #             voice_speedup = adjustments.get(sequence, {}).get('voice_speedup_ratio', 1.0)
                
    #             # Create filter for this audio segment
    #             if voice_speedup != 1.0:
    #                 # Apply speed adjustment and delay
    #                 filter_part = f"[{input_index}]atempo={voice_speedup}[speed{i}]; [speed{i}]adelay={int(start_seconds * 1000)}|{int(start_seconds * 1000)}[delayed{i}]"
    #             else:
    #                 # Just apply delay
    #                 filter_part = f"[{input_index}]adelay={int(start_seconds * 1000)}|{int(start_seconds * 1000)}[delayed{i}]"
                
    #             filter_complex_parts.append(filter_part)
    #         # Create mixing filter
    #         delayed_inputs = [f"[delayed{i}]" for i in range(len(valid_segments))]
    #         mix_filter = f"{' '.join(delayed_inputs)}amix=inputs={len(delayed_inputs)}:duration=longest:dropout_transition=0[mixed]"
    #         filter_complex_parts.append(mix_filter)
            
    #         # Combine original video audio with mixed TTS audio
    #         final_mix = "[0:a][mixed]amix=inputs=2:duration=longest:weights=0.1 1.0[final_audio]"
    #         filter_complex_parts.append(final_mix)
            
    #         # Join all filter parts
    #         filter_complex = "; ".join(filter_complex_parts)
            
    #         # Build ffmpeg command
    #         temp_output = os.path.join(self.work_dir, f"temp_dubbed_{self.base_name}.mp4")
            
    #         cmd_parts = ["ffmpeg", "-y"]
            
    #         # Add all inputs
    #         for audio_input in audio_inputs:
    #             cmd_parts.extend(["-i", audio_input])  # Fixed: Proper input format
            
    #         cmd_parts.extend([
    #             "-filter_complex", filter_complex,
    #             "-map", "0:v",  # Video from original
    #             "-map", "[final_audio]",  # Final mixed audio
    #             "-c:v", "copy",  # Copy video codec
    #             "-c:a", "aac",  # Audio codec
    #             "-b:a", "128k",  # Audio bitrate
    #             temp_output
    #         ])
            
    #         print("Running FFmpeg command...")
    #         print(f"Number of inputs: {len(audio_inputs)}")
    #         print(f"Number of valid segments: {len(valid_segments)}")
            
    #         try:
    #             result = subprocess.run(cmd_parts, capture_output=True, text=True, check=True)
    #             print("FFmpeg completed successfully")
    #         except subprocess.CalledProcessError as e:
    #             print(f"FFmpeg error: {e}")
    #             print(f"FFmpeg stderr: {e.stderr}")
    #             print(f"FFmpeg stdout: {e.stdout}")  # Added: Show stdout for more debug info
    #             return False
            
    #         # Move temp file to final output
    #         try:
    #             if os.path.exists(temp_output):
    #                 os.rename(temp_output, self.output_path)
    #                 print(f"Final dubbed video created: {self.output_path}")
    #                 return True
    #             else:
    #                 print("Error: Temporary output file was not created")
    #                 return False
    #         except Exception as e:
    #             print(f"Error moving final output: {e}")
    #             return False

    def run_full_pipeline(self):
        """Run the complete dubbing pipeline"""
        print("=" * 60)
        print("UNIFIED VIDEO DUBBING PIPELINE")
        print("=" * 60)
        
        pipeline_start = time.time()
        
        # Step 1: Transcribe video
        if not self.transcribe_video():
            print("❌ Step 1 failed: Video transcription")
            return False
        print("✅ Step 1 completed: Video transcription")
        
        # Ask user to check the transcription file
        print(f"\nPlease check the transcription SRT file at: {self.srt_original}")
        print("Review the transcription quality and make sure it's acceptable.")
        
        while True:
            user_input = input("\nIs the transcription approved? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                break
            elif user_input in ['n', 'no']:
                print("Please review and manually edit the SRT file if needed.")
                print("You can edit the file and run the script again.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
        # Step 2: Translate to Persian
        if not self.translate_to_persian():
            print("❌ Step 2 failed: Persian translation")
            return False
        print("✅ Step 2 completed: Persian translation")
        
        # Step 3: Refine Persian translation
        if not self.refine_persian_srt():
            print("❌ Step 3 failed: Persian refinement")
            return False
        print("✅ Step 3 completed: Persian refinement")
        
        # Step 4: Generate TTS audio
        if not self.generate_tts_audio():
            print("❌ Step 4 failed: TTS audio generation")
            return False
        print("✅ Step 4 completed: TTS audio generation")
        
        # Step 5: Adjust timing
        if not self.adjust_srt_timing():
            print("❌ Step 5 failed: Timing adjustment")
            return False
        print("✅ Step 5 completed: Timing adjustment")
        
        # Step 6: Create final video
        if not self.create_final_dubbed_video():
            print("❌ Step 6 failed: Final video creation")
            return False
        print("✅ Step 6 completed: Final video creation")
        
        pipeline_end = time.time()
        total_time = pipeline_end - pipeline_start
        
        print("=" * 60)
        print("🎉 DUBBING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Output video: {self.output_path}")
        print(f"Work directory: {self.work_dir}")
        print("=" * 60)
        
        return True

    def cleanup_temp_files(self, keep_audio=True):
        """Clean up temporary files (optional)"""
        print("Cleaning up temporary files...")
        
        # Remove transcript files but keep final SRT files
        if os.path.exists(self.srt_original):
            os.remove(self.srt_original)
            print(f"Removed: {self.srt_original}")
        
        # Optionally remove audio files (they take up space)
        if not keep_audio:
            import shutil
            if os.path.exists(self.audio_dir):
                shutil.rmtree(self.audio_dir)
                print(f"Removed audio directory: {self.audio_dir}")
        
        print("Cleanup completed")


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python video_dubber.py input_video.mp4 output_video.mp4")
        print("\nRequired setup:")
        print("1. Create 'keys.txt' file with Google API keys (one per line)")
        print("2. Download Persian TTS model to './vits-piper-fa_IR-amir-medium/'")
        print("3. Install required packages:")
        print("   pip install openai-whisper google-generativeai sherpa-onnx soundfile")
        print("4. Install FFmpeg")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    
    # Validate input file
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found")
        sys.exit(1)
    
    # Check if output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_video))
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist")
        sys.exit(1)
    
    # Initialize dubber and run pipeline
    dubber = VideoDubber(input_video, output_video)
    
    try:
        success = dubber.run_full_pipeline()
        if success:
            print(f"\n🎬 Success! Dubbed video saved as: {output_video}")
            
            # Ask if user wants to clean up temporary files
            cleanup_input = input("\nDo you want to clean up temporary files? (y/n): ").strip().lower()
            if cleanup_input in ['y', 'yes']:
                keep_audio_input = input("Keep audio files? (y/n): ").strip().lower()
                keep_audio = keep_audio_input in ['y', 'yes']
                dubber.cleanup_temp_files(keep_audio=keep_audio)
        else:
            print(f"\n❌ Pipeline failed. Check the logs above for details.")
            print(f"Work directory preserved at: {dubber.work_dir}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user")
        print(f"Work directory preserved at: {dubber.work_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"Work directory preserved at: {dubber.work_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()