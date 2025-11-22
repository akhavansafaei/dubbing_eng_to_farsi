#!/usr/bin/env python3
"""
Unified Video Dubbing Script
Complete pipeline for dubbing videos from English to Persian

This script performs the following steps:
1. Transcribes video using Whisper (English)
2. Translates transcription to Persian using local Gemma3
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

import threading
import signal
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
import os
import time
import librosa
import numpy as np
import os
import warnings
from transformers import logging as transformers_logging

# Suppress transformers warnings
transformers_logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
# from huggingface_hub import login

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM
    import torch
    from huggingface_hub import login
except ImportError:
    print("Error: Please install transformers, torch, and huggingface_hub")
    print("Install command: pip install transformers torch huggingface_hub")
    sys.exit(1)

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
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")

try:
    from transformers import AutoTokenizer, Gemma3ForCausalLM
    import torch
    from huggingface_hub import login
except ImportError:
    print("Error: Please install transformers, torch, and huggingface_hub")
    print("Install command: pip install transformers torch huggingface_hub")
    sys.exit(1)

try:
    import sherpa_onnx
    import soundfile as sf
except ImportError:
    print("Error: Please install sherpa-onnx and soundfile")
    print("Install command: pip install sherpa-onnx soundfile")
    sys.exit(1)

class VideoDubber:
    #          llm_type="local", openai_api_key=None, tts_type="local", openai_tts_voice="alloy", openai_tts_model="tts-1",
    #          # Local LLM parameters
    #          local_llm_temperature=0.3, local_llm_top_p=0.85, local_llm_top_k=40, local_llm_max_tokens=512,
    #          local_llm_repetition_penalty=1.1, local_llm_do_sample=True,
    #          # Model dtype parameter
    #          torch_dtype="auto"):
    def __init__(self, video_path, output_path, start_step=1, hf_token='', model_id="google/gemma-3-4b-it", 
             llm_type="local", openai_api_key=None, tts_type="local", openai_tts_voice="alloy", openai_tts_model=None,
             # Add local LLM parameters
             local_llm_temperature=0.3, local_llm_top_p=0.85, local_llm_top_k=40, local_llm_max_tokens=512,
             local_llm_repetition_penalty=1.1, local_llm_do_sample=True):
    
        # # Initialize device FIRST (before torch_dtype parsing)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.video_path = video_path
        # self.output_path = output_path
        # self.base_name = Path(video_path).stem
        # self.work_dir = f"./dubbing_work_{self.base_name}"
        
        # # Auto-detect model if not specified
        # if model_id is None:
        #     self.model_id = self.detect_optimal_model()
        # else:
        #     # Validate that only supported models are used
        #     supported_models = ["google/gemma-3-4b-it", "google/gemma-3-12b-it"]
        #     if model_id not in supported_models:
        #         print(f"⚠️  Warning: {model_id} is not in supported models list")
        #         print(f"📋 Supported models: {', '.join(supported_models)}")
        #         print("🔄 Auto-detecting optimal model...")
        #         self.model_id = self.detect_optimal_model()
        #     else:
        #         self.model_id = model_id
        
        # # Set torch dtype
        # self.torch_dtype_str = torch_dtype
        # self.torch_dtype = self.parse_torch_dtype(torch_dtype)
            
        # # Initialize local LLM configuration
        # self.model_id = model_id
        # self.hf_token = hf_token
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = None
        # self.tokenizer = None
        # self.summary = ""
        
        # # Model configuration
        # self.model_id = model_id
        # self.use_quantization = False
        # # self.quantization_type = quantization_type
        # # self.load_in_8bit = load_in_8bit
        # # self.load_in_4bit = load_in_4bit
        # # self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        # # self.bnb_4bit_quant_type = bnb_4bit_quant_type
        # # self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        
        # # Remove quantization parameters - no longer needed
        # self.use_quantization = False
        # # Auto-configure quantization based on quantization_type
        # if self.use_quantization:
        #     if self.quantization_type == "8bit":
        #         self.load_in_8bit = True
        #         self.load_in_4bit = False
        #     elif self.quantization_type == "4bit":
        #         self.load_in_8bit = False
        #         self.load_in_4bit = True
        #     else:
        #         print(f"Warning: Unknown quantization type '{self.quantization_type}', disabling quantization")
        #         self.use_quantization = False
        
        # # Local LLM generation parameters
        # self.local_llm_temperature = local_llm_temperature
        # self.local_llm_top_p = local_llm_top_p
        # self.local_llm_top_k = local_llm_top_k
        # self.local_llm_max_tokens = local_llm_max_tokens
        # self.local_llm_repetition_penalty = local_llm_repetition_penalty
        # self.local_llm_do_sample = local_llm_do_sample
        
        # # TTS configuration
        # self.model_dir = "./vits-piper-fa_IR-amir-medium"
        # self.tts_engine = None
        
        # # File paths
        # self.transcript_dir = os.path.join(self.work_dir, "transcripts")
        # self.audio_dir = os.path.join(self.work_dir, "audio")
        # self.srt_original = os.path.join(self.transcript_dir, f"{self.base_name}.srt")
        # self.srt_persian = os.path.join(self.work_dir, f"{self.base_name}_persian.srt")
        # self.srt_refined = os.path.join(self.work_dir, f"{self.base_name}_refined.srt")
        # self.srt_adjusted = os.path.join(self.work_dir, f"{self.base_name}_adjusted.srt")
        # self.adjustment_report = os.path.join(self.work_dir, f"{self.base_name}_adjustment_report.csv")
        
        # self.start_step = start_step
        # self.user_response = None
        # self.timeout_occurred = False
        
        # # LLM configuration
        # self.llm_type = llm_type  # "local" or "openai"
        # self.openai_api_key = openai_api_key
        # self.openai_client = None
        
        # # TTS configuration
        # self.tts_type = tts_type  # "local" or "openai"
        # self.openai_tts_voice = openai_tts_voice
        # self.openai_tts_model = openai_tts_model
        
        # # Create work directories
        # os.makedirs(self.work_dir, exist_ok=True)
        # os.makedirs(self.transcript_dir, exist_ok=True)
        # os.makedirs(self.audio_dir, exist_ok=True)
        
        # # Suppress warnings
        # self.setup_logging()
        self.video_path = video_path
        self.output_path = output_path
        self.base_name = Path(video_path).stem
        self.work_dir = f"./dubbing_work_{self.base_name}"
        
        # Initialize device FIRST (before torch_dtype parsing)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize local LLM configuration
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.summary = ""
        # self.torch_dtype =torch.float32
        # Set torch dtype (now self.device is available)
        # self.torch_dtype_str = torch_dtype
        # self.torch_dtype = self.parse_torch_dtype(torch_dtype)
        
        # Auto-detect model if not specified
        if model_id is None:
            self.model_id = self.detect_optimal_model()
        else:
            # Validate that only supported models are used
            supported_models = ["google/gemma-3-4b-it", "google/gemma-3-12b-it"]
            if model_id not in supported_models:
                print(f"⚠️  Warning: {model_id} is not in supported models list")
                print(f"📋 Supported models: {', '.join(supported_models)}")
                print("🔄 Auto-detecting optimal model...")
                self.model_id = self.detect_optimal_model()
            else:
                self.model_id = model_id
        
        # LLM configuration
        self.llm_type = llm_type
        self.openai_api_key = openai_api_key
        self.openai_client = None
        
        # Remove quantization parameters - no longer needed
        self.use_quantization = False
        
        # TTS configuration
        self.tts_type = tts_type
        self.openai_tts_voice = openai_tts_voice
        self.openai_tts_model = openai_tts_model
        self.model_dir = "./vits-piper-fa_IR-amir-medium"
        self.tts_engine = None
        
        # Local LLM generation parameters
        self.local_llm_temperature = local_llm_temperature
        self.local_llm_top_p = local_llm_top_p
        self.local_llm_top_k = local_llm_top_k
        self.local_llm_max_tokens = local_llm_max_tokens
        self.local_llm_repetition_penalty = local_llm_repetition_penalty
        self.local_llm_do_sample = local_llm_do_sample
        
        # File paths
        self.transcript_dir = os.path.join(self.work_dir, "transcripts")
        self.audio_dir = os.path.join(self.work_dir, "audio")
        self.srt_original = os.path.join(self.transcript_dir, f"{self.base_name}.srt")
        self.srt_persian = os.path.join(self.work_dir, f"{self.base_name}_persian.srt")
        self.srt_refined = os.path.join(self.work_dir, f"{self.base_name}_refined.srt")
        self.srt_adjusted = os.path.join(self.work_dir, f"{self.base_name}_adjusted.srt")
        self.adjustment_report = os.path.join(self.work_dir, f"{self.base_name}_adjustment_report.csv")
        
        self.start_step = start_step
        self.user_response = None
        self.timeout_occurred = False
        
        # Create work directories
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Suppress warnings
        self.setup_logging()
    
    def timeout_prompt_with_approval(self, prompt_message, file_path=None, timeout_minutes=5):
        """
        Ask user for approval with automatic timeout every 5 minutes
        """
        def timeout_handler():
            nonlocal timeout_count
            timeout_count = 0
            while self.user_response is None and not self.timeout_occurred:
                time.sleep(timeout_minutes * 60)  # Wait 5 minutes
                if self.user_response is None:
                    timeout_count += 1
                    print(f"\n⏰ Timeout #{timeout_count} - Still waiting for your response...")
                    print(f"⏰ Waiting {timeout_minutes} more minutes for approval...")
                    if file_path:
                        print(f"📁 File to review: {file_path}")
                    print(f"💭 Question: {prompt_message}")
        
        timeout_count = 0
        self.user_response = None
        self.timeout_occurred = False
        
        # Start timeout thread
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
        
        print(f"\n{prompt_message}")
        if file_path:
            print(f"📁 Please review: {file_path}")
        
        while True:
            try:
                print(f"\n⏱️  You have {timeout_minutes} minutes to respond (auto-reminder every {timeout_minutes} min)")
                user_input = input("Is this approved? (y/n): ").strip().lower()
                self.user_response = user_input
                
                if user_input in ['y', 'yes']:
                    print("✅ Approved! Continuing...")
                    return True
                elif user_input in ['n', 'no']:
                    print("❌ Not approved. Stopping pipeline.")
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
                    self.user_response = None  # Reset to continue asking
                    
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Process interrupted by user")
                self.timeout_occurred = True
                return False
    
    def ensure_llm_initialized(self):
        """Ensure LLM is initialized for steps that need it"""
        if self.llm_type == "openai":
            if not self.openai_client:
                return self.initialize_openai_client()
            return True
        else:
            # For local LLM
            if not self.model or not self.tokenizer:
                return self.initialize_local_llm()
            return True
    
    def initialize_openai_client(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            print("Error: OpenAI library not installed")
            return False
        
        if not self.openai_api_key:
            print("Error: OpenAI API key not provided")
            return False
        
        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            print("✓ OpenAI client initialized")
            return True
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return False
    
    def check_step_completion(self, step_num):
        """Check if a step has been completed based on expected output files"""
        completion_files = {
            1: self.srt_original,
            2: self.srt_persian,
            3: self.srt_refined,
            4: self.audio_dir,  # Check if audio directory exists with files
            5: self.srt_adjusted,
            6: self.output_path
        }
        
        if step_num in completion_files:
            file_path = completion_files[step_num]
            if step_num == 4:  # Special check for audio files
                return os.path.exists(file_path) and len(os.listdir(file_path)) > 0
            else:
                return os.path.exists(file_path)
        return False
    def get_next_required_step(self):
        """Determine the next step that needs to be executed"""
        for step in range(1, 7):
            if not self.check_step_completion(step):
                return step
        return 7  # All steps completed
    
    def authenticate_huggingface(self):
        """Authenticate with Hugging Face"""
        if self.hf_token:
            try:
                login(self.hf_token)
                print("✓ Authenticated with Hugging Face")
                return True
            except Exception as e:
                print(f"Warning: HuggingFace authentication failed: {e}")
                print("Trying to proceed without authentication...")
                return False
        else:
            print("No HuggingFace token provided, trying to proceed without authentication...")
            return False

    def detect_optimal_model(self):
        """Detect optimal model based on available GPU VRAM"""
        if not torch.cuda.is_available():
            print("⚠️  No CUDA GPU detected. Using smaller model for CPU.")
            return "google/gemma-3-4b-it"
        
        try:
            # Get GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            
            # Check available memory (not just total)
            torch.cuda.empty_cache()  # Clear cache first
            available_memory = torch.cuda.memory_reserved(0) if torch.cuda.memory_reserved(0) > 0 else gpu_memory_gb
            
            print(f"🖥️  GPU: {gpu_name}")
            print(f"📊 Total GPU Memory: {gpu_memory_gb:.1f} GB")
            
            # Memory requirements (rough estimates for fp16)
            # 12B model needs ~24GB for inference, 4B needs ~8GB
            if gpu_memory_gb >= 20:
                selected_model = "google/gemma-3-12b-it"
                print(f"✅ Selected {selected_model} (12B parameters)")
                print(f"💾 GPU memory should be sufficient ({gpu_memory_gb:.1f} GB available)")
            else:
                selected_model = "google/gemma-3-4b-it"
                print(f"✅ Selected {selected_model} (4B parameters)")
                print(f"💾 Using smaller model due to GPU memory constraint ({gpu_memory_gb:.1f} GB available)")
                
            return selected_model
            
        except Exception as e:
            print(f"⚠️  Error detecting GPU: {e}")
            print("🔄 Falling back to 4B model")
            return "google/gemma-3-4b-it"
    
    # def initialize_local_llm(self):
    #     """Initialize LLM (local or OpenAI) with optional quantization"""
    #     if self.llm_type == "openai":
    #         return self.initialize_openai_client()
    #     else:
    #         print(f"Initializing local model: {self.model_id}")
            
    #         # Authenticate if token provided
    #         self.authenticate_huggingface()
            
    #         try:
    #             # Configure quantization if requested
    #             quantization_config = None
    #             if self.use_quantization and QUANTIZATION_AVAILABLE:
    #                 print(f"Configuring {self.quantization_type} quantization...")
                    
    #                 if self.quantization_type == "4bit":
    #                     # Parse compute dtype
    #                     if self.bnb_4bit_compute_dtype == "float16":
    #                         compute_dtype = torch.float16
    #                     elif self.bnb_4bit_compute_dtype == "bfloat16":
    #                         compute_dtype = torch.bfloat16
    #                     else:
    #                         compute_dtype = torch.bfloat16  # Default for better stability
                        
    #                     quantization_config = BitsAndBytesConfig(
    #                         load_in_4bit=True,
    #                         bnb_4bit_compute_dtype=compute_dtype,
    #                         bnb_4bit_quant_type=self.bnb_4bit_quant_type,
    #                         bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
    #                     )
    #                     print(f"✓ 4-bit quantization configured:")
    #                     print(f"  - Compute dtype: {self.bnb_4bit_compute_dtype}")
    #                     print(f"  - Quantization type: {self.bnb_4bit_quant_type}")
    #                     print(f"  - Double quantization: {self.bnb_4bit_use_double_quant}")
                        
    #                 elif self.quantization_type == "8bit":
    #                     quantization_config = BitsAndBytesConfig(
    #                         load_in_8bit=True,
    #                     )
    #                     print("✓ 8-bit quantization configured")
                        
    #             elif self.use_quantization and not QUANTIZATION_AVAILABLE:
    #                 print("⚠️  Quantization requested but BitsAndBytesConfig not available")
    #                 print("⚠️  Install with: pip install bitsandbytes accelerate")
    #                 print("⚠️  Proceeding without quantization...")
                
    #             # Load tokenizer first
    #             print(f"Loading tokenizer: {self.model_id}")
    #             start_time = time.time()
    #             self.tokenizer = AutoTokenizer.from_pretrained(
    #                 self.model_id,
    #                 trust_remote_code=True
    #             )
                
    #             # Ensure pad token is set
    #             if self.tokenizer.pad_token is None:
    #                 self.tokenizer.pad_token = self.tokenizer.eos_token
                
    #             tokenizer_load_time = time.time() - start_time
    #             print(f"✓ Tokenizer loaded in {tokenizer_load_time:.2f} seconds")
                
    #             # Choose the appropriate model class based on model_id
    #             print(f"Loading model: {self.model_id}")
    #             if quantization_config:
    #                 print(f"Quantization: {self.quantization_type}")
                
    #             start_time = time.time()
                
    #             # Select model class based on model name
    #             if "gemma-3" in self.model_id.lower():
    #                 model_class = Gemma3ForCausalLM
    #                 print("Using Gemma3ForCausalLM")
    #             else:
    #                 # Try AutoModelForCausalLM as fallback
    #                 model_class = AutoModelForCausalLM
    #                 print(f"Using AutoModelForCausalLM for {self.model_id}")
                
    #             # Load model with appropriate configuration
    #             if quantization_config:
    #                 # For quantized models, use device_map="auto" and appropriate dtype
    #                 torch_dtype = torch.bfloat16 if self.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
                    
    #                 self.model = model_class.from_pretrained(
    #                     self.model_id,
    #                     quantization_config=quantization_config,
    #                     device_map="auto",
    #                     torch_dtype=torch_dtype,
    #                     trust_remote_code=True,
    #                 ).eval()
    #             else:
    #                 # For non-quantized models
    #                 torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
    #                 self.model = model_class.from_pretrained(
    #                     self.model_id,
    #                     torch_dtype=torch_dtype,
    #                     trust_remote_code=True,
    #                 ).to(self.device).eval()
                
    #             model_load_time = time.time() - start_time
    #             print(f"✓ Model loaded in {model_load_time:.2f} seconds")
                
    #             # Optional: Compile model for better performance (only if CUDA available)
    #             if torch.cuda.is_available() and hasattr(torch, 'compile'):
    #                 try:
    #                     print("🔧 Compiling model for better performance...")
    #                     self.model = torch.compile(
    #                         self.model, 
    #                         mode="reduce-overhead", 
    #                         fullgraph=False, 
    #                         dynamic=True
    #                     )
    #                     print("✓ Model compilation completed")
    #                 except Exception as e:
    #                     print(f"⚠️  Model compilation failed: {e}")
    #                     print("⚠️  Continuing without compilation...")
                
    #             # Print device and memory info
    #             if quantization_config:
    #                 print(f"✓ Using quantized model with device_map='auto'")
    #                 if torch.cuda.is_available():
    #                     print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    #             elif self.device == "cuda":
    #                 gpu_name = torch.cuda.get_device_name(0)
    #                 gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    #                 print(f"✓ Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    #             else:
    #                 print("✓ Using CPU")
                
    #             # Print model info
    #             if hasattr(self.model, 'config'):
    #                 if hasattr(self.model.config, 'num_parameters'):
    #                     params = self.model.config.num_parameters
    #                     print(f"✓ Model parameters: ~{params/1e9:.1f}B")
    #                 elif hasattr(self.model, 'num_parameters'):
    #                     try:
    #                         params = self.model.num_parameters()
    #                         print(f"✓ Model parameters: ~{params/1e9:.1f}B")
    #                     except:
    #                         print("✓ Model loaded successfully")
                
    #             return True
                
    #         except Exception as e:
    #             print(f"Error initializing local LLM: {e}")
    #             print(f"Model ID: {self.model_id}")
    #             if self.use_quantization:
    #                 print("Try running without quantization or install: pip install bitsandbytes accelerate")
    #             return False
    
    # def parse_torch_dtype(self, dtype_str):
    #     """Parse torch dtype from string parameter"""
    #     if dtype_str == "auto":
    #         # Auto-detect based on device and model
    #         if self.device == "cuda":
    #             # Check if bfloat16 is supported
    #             if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    #                 print("🔧 Auto-detected dtype: bfloat16 (optimal for modern GPUs)")
    #                 return torch.bfloat16
    #             else:
    #                 print("🔧 Auto-detected dtype: float16 (bfloat16 not supported)")
    #                 return torch.float16
    #         else:
    #             print("🔧 Auto-detected dtype: float32 (CPU inference)")
    #             return torch.float32
    #     elif dtype_str == "float16" or dtype_str == "fp16":
    #         print("🔧 Using dtype: float16")
    #         return torch.float16
    #     elif dtype_str == "bfloat16" or dtype_str == "bf16":
    #         print("🔧 Using dtype: bfloat16")
    #         return torch.bfloat16
    #     elif dtype_str == "float32" or dtype_str == "fp32":
    #         print("🔧 Using dtype: float32")
    #         return torch.float32
    #     else:
    #         print(f"⚠️  Warning: Unknown dtype '{dtype_str}', using auto-detection")
    #         return self.parse_torch_dtype("auto")

    # def get_dtype_info(self):
    #     """Get information about the selected dtype"""
    #     dtype_info = {
    #         torch.float16: {"name": "float16", "memory": "Half precision", "speed": "Fast", "compatibility": "Good"},
    #         torch.bfloat16: {"name": "bfloat16", "memory": "Half precision", "speed": "Fast", "compatibility": "Modern GPUs"},
    #         torch.float32: {"name": "float32", "memory": "Full precision", "speed": "Slower", "compatibility": "Universal"}
    #     }
    #     return dtype_info.get(self.torch_dtype, {"name": "unknown", "memory": "Unknown", "speed": "Unknown", "compatibility": "Unknown"})
    
    def setup_logging(self):
        """Setup logging to suppress unnecessary warnings"""
        import logging
        import warnings
        from transformers import logging as transformers_logging
        
        # Suppress transformers warnings
        transformers_logging.set_verbosity_error()
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        warnings.filterwarnings("ignore", message=".*generation flags.*")
        
        # Set logging level for transformers
        logging.getLogger("transformers").setLevel(logging.ERROR)
    
    def initialize_local_llm(self):
        """Initialize local Gemma-3 LLM with configurable dtype"""
        if self.llm_type == "openai":
            return self.initialize_openai_client()
        else:
            print(f"Initializing local model: {self.model_id}")
            
            # Authenticate if token provided
            self.authenticate_huggingface()
            
            try:
                # Load tokenizer first
                print(f"Loading tokenizer: {self.model_id}")
                start_time = time.time()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )
                
                # Ensure pad token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                tokenizer_load_time = time.time() - start_time
                print(f"✅ Tokenizer loaded in {tokenizer_load_time:.2f} seconds")
                
                # Display dtype information
                # dtype_info = self.get_dtype_info()
                # print(f"🔧 Model dtype: {dtype_info['name']} ({dtype_info['memory']}, {dtype_info['speed']})")
                
                # Check dtype compatibility
                # if self.torch_dtype == torch.bfloat16:
                #     if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                #         print("⚠️  Warning: bfloat16 not supported on this GPU, falling back to float16")
                #         self.torch_dtype = torch.float16
                #     elif not torch.cuda.is_available():
                #         print("⚠️  Warning: bfloat16 not recommended for CPU, falling back to float32")
                #         self.torch_dtype = torch.float32
                
                # Load model
                print(f"Loading model: {self.model_id}")
                print(f"Device: {self.device}")
                # print(f"Dtype: {self.torch_dtype}")
                
                start_time = time.time()
                
                # Use Gemma3ForCausalLM for both models
                from transformers import Gemma3ForCausalLM
                
                self.model = Gemma3ForCausalLM.from_pretrained(
                    self.model_id,
                    # torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    device_map="auto",  # Let transformers handle device mapping
                ).eval()
                
                model_load_time = time.time() - start_time
                print(f"✅ Model loaded in {model_load_time:.2f} seconds")
                
                # Print device and memory info
                if self.device == "cuda":
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"🖥️  Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                    
                    # Check memory usage after loading
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved = torch.cuda.memory_reserved(0) / (1024**3)
                        print(f"📊 GPU Memory - Allocated: {allocated:.1f} GB, Reserved: {reserved:.1f} GB")
                else:
                    print("🖥️  Using CPU")
                
                # Print model info
                # model_size = "12B" if "12b" in self.model_id.lower() else "4B"
                # print(f"🧠 Model: Gemma-3 {model_size} ({dtype_info['name']})")
                
                # Memory efficiency info
                # if self.torch_dtype == torch.float16 or self.torch_dtype == torch.bfloat16:
                #     print("💾 Memory efficiency: ~50% reduction vs float32")
                
                return True
                
            except Exception as e:
                print(f"❌ Error initializing local LLM: {e}")
                print(f"Model ID: {self.model_id}")
                # print(f"Dtype: {self.torch_dtype}")
                
                # Try fallback to float16 if bfloat16 fails
                # if self.torch_dtype == torch.bfloat16:
                #     print("🔄 Trying fallback to float16...")
                #     self.torch_dtype = torch.float16
                #     return self.initialize_local_llm()
                
                # Try fallback to 4B model if 12B fails
                if "12b" in self.model_id.lower():
                    print("🔄 Trying fallback to 4B model...")
                    self.model_id = "google/gemma-3-4b-it"
                    return self.initialize_local_llm()
                
                return False
            
    # def initialize_local_llm(self):
    #     """Initialize LLM (local or OpenAI)"""
    #     if self.llm_type == "openai":
    #         return self.initialize_openai_client()
    #     else:
    #         # Existing local LLM initialization code...
    #         print("Initializing local Gemma3 model...")
            
    #         # Authenticate if token provided
    #         self.authenticate_huggingface()
            
    #         try:
    #             # Load model
    #             print(f"Loading model: {self.model_id}")
    #             print(f"Device: {self.device}")
                
    #             start_time = time.time()
    #             self.model = Gemma3ForCausalLM.from_pretrained(self.model_id).to(self.device).eval()
    #             model_load_time = time.time() - start_time
                
    #             # Load tokenizer
    #             start_time = time.time()
    #             self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    #             tokenizer_load_time = time.time() - start_time
                
    #             print(f"✓ Model loaded in {model_load_time:.2f} seconds")
    #             print(f"✓ Tokenizer loaded in {tokenizer_load_time:.2f} seconds")
                
    #             # Print device info
    #             if self.device == "cuda":
    #                 gpu_name = torch.cuda.get_device_name(0)
    #                 gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    #                 print(f"✓ Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    #             else:
    #                 print("✓ Using CPU")
                    
    #             return True
                
    #         except Exception as e:
    #             print(f"Error initializing local LLM: {e}")
    #             return False
    
    def text_to_speech_openai(self, text, output_filename):
        """Generate TTS audio using OpenAI"""
        try:
            with self.openai_client.audio.speech.with_streaming_response.create(
                model=self.openai_tts_model,
                voice=self.openai_tts_voice,
                input=text
            ) as response:
                response.stream_to_file(output_filename)
            return True
        except Exception as e:
            print(f"An error occurred during OpenAI TTS generation for '{text}': {e}")
            return False
    
    def validate_model_id(self, model_id):
        """Validate if model ID is accessible"""
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return True
        except Exception as e:
            print(f"Warning: Could not validate model '{model_id}': {e}")
            return False
      
    def get_openai_llm_response(self, prompt, max_retries=3, max_tokens=512, temperature=0.7):
        """Get response from OpenAI API"""
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"\nError on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"Failed to get response after {max_retries} attempts.")
                    return "[Translation Failed]"
        return "[Translation Failed]"
    
    def get_task_specific_params(self, task_type="translation"):
        """Get optimized parameters for different tasks with dynamic token lengths"""
        if task_type == "translation":
            return {
                "temperature": 0.3,    # Low for consistency
                "top_p": 0.85,         # Focused sampling
                "max_new_tokens": 128  # Reduced from 256 - most translations are short
            }
        elif task_type == "refinement":
            return {
                "temperature": 0.4,    # Slightly higher for creativity
                "top_p": 0.9,          # More diverse sampling
                "max_new_tokens": 256  # Moderate for refinements
            }
        elif task_type == "summary":
            return {
                "temperature": 0.2,    # Very low for factual summary
                "top_p": 0.8,          # Conservative sampling
                "max_new_tokens": 150  # Reduced from 256 - summaries should be concise
            }
        elif task_type == "overlap_refinement":
            return {
                "temperature": 0.35,   # Balanced for multi-line refinement
                "top_p": 0.88,         # Slightly more diverse
                "max_new_tokens": 512  # Longer for processing multiple lines
            }
        else:
            return {
                "temperature": self.local_llm_temperature,
                "top_p": self.local_llm_top_p,
                "max_new_tokens": self.local_llm_max_tokens
            }
    
    def steps_require_llm(self):
        """Return list of steps that require LLM initialization"""
        return [2, 3]  # Translation and refinement steps

    def check_llm_needed_for_pipeline(self, start_step):
        """Check if LLM will be needed for the pipeline starting from start_step"""
        llm_steps = self.steps_require_llm()
        next_required = self.get_next_required_step()
        actual_start = max(start_step, next_required)
        
        # Check if any LLM-requiring step will be executed
        for step in range(actual_start, 7):
            if step in llm_steps and not self.check_step_completion(step):
                return True
        return False
    
    # def get_llm_response(self, prompt, max_retries=3, max_new_tokens=None, temperature=None, top_p=None):
    #     """Get response from configured LLM (local or OpenAI)"""
    #     # Check if LLM is properly initialized
    #     if self.llm_type == "openai":
    #         if not self.openai_client:
    #             print("Error: OpenAI client not initialized")
    #             return "[Translation Failed - OpenAI client not initialized]"
    #         return self.get_openai_llm_response(prompt, max_retries, max_new_tokens or self.local_llm_max_tokens, temperature or 0.7)
    #     else:
    #         # Check local LLM initialization
    #         if not self.model or not self.tokenizer:
    #             print("Error: Local LLM model or tokenizer not initialized")
    #             return "[Translation Failed - Local LLM not initialized]"
            
    #         # Use instance parameters with fallbacks
    #         temperature = temperature if temperature is not None else self.local_llm_temperature
    #         top_p = top_p if top_p is not None else self.local_llm_top_p
    #         max_new_tokens = max_new_tokens if max_new_tokens is not None else self.local_llm_max_tokens
            
    #         for attempt in range(max_retries):
    #             try:
    # # def get_llm_response(self, prompt, max_retries=3, max_new_tokens=None, temperature=None, top_p=None):
    # #     """Get response from configured LLM (local or OpenAI)"""
    # #     if self.llm_type == "openai":
    # #         return self.get_openai_llm_response(prompt, max_retries, max_new_tokens or self.local_llm_max_tokens, temperature or 0.7)
    # #     else:
    # #         # Use instance parameters with fallbacks
    # #         temperature = temperature if temperature is not None else self.local_llm_temperature
    # #         top_p = top_p if top_p is not None else self.local_llm_top_p
    # #         max_new_tokens = max_new_tokens if max_new_tokens is not None else self.local_llm_max_tokens
            
    # #         for attempt in range(max_retries):
    # #             try:
    #                 # Construct messages for chat template
    #                 messages = [
    #                     {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #                 ]
                    
    #                 # Tokenize input
    #                 inputs = self.tokenizer.apply_chat_template(
    #                     messages, 
    #                     add_generation_prompt=True, 
    #                     tokenize=True, 
    #                     return_dict=True, 
    #                     return_tensors="pt"
    #                 ).to(self.device)
                    
    #                 # Generate response with improved parameters
    #                 with torch.inference_mode():
    #                     outputs = self.model.generate(
    #                         **inputs, 
    #                         max_new_tokens=max_new_tokens,
    #                         do_sample=self.local_llm_do_sample,
    #                         temperature=temperature,
    #                         top_p=top_p,
    #                         top_k=self.local_llm_top_k,
    #                         repetition_penalty=self.local_llm_repetition_penalty,
    #                         pad_token_id=self.tokenizer.eos_token_id,
    #                         eos_token_id=self.tokenizer.eos_token_id,
    #                         # Add these for better quality
    #                         no_repeat_ngram_size=3,  # Prevent repetitive phrases
    #                         early_stopping=True,      # Stop when EOS is generated
    #                         num_beams=1 if self.local_llm_do_sample else 3,  # Use beam search when not sampling
    #                     )
                    
    #                 # Decode the output (only the new tokens)
    #                 input_length = inputs.input_ids.shape[1]
    #                 response_tokens = outputs[0][input_length:]
    #                 response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    
    #                 return response.strip()
                    
    #             except Exception as e:
    #                 print(f"\nError on attempt {attempt + 1}: {e}")
    #                 if attempt < max_retries - 1:
    #                     time.sleep(1)
    #                 else:
    #                     print(f"Failed to get response after {max_retries} attempts.")
    #                     return "[Translation Failed]"
            
    #         return "[Translation Failed]"
    def get_llm_response(self, prompt, max_retries=3, max_new_tokens=None, temperature=None, top_p=None):
        """Get response from configured LLM (local or OpenAI)"""
        # Check if LLM is properly initialized
        if self.llm_type == "openai":
            if not self.openai_client:
                print("Error: OpenAI client not initialized")
                return "[Translation Failed - OpenAI client not initialized]"
            return self.get_openai_llm_response(prompt, max_retries, max_new_tokens or self.local_llm_max_tokens, temperature or 0.7)
        else:
            # Check local LLM initialization
            if not self.model or not self.tokenizer:
                print("Error: Local LLM model or tokenizer not initialized")
                return "[Translation Failed - Local LLM not initialized]"
            
            # Use instance parameters with fallbacks
            temperature = temperature if temperature is not None else self.local_llm_temperature
            top_p = top_p if top_p is not None else self.local_llm_top_p
            max_new_tokens = max_new_tokens if max_new_tokens is not None else self.local_llm_max_tokens
            
            for attempt in range(max_retries):
                try:
                    # Construct messages for chat template
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ]
                    
                    # Tokenize input
                    inputs = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        tokenize=True, 
                        return_dict=True, 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate response with improved parameters
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs, 
                            max_new_tokens=max_new_tokens,
                            do_sample=self.local_llm_do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=self.local_llm_top_k,
                            repetition_penalty=self.local_llm_repetition_penalty,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            # Add these for better quality
                            no_repeat_ngram_size=3,  # Prevent repetitive phrases
                            early_stopping=True,      # Stop when EOS is generated
                            num_beams=1 if self.local_llm_do_sample else 3,  # Use beam search when not sampling
                        )
                    
                    # Decode the output (only the new tokens)
                    input_length = inputs.input_ids.shape[1]
                    response_tokens = outputs[0][input_length:]
                    response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    
                    return response.strip()
                    
                except Exception as e:
                    print(f"\nError on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        print(f"Failed to get response after {max_retries} attempts.")
                        return "[Translation Failed]"
            
            return "[Translation Failed]"
    # def get_llm_response(self, prompt, max_retries=3, max_new_tokens=512, temperature=0.7, top_p=0.95):
    #     """Get response from configured LLM (local or OpenAI)"""
    #     if self.llm_type == "openai":
    #         return self.get_openai_llm_response(prompt, max_retries, max_new_tokens, temperature)
    #     else:
    #         # Existing local LLM code...
    #         for attempt in range(max_retries):
    #             try:
    #                 # Construct messages for chat template
    #                 messages = [
    #                     {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #                 ]
                    
    #                 # Tokenize input
    #                 inputs = self.tokenizer.apply_chat_template(
    #                     messages, 
    #                     add_generation_prompt=True, 
    #                     tokenize=True, 
    #                     return_dict=True, 
    #                     return_tensors="pt"
    #                 ).to(self.device)
                    
    #                 # Generate response
    #                 with torch.inference_mode():
    #                     outputs = self.model.generate(
    #                         **inputs, 
    #                         max_new_tokens=max_new_tokens, 
    #                         do_sample=True, 
    #                         top_p=top_p, 
    #                         temperature=temperature,
    #                         pad_token_id=self.tokenizer.eos_token_id
    #                     )
                    
    #                 # Decode the output (only the new tokens)
    #                 input_length = inputs.input_ids.shape[1]
    #                 response_tokens = outputs[0][input_length:]
    #                 response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    
    #                 return response.strip()
                    
    #             except Exception as e:
    #                 print(f"\nError on attempt {attempt + 1}: {e}")
    #                 if attempt < max_retries - 1:
    #                     time.sleep(0)
    #                 else:
    #                     print(f"Failed to get response after {max_retries} attempts.")
    #                     return "[Translation Failed]"
            
    #         return "[Translation Failed]"

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
        return f"""لطفا یک خلاصه مختصر و جامع از متن انگلیسی زیر ارائه دهید. این خلاصه برای ارائه زمینه کلی برای کار ترجمه استفاده خواهد شد.

                    متن:
                    {full_text}

                    خلاصه را به زبان فارسی و در حدود 2-3 جمله ارائه دهید."""

    def create_translation_prompt(self, current_text, previous_english, next_english, summary):
        """Create translation prompt with context"""
        surrounding_context = []
        if previous_english:
            surrounding_context.append(f"جمله انگلیسی قبلی:\n{previous_english}")
        if next_english:
            surrounding_context.append(f"جمله انگلیسی بعدی:\n{next_english}")
        surrounding_context_section = "\n\n".join(surrounding_context) if surrounding_context else "هیچ جمله اطراف وجود ندارد."

        return f"""شما یک مترجم خبره انگلیسی به فارسی برای زیرنویس هستید. وظیفه شما ترجمه "متن فعلی برای ترجمه" با حفظ انسجام با زمینه ارائه شده است.

                ---
                زمینه 1: خلاصه کلی
                {summary}
                ---
                زمینه 2: جملات انگلیسی اطراف
                {surrounding_context_section}
                ---

                متن فعلی برای ترجمه:
                {current_text}

                دستورالعمل:
                1. "متن فعلی برای ترجمه" را به فارسی طبیعی و روان ترجمه کنید.
                2. ترجمه را مختصر و برای زیرنویس قابل خواندن نگه دارید.
                3. فقط ترجمه فارسی را برگردانید. هیچ متن، برچسب یا توضیح اضافی ندهید."""

    # def generate_summary(self, segments):
    #     """Generate summary of transcript for context"""
    #     print("Generating transcript summary...")
    #     full_text = ' '.join([segment['text'] for segment in segments])
    #     summary_prompt = self.create_summary_prompt(full_text[:8000])  # Limit context size
    #     self.summary = self.get_llm_response(summary_prompt, max_new_tokens=256)
    #     if not self.summary or "failed" in self.summary.lower():
    #         print("Warning: Could not generate a summary. Proceeding without it.")
    #         self.summary = "خلاصه‌ای در دسترس نیست."
    #     else:
    #         print(f"Summary generated: {self.summary[:120]}...")
    #     return self.summary
    
    # def generate_summary(self, segments):
    #     """Generate summary of transcript for context with optimized parameters"""
    #     print("Generating transcript summary...")
    #     full_text = ' '.join([segment['text'] for segment in segments])
    #     summary_prompt = self.create_summary_prompt(full_text[:8000])
        
    #     # Use summary-specific parameters
    #     summary_params = self.get_task_specific_params("summary")
    #     self.summary = self.get_llm_response(summary_prompt, **summary_params)
        
    #     if not self.summary or "failed" in self.summary.lower():
    #         print("Warning: Could not generate a summary. Proceeding without it.")
    #         self.summary = "خلاصه‌ای در دسترس نیست."
    #     else:
    #         print(f"Summary generated: {self.summary[:120]}...")
    #     return self.summary
    
    def generate_summary(self, segments):
        """Generate summary of transcript for context"""
        print("Generating transcript summary...")
        
        # Ensure LLM is initialized
        if not self.ensure_llm_initialized():
            print("Warning: Could not initialize LLM for summary generation. Proceeding without summary.")
            self.summary = "خلاصه‌ای در دسترس نیست."
            return self.summary
        
        full_text = ' '.join([segment['text'] for segment in segments])
        summary_prompt = self.create_summary_prompt(full_text[:8000])
        
        # Get summary-specific parameters
        summary_params = self.get_task_specific_params("summary")
        self.summary = self.get_llm_response(summary_prompt, **summary_params)
        
        if not self.summary or "failed" in self.summary.lower():
            print("Warning: Could not generate a summary. Proceeding without it.")
            self.summary = "خلاصه‌ای در دسترس نیست."
        else:
            print(f"Summary generated: {self.summary[:120]}...")
        return self.summary

    def translate_segments(self, segments):
        """Translate segments to Persian with optimized parameters"""
        if not segments:
            return []

        translated_segments = []
        print(f"Translating {len(segments)} segments with optimized parameters...")
        
        # Get translation-specific parameters
        translation_params = self.get_task_specific_params("translation")
        print(f"Using translation parameters: temp={translation_params['temperature']}, top_p={translation_params['top_p']}")
        
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
            
            # Use task-specific parameters
            persian_text = self.get_llm_response(
                translation_prompt, 
                **translation_params
            )

            translated_segments.append({**current_segment, 'text': persian_text})
            time.sleep(0)

        return translated_segments
    
    # def translate_segments(self, segments):
    #     """Translate segments to Persian"""
    #     if not segments:
    #         return []

    #     translated_segments = []
    #     print(f"Translating {len(segments)} segments...")
        
    #     for i in tqdm(range(len(segments)), desc="Translating"):
    #         current_segment = segments[i]

    #         previous_english_text = segments[i-1]['text'] if i > 0 else ""
    #         next_english_text = segments[i+1]['text'] if i < len(segments)-1 else ""

    #         translation_prompt = self.create_translation_prompt(
    #             current_segment['text'],
    #             previous_english_text,
    #             next_english_text,
    #             self.summary
    #         )
    #         persian_text = self.get_llm_response(translation_prompt, max_new_tokens=256)

    #         translated_segments.append({**current_segment, 'text': persian_text})
    #         time.sleep(0)  # Small delay to prevent overheating

    #     return translated_segments

    def write_srt_file(self, segments, output_path):
        """Write segments to SRT file"""
        with open(output_path, 'w', encoding='utf-8') as file:
            for segment in segments:
                file.write(f"{segment['sequence']}\n")
                file.write(f"{segment['start']} --> {segment['end']}\n")
                file.write(f"{segment['text']}\n\n")

    def translate_to_persian(self):
        """Step 2: Translate English SRT to Persian"""
        print(f"Step 2: Translating to Persian...")
        
        if not self.initialize_local_llm():
            print("Error: Failed to initialize local LLM")
            return False

        segments = self.parse_srt_file(self.srt_original)
        if not segments:
            print("No valid subtitle segments found.")
            return False
        
        print(f"Found {len(segments)} subtitle segments.")
        self.generate_summary(segments)
        translated_segments = self.translate_segments(segments)
        self.write_srt_file(translated_segments, self.srt_persian)
        
        print(f"Persian translation saved: {self.srt_persian}")
        return True

    # PART 3: Refinement
    def create_refinement_prompt(self, srt_line, original_prev, original_next, refined_prev):
        """Create refinement prompt for Stage 1 processing"""
        local_context_parts = []
        if original_prev:
            local_context_parts.append(f"خط اصلی قبلی:\n{original_prev}")
        if original_next:
            local_context_parts.append(f"خط اصلی بعدی:\n{original_next}")
        local_context_section = "\n\n".join(local_context_parts) if local_context_parts else "هیچ خط اطراف در SRT اصلی وجود ندارد."

        refined_history_section = refined_prev if refined_prev else "این اولین خطی است که اصلاح می‌شود."

        return f"""شما یک متخصص زبان فارسی هستید که در ویرایش زیرنویس برای بهبود روانی تخصص دارید. خط زیرنویس زیر را بر اساس زمینه ارائه شده اصلاح کنید:

            ---
            زمینه 1: محلی (زیرنویس‌های اصلی اطراف)
            {local_context_section}
            ---
            زمینه 2: تاریخچه (خط قبلی که تازه اصلاح کردید)
            خط قبلی اصلاح شده: {refined_history_section}
            ---

            خط برای اصلاح:
            {srt_line}
            ---

            دستورالعمل:
            1. "خط برای اصلاح" را بازنویسی کنید تا طبیعی، دستوری صحیح و منسجم با "خط قبلی اصلاح شده" باشد، بدون حذف هیچ اطلاعاتی. طول باید تقریباً مشابه اصل باقی بماند.
            2. اطمینان حاصل کنید که خط اصلاح شده با زمینه اطراف به خوبی جریان دارد.
            3. **مهم**: مدت زمان یا تعداد کاراکتر را به طور قابل توجهی تغییر ندهید - زمان‌بندی و خوانایی زیرنویس را حفظ کنید.
            4. فقط خط فارسی اصلاح شده را برگردانید، بدون هیچ نظر اضافی.
            5. اگر قطعه جمله‌ای متعلق به خط زیرنویس بعدی است، آن را به تناسب منتقل کنید تا کاملی معنایی و دقت دستوری حفظ شود.
            6. علائم نگارشی نظیر ! ، ... ، . و غیره را حذف کن"""

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

    def get_llm_response_stage2(self, prompt, max_retries=5):
        """Get response from Stage 2 LLM (same model for local setup)"""
        return self.get_llm_response(prompt, max_retries=max_retries, max_new_tokens=1024)

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
                                time.sleep(0)
                    else:
                        print(f"⚠ Retry {retry+1}/{max_chunk_retries}: LLM response failed")
                        if retry < max_chunk_retries - 1:
                            time.sleep(0)
                
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
                time.sleep(0)  # Small delay to prevent overheating
    
        # Update the segments with final refined texts
        final_segments = []
        for i, segment in enumerate(segments):
            final_segments.append({**segment, 'text': refined_texts[i]})
    
        return final_segments

    def refine_persian_srt(self):
        """Step 3: Advanced Two-Stage Persian SRT Refinement"""
        print(f"Step 3: Advanced refinement of Persian translation...")
        
        # Ensure LLM is initialized before refinement
        if not self.ensure_llm_initialized():
            print("Error: Failed to initialize LLM for refinement")
            return False
        
        segments = self.parse_srt_file(self.srt_persian)
        if not segments:
            print("No valid segments found in Persian SRT file.")
            return False
    
    # def refine_persian_srt(self):
    #     """Step 3: Advanced Two-Stage Persian SRT Refinement with optimized parameters"""
    #     print(f"Step 3: Advanced refinement of Persian translation...")
        
    #     segments = self.parse_srt_file(self.srt_persian)
    #     if not segments:
    #         print("No valid segments found in Persian SRT file.")
    #         return False
        
        print(f"Found {len(segments)} segments to refine.")
        
        # Get refinement-specific parameters
        refinement_params = self.get_task_specific_params("refinement")
        print(f"Using refinement parameters: temp={refinement_params['temperature']}, top_p={refinement_params['top_p']}")
        
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
                
                # Use refinement-specific parameters
                refined_text = self.get_llm_response(prompt, **refinement_params)
                time.sleep(0)
            
            stage1_segments.append({**segment, 'text': refined_text})
            refined_history.append(refined_text)
    
    # ... rest of the method remains the same
    
    # def refine_persian_srt(self):
    #     """Step 3: Advanced Two-Stage Persian SRT Refinement"""
    #     print(f"Step 3: Advanced refinement of Persian translation...")
        
    #     segments = self.parse_srt_file(self.srt_persian)
    #     if not segments:
    #         print("No valid segments found in Persian SRT file.")
    #         return False
        
    #     print(f"Found {len(segments)} segments to refine.")
        
    #     # Stage 1: Sequential refinement with context
    #     print("Starting Stage 1: Sequential refinement with local context...")
    #     stage1_segments = []
    #     refined_history = []
        
    #     for i, segment in enumerate(tqdm(segments, desc="Stage 1: Sequential refinement")):
    #         if not segment['text'].strip():
    #             refined_text = ""
    #         else:
    #             # Get context from surrounding segments
    #             original_prev = segments[i-1]['text'] if i > 0 else None
    #             original_next = segments[i+1]['text'] if i < len(segments) - 1 else None
    #             refined_prev = refined_history[-1] if refined_history else None

    #             prompt = self.create_refinement_prompt(
    #                 segment['text'], original_prev, original_next, refined_prev
    #             )
    #             refined_text = self.get_llm_response(prompt, max_new_tokens=256)
    #             time.sleep(0)  # Small delay
            
    #         stage1_segments.append({**segment, 'text': refined_text})
    #         refined_history.append(refined_text)

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
        print(f"Model used: {self.model_id} (Local Gemma3)")
        print(f"Device: {self.device}")
        print(f"Final output: {self.srt_refined}")
        print("="*50)
        
        return True

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


    # PART 4: TTS Audio Generation
    def initialize_tts(self):
        """Initialize TTS engine (local or OpenAI)"""
        if self.tts_type == "openai":
            return self.initialize_openai_client()
        else:
            # Existing local TTS initialization code...
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
        """Generate TTS audio using configured engine"""
        if self.tts_type == "openai":
            return self.text_to_speech_openai(text, output_filename)
        else:
            # Existing local TTS code...
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
            # 'api_keys_used': len(self.api_keys)
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"Processing report saved to: {report_path}")
        except Exception as e:
            print(f"Warning: Could not save processing report: {e}")
    
    def run_full_pipeline(self):
        """Run the complete dubbing pipeline with step control"""
        print("=" * 60)
        print("UNIFIED VIDEO DUBBING PIPELINE")
        print("=" * 60)
        
        # Determine actual starting step
        next_required_step = self.get_next_required_step()
        actual_start_step = max(self.start_step, next_required_step)
        
        # Pre-check if LLM will be needed
        if self.check_llm_needed_for_pipeline(actual_start_step):
            print("🔧 LLM will be needed for this pipeline. Pre-checking initialization...")
            if not self.ensure_llm_initialized():
                print("❌ Failed to initialize LLM. Cannot proceed.")
                return False
            print("✅ LLM pre-check completed successfully")
        
        # ... rest of the method
    
    # def run_full_pipeline(self):
        # """Run the complete dubbing pipeline with step control"""
        # print("=" * 60)
        # print("UNIFIED VIDEO DUBBING PIPELINE")
        # print("=" * 60)
        
        # # Determine actual starting step
        # next_required_step = self.get_next_required_step()
        # actual_start_step = max(self.start_step, next_required_step)
        
        if actual_start_step > self.start_step:
            print(f"📋 Requested start step: {self.start_step}")
            print(f"🔍 Next required step: {actual_start_step}")
            print(f"✅ Some steps already completed, starting from step {actual_start_step}")
        else:
            print(f"🚀 Starting from step {actual_start_step} as requested")
        
        # Show completion status
        for step in range(1, 7):
            status = "✅ COMPLETED" if self.check_step_completion(step) else "⏳ PENDING"
            step_names = {
                1: "Video transcription",
                2: "Persian translation", 
                3: "Persian refinement",
                4: "TTS audio generation",
                5: "Timing adjustment",
                6: "Final video creation"
            }
            print(f"Step {step}: {step_names[step]} - {status}")
        
        print("-" * 60)
        
        # Initialize LLM early if starting from step 3 or later and using LLM
        if actual_start_step >= 3 and actual_start_step <= 4:
            print("🔧 Pre-initializing LLM for refinement steps...")
            if not self.ensure_llm_initialized():
                print("❌ Failed to initialize LLM. Cannot proceed with refinement.")
                return False
            print("✅ LLM pre-initialization completed")
        
        pipeline_start = time.time()
        
        # Step 1: Transcribe video
        if actual_start_step <= 1:
            if not self.transcribe_video():
                print("❌ Step 1 failed: Video transcription")
                return False
            print("✅ Step 1 completed: Video transcription")
            
            # Ask user to approve with timeout
            if not self.timeout_prompt_with_approval(
                f"Step 1 Complete: Please check the transcription quality",
                self.srt_original
            ):
                return False
        else:
            print("⏭️  Step 1: Already completed (skipped)")
        
        # Step 2: Translate to Persian
        if actual_start_step <= 2:
            if not self.translate_to_persian():
                print("❌ Step 2 failed: Persian translation")
                return False
            print("✅ Step 2 completed: Persian translation")
            
            # Ask user to approve with timeout
            if not self.timeout_prompt_with_approval(
                f"Step 2 Complete: Please check the Persian translation quality",
                self.srt_persian
            ):
                return False
        else:
            print("⏭️  Step 2: Already completed (skipped)")
        
        # Step 3: Refine Persian translation
        if actual_start_step <= 3:
            if not self.refine_persian_srt():
                print("❌ Step 3 failed: Persian refinement")
                return False
            print("✅ Step 3 completed: Persian refinement")
            
            # Ask user to approve with timeout
            if not self.timeout_prompt_with_approval(
                f"Step 3 Complete: Please check the refined Persian translation",
                self.srt_refined
            ):
                return False
        else:
            print("⏭️  Step 3: Already completed (skipped)")
    
    # def run_full_pipeline(self):
    #     """Run the complete dubbing pipeline with step control"""
    #     print("=" * 60)
    #     print("UNIFIED VIDEO DUBBING PIPELINE")
    #     print("=" * 60)
        
    #     # Determine actual starting step
    #     next_required_step = self.get_next_required_step()
    #     actual_start_step = max(self.start_step, next_required_step)
        
    #     if actual_start_step > self.start_step:
    #         print(f"📋 Requested start step: {self.start_step}")
    #         print(f"🔍 Next required step: {actual_start_step}")
    #         print(f"✅ Some steps already completed, starting from step {actual_start_step}")
    #     else:
    #         print(f"🚀 Starting from step {actual_start_step} as requested")
        
    #     # Show completion status
    #     for step in range(1, 7):
    #         status = "✅ COMPLETED" if self.check_step_completion(step) else "⏳ PENDING"
    #         step_names = {
    #             1: "Video transcription",
    #             2: "Persian translation", 
    #             3: "Persian refinement",
    #             4: "TTS audio generation",
    #             5: "Timing adjustment",
    #             6: "Final video creation"
    #         }
    #         print(f"Step {step}: {step_names[step]} - {status}")
        
    #     print("-" * 60)
        
    #     pipeline_start = time.time()
        
    #     # Step 1: Transcribe video
    #     if actual_start_step <= 1:
    #         if not self.transcribe_video():
    #             print("❌ Step 1 failed: Video transcription")
    #             return False
    #         print("✅ Step 1 completed: Video transcription")
            
    #         # Ask user to approve with timeout
    #         if not self.timeout_prompt_with_approval(
    #             f"Step 1 Complete: Please check the transcription quality",
    #             self.srt_original
    #         ):
    #             return False
    #     else:
    #         print("⏭️  Step 1: Already completed (skipped)")
        
    #     # Step 2: Translate to Persian
    #     if actual_start_step <= 2:
    #         if not self.translate_to_persian():
    #             print("❌ Step 2 failed: Persian translation")
    #             return False
    #         print("✅ Step 2 completed: Persian translation")
            
    #         # Ask user to approve with timeout
    #         if not self.timeout_prompt_with_approval(
    #             f"Step 2 Complete: Please check the Persian translation quality",
    #             self.srt_persian
    #         ):
    #             return False
    #     else:
    #         print("⏭️  Step 2: Already completed (skipped)")
        
    #     # Step 3: Refine Persian translation
    #     if actual_start_step <= 3:
    #         if not self.refine_persian_srt():
    #             print("❌ Step 3 failed: Persian refinement")
    #             return False
    #         print("✅ Step 3 completed: Persian refinement")
            
    #         # Ask user to approve with timeout
    #         if not self.timeout_prompt_with_approval(
    #             f"Step 3 Complete: Please check the refined Persian translation",
    #             self.srt_refined
    #         ):
    #             return False
    #     else:
    #         print("⏭️  Step 3: Already completed (skipped)")
        
        # Step 4: Generate TTS audio
        if actual_start_step <= 4:
            if not self.generate_tts_audio():
                print("❌ Step 4 failed: TTS audio generation")
                return False
            print("✅ Step 4 completed: TTS audio generation")
            
            # Ask user to approve with timeout
            if not self.timeout_prompt_with_approval(
                f"Step 4 Complete: Please check the generated audio files",
                self.audio_dir
            ):
                return False
        else:
            print("⏭️  Step 4: Already completed (skipped)")
        
        # Step 5: Adjust timing
        if actual_start_step <= 5:
            if not self.adjust_srt_timing():
                print("❌ Step 5 failed: Timing adjustment")
                return False
            print("✅ Step 5 completed: Timing adjustment")
            
            # Ask user to approve with timeout
            if not self.timeout_prompt_with_approval(
                f"Step 5 Complete: Please check the timing adjustments",
                self.srt_adjusted
            ):
                return False
        else:
            print("⏭️  Step 5: Already completed (skipped)")
        
        # Step 6: Create final video
        if actual_start_step <= 6:
            if not self.create_final_dubbed_video():
                print("❌ Step 6 failed: Final video creation")
                return False
            print("✅ Step 6 completed: Final video creation")
        else:
            print("⏭️  Step 6: Already completed (skipped)")
        
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

# def main():
#     """Main function with step control, OpenAI support, and local LLM parameters"""
#     if len(sys.argv) < 3:
#         print("Usage: python video_dubber.py input_video.mp4 output_video.mp4 [start_step] [options]")
#         print("\nOptional start_step parameter:")
#         print("  1 - Start from transcription (default)")
#         print("  2 - Start from Persian translation")
#         print("  3 - Start from Persian refinement")
#         print("  4 - Start from TTS audio generation")
#         print("  5 - Start from timing adjustment")
#         print("  6 - Start from final video creation")
#         print("\nNote: The system will automatically detect completed steps")
#         print("and start from the next required step if it's later than requested.")
#         print("\nLLM Options:")
#         print("  --llm-type local    - Use local Gemma3 model (default)")
#         print("  --llm-type openai   - Use OpenAI GPT API")
#         print("  --openai-api-key KEY - OpenAI API key (required for OpenAI services)")
#         print("\nLocal LLM Quality Parameters:")
#         print("  --local-temp FLOAT         Temperature (0.1-1.0, default: 0.3)")
#         print("  --local-top-p FLOAT        Top-p sampling (0.1-1.0, default: 0.85)")
#         print("  --local-top-k INT          Top-k sampling (1-100, default: 40)")
#         print("  --local-max-tokens INT     Max tokens per response (default: 512)")
#         print("  --local-rep-penalty FLOAT Repetition penalty (1.0-1.3, default: 1.1)")
#         print("\nLocal LLM Preset configurations:")
#         print("  --preset conservative      temp=0.2, top_p=0.8, top_k=30 (most consistent)")
#         print("  --preset balanced          temp=0.3, top_p=0.85, top_k=40 (default)")
#         print("  --preset creative          temp=0.5, top_p=0.9, top_k=50 (more varied)")
#         print("\nTTS Options:")
#         print("  --tts-type local    - Use local Persian TTS (default)")
#         print("  --tts-type openai   - Use OpenAI TTS API")
#         print("  --openai-tts-voice VOICE - Voice for OpenAI TTS (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse)")
#         print("  --openai-tts-model MODEL - Model for OpenAI TTS (tts-1, tts-1-hd, gpt-4o-mini-tts)")
#         print("\nRequired setup:")
#         print("1. For local LLM: HuggingFace token may be needed for some models")
#         print("2. For local TTS: Download Persian TTS model to './vits-piper-fa_IR-amir-medium/'")
#         print("3. For OpenAI services: Provide --openai-api-key")
#         print("4. Install required packages:")
#         print("   pip install openai-whisper transformers torch huggingface_hub sherpa-onnx soundfile openai")
#         print("5. Install FFmpeg")
#         sys.exit(1)
    
#     # Parse basic arguments
#     input_video = sys.argv[1]
#     output_video = sys.argv[2]
    
#     # Initialize default values
#     start_step = 1
#     llm_type = "local"
#     tts_type = "local"
#     openai_api_key = 
#       openai_tts_voice = "alloy"
#     openai_tts_model = "gpt-4o-mini-tts"
    
#     # Local LLM parameters with defaults
#     local_llm_temperature = 0.3
#     local_llm_top_p = 0.85
#     local_llm_top_k = 40
#     local_llm_max_tokens = 512
#     local_llm_repetition_penalty = 1.1
#     local_llm_do_sample = True
    
#     # Parse arguments starting from index 3
#     i = 3
#     while i < len(sys.argv):
#         arg = sys.argv[i]
        
#         # Check if it's a start_step (single digit without dashes)
#         if arg.isdigit() and len(arg) == 1:
#             start_step = int(arg)
#             i += 1
#         elif arg == "--llm-type" and i + 1 < len(sys.argv):
#             llm_type = sys.argv[i + 1]
#             if llm_type not in ["local", "openai"]:
#                 print("Error: --llm-type must be 'local' or 'openai'")
#                 sys.exit(1)
#             i += 2
#         elif arg == "--tts-type" and i + 1 < len(sys.argv):
#             tts_type = sys.argv[i + 1]
#             if tts_type not in ["local", "openai"]:
#                 print("Error: --tts-type must be 'local' or 'openai'")
#                 sys.exit(1)
#             i += 2
#         elif arg == "--openai-api-key" and i + 1 < len(sys.argv):
#             openai_api_key = sys.argv[i + 1]
#             i += 2
#         elif arg == "--openai-tts-voice" and i + 1 < len(sys.argv):
#             openai_tts_voice = sys.argv[i + 1]
#             valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]
#             if openai_tts_voice not in valid_voices:
#                 print(f"Error: --openai-tts-voice must be one of: {', '.join(valid_voices)}")
#                 sys.exit(1)
#             i += 2
#         elif arg == "--openai-tts-model" and i + 1 < len(sys.argv):
#             openai_tts_model = sys.argv[i + 1]
#             valid_tts_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
#             if openai_tts_model not in valid_tts_models:
#                 print(f"Error: --openai-tts-model must be one of: {', '.join(valid_tts_models)}")
#                 sys.exit(1)
#             i += 2
#         # Local LLM preset configurations
#         elif arg == "--preset" and i + 1 < len(sys.argv):
#             preset = sys.argv[i + 1]
#             if preset == "conservative":
#                 local_llm_temperature = 0.2
#                 local_llm_top_p = 0.8
#                 local_llm_top_k = 30
#                 local_llm_repetition_penalty = 1.15
#                 print(f"Applied conservative preset: temp=0.2, top_p=0.8, top_k=30, rep_penalty=1.15")
#             elif preset == "balanced":
#                 local_llm_temperature = 0.3
#                 local_llm_top_p = 0.85
#                 local_llm_top_k = 40
#                 local_llm_repetition_penalty = 1.1
#                 print(f"Applied balanced preset: temp=0.3, top_p=0.85, top_k=40, rep_penalty=1.1")
#             elif preset == "creative":
#                 local_llm_temperature = 0.5
#                 local_llm_top_p = 0.9
#                 local_llm_top_k = 50
#                 local_llm_repetition_penalty = 1.05
#                 print(f"Applied creative preset: temp=0.5, top_p=0.9, top_k=50, rep_penalty=1.05")
#             else:
#                 print(f"Error: --preset must be 'conservative', 'balanced', or 'creative'")
#                 sys.exit(1)
#             i += 2
#         # Individual local LLM parameters
#         elif arg == "--local-temp" and i + 1 < len(sys.argv):
#             try:
#                 local_llm_temperature = float(sys.argv[i + 1])
#                 if not (0.1 <= local_llm_temperature <= 1.0):
#                     print("Error: --local-temp must be between 0.1 and 1.0")
#                     sys.exit(1)
#             except ValueError:
#                 print("Error: --local-temp must be a valid float")
#                 sys.exit(1)
#             i += 2
#         elif arg == "--local-top-p" and i + 1 < len(sys.argv):
#             try:
#                 local_llm_top_p = float(sys.argv[i + 1])
#                 if not (0.1 <= local_llm_top_p <= 1.0):
#                     print("Error: --local-top-p must be between 0.1 and 1.0")
#                     sys.exit(1)
#             except ValueError:
#                 print("Error: --local-top-p must be a valid float")
#                 sys.exit(1)
#             i += 2
#         elif arg == "--local-top-k" and i + 1 < len(sys.argv):
#             try:
#                 local_llm_top_k = int(sys.argv[i + 1])
#                 if not (1 <= local_llm_top_k <= 100):
#                     print("Error: --local-top-k must be between 1 and 100")
#                     sys.exit(1)
#             except ValueError:
#                 print("Error: --local-top-k must be a valid integer")
#                 sys.exit(1)
#             i += 2
#         elif arg == "--local-max-tokens" and i + 1 < len(sys.argv):
#             try:
#                 local_llm_max_tokens = int(sys.argv[i + 1])
#                 if not (50 <= local_llm_max_tokens <= 2048):
#                     print("Error: --local-max-tokens must be between 50 and 2048")
#                     sys.exit(1)
#             except ValueError:
#                 print("Error: --local-max-tokens must be a valid integer")
#                 sys.exit(1)
#             i += 2
#         elif arg == "--local-rep-penalty" and i + 1 < len(sys.argv):
#             try:
#                 local_llm_repetition_penalty = float(sys.argv[i + 1])
#                 if not (1.0 <= local_llm_repetition_penalty <= 1.5):
#                     print("Error: --local-rep-penalty must be between 1.0 and 1.5")
#                     sys.exit(1)
#             except ValueError:
#                 print("Error: --local-rep-penalty must be a valid float")
#                 sys.exit(1)
#             i += 2
#         else:
#             # Try to parse as start_step if it's a digit
#             if arg.isdigit():
#                 start_step = int(arg)
#             else:
#                 print(f"Warning: Unknown argument '{arg}', ignoring...")
#             i += 1
    
#     # Validate start step
#     if start_step < 1 or start_step > 6:
#         print("Error: start_step must be between 1 and 6")
#         sys.exit(1)
    
#     # Validate OpenAI requirements
#     if (llm_type == "openai" or tts_type == "openai") and not openai_api_key:
#         print("Error: OpenAI API key required when using OpenAI services")
#         print("Provide --openai-api-key YOUR_API_KEY")
#         sys.exit(1)
    
#     # Validate OpenAI library availability
#     if (llm_type == "openai" or tts_type == "openai") and not OPENAI_AVAILABLE:
#         print("Error: OpenAI library not installed but required for OpenAI services")
#         print("Install with: pip install openai")
#         sys.exit(1)
    
#     # Validate input file
#     if not os.path.exists(input_video):
#         print(f"Error: Input video file '{input_video}' not found")
#         sys.exit(1)
    
#     # Check if output directory exists
#     output_dir = os.path.dirname(os.path.abspath(output_video))
#     if not os.path.exists(output_dir):
#         print(f"Error: Output directory '{output_dir}' does not exist")
#         sys.exit(1)
    
#     # Display configuration
#     print("=" * 60)
#     print("CONFIGURATION")
#     print("=" * 60)
#     print(f"Input video: {input_video}")
#     print(f"Output video: {output_video}")
#     print(f"Start step: {start_step}")
#     print(f"LLM type: {llm_type}")
#     print(f"TTS type: {tts_type}")
    
#     if llm_type == "local":
#         print(f"Local LLM parameters:")
#         print(f"  Temperature: {local_llm_temperature}")
#         print(f"  Top-p: {local_llm_top_p}")
#         print(f"  Top-k: {local_llm_top_k}")
#         print(f"  Max tokens: {local_llm_max_tokens}")
#         print(f"  Repetition penalty: {local_llm_repetition_penalty}")
#         print(f"  Do sample: {local_llm_do_sample}")
    
#     if openai_api_key:
#         print(f"OpenAI API key: {'*' * (len(openai_api_key) - 4)}{openai_api_key[-4:]}")
    
#     if tts_type == "openai":
#         print(f"OpenAI TTS voice: {openai_tts_voice}")
#         print(f"OpenAI TTS model: {openai_tts_model}")
    
#     print("=" * 60)
    
#     # Initialize dubber with all parameters
#     dubber = VideoDubber(
#         input_video, 
#         output_video, 
#         start_step=start_step,
#         llm_type=llm_type,
#         openai_api_key=openai_api_key,
#         tts_type=tts_type,
#         openai_tts_voice=openai_tts_voice,
#         openai_tts_model=openai_tts_model,
#         # Local LLM parameters
#         local_llm_temperature=local_llm_temperature,
#         local_llm_top_p=local_llm_top_p,
#         local_llm_top_k=local_llm_top_k,
#         local_llm_max_tokens=local_llm_max_tokens,
#         local_llm_repetition_penalty=local_llm_repetition_penalty,
#         local_llm_do_sample=local_llm_do_sample
#     )

def main():
    """Main function with step control, OpenAI support, and local LLM parameters"""
    if len(sys.argv) < 3:
        print("Usage: python video_dubber.py input_video.mp4 output_video.mp4 [start_step] [options]")
        print("\nOptional start_step parameter:")
        print("  1 - Start from transcription (default)")
        print("  2 - Start from Persian translation")
        print("  3 - Start from Persian refinement")
        print("  4 - Start from TTS audio generation")
        print("  5 - Start from timing adjustment")
        print("  6 - Start from final video creation")
        print("\nNote: The system will automatically detect completed steps")
        print("and start from the next required step if it's later than requested.")
        print("\nLLM Options:")
        print("  --llm-type local    - Use local Gemma3 model (default)")
        print("  --llm-type openai   - Use OpenAI GPT API")
        print("  --openai-api-key KEY - OpenAI API key (required for OpenAI services)")
        print("\nLocal LLM Quality Parameters:")
        print("  --local-temp FLOAT         Temperature (0.1-1.0, default: 0.3)")
        print("  --local-top-p FLOAT        Top-p sampling (0.1-1.0, default: 0.85)")
        print("  --local-top-k INT          Top-k sampling (1-100, default: 40)")
        print("  --local-max-tokens INT     Max tokens per response (default: 512)")
        print("  --local-rep-penalty FLOAT Repetition penalty (1.0-1.3, default: 1.1)")
        print("\nLocal LLM Preset configurations:")
        print("  --preset conservative      temp=0.2, top_p=0.8, top_k=30 (most consistent)")
        print("  --preset balanced          temp=0.3, top_p=0.85, top_k=40 (default)")
        print("  --preset creative          temp=0.5, top_p=0.9, top_k=50 (more varied)")
        print("\nTTS Options:")
        print("  --tts-type local    - Use local Persian TTS (default)")
        print("  --tts-type openai   - Use OpenAI TTS API")
        print("  --openai-tts-voice VOICE - Voice for OpenAI TTS (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse)")
        print("  --openai-tts-model MODEL - Model for OpenAI TTS (tts-1, tts-1-hd, gpt-4o-mini-tts)")
        print("\nRequired setup:")
        print("1. For local LLM: HuggingFace token may be needed for some models")
        print("2. For local TTS: Download Persian TTS model to './vits-piper-fa_IR-amir-medium/'")
        print("3. For OpenAI services: Provide --openai-api-key")
        print("4. Install required packages:")
        print("   pip install openai-whisper transformers torch huggingface_hub sherpa-onnx soundfile openai librosa")
        print("5. Install FFmpeg")
        print("\nExample usage:")
        print("  # Basic usage with local models")
        print("  python video_dubber.py video.mp4 output.mp4")
        print("  # Conservative local LLM settings")
        print("  python video_dubber.py video.mp4 output.mp4 --preset conservative")
        print("  # OpenAI for both LLM and TTS")
        print("  python video_dubber.py video.mp4 output.mp4 --llm-type openai --tts-type openai --openai-api-key YOUR_KEY")
        print("  # Start from step 3 with custom parameters")
        print("  python video_dubber.py video.mp4 output.mp4 3 --local-temp 0.2 --local-top-p 0.8")
        sys.exit(1)
    
    # Parse basic arguments
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    
    # Initialize default values
    start_step = 1
    llm_type = "local"
    tts_type = "local"
    openai_api_key = ""
    openai_tts_voice = "alloy"
    openai_tts_model = "gpt-4o-mini-tts"
    
    # Local LLM parameters with defaults
    local_llm_temperature = 0.3
    local_llm_top_p = 0.85
    local_llm_top_k = 40
    local_llm_max_tokens = 512
    local_llm_repetition_penalty = 1.1
    local_llm_do_sample = True
    
    # Parse arguments starting from index 3
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Check if it's a start_step (single digit without dashes)
        if arg.isdigit() and len(arg) == 1:
            start_step = int(arg)
            i += 1
        elif arg == "--llm-type" and i + 1 < len(sys.argv):
            llm_type = sys.argv[i + 1]
            if llm_type not in ["local", "openai"]:
                print("Error: --llm-type must be 'local' or 'openai'")
                sys.exit(1)
            i += 2
        elif arg == "--tts-type" and i + 1 < len(sys.argv):
            tts_type = sys.argv[i + 1]
            if tts_type not in ["local", "openai"]:
                print("Error: --tts-type must be 'local' or 'openai'")
                sys.exit(1)
            i += 2
        elif arg == "--openai-api-key" and i + 1 < len(sys.argv):
            openai_api_key = sys.argv[i + 1]
            i += 2
        elif arg == "--openai-tts-voice" and i + 1 < len(sys.argv):
            openai_tts_voice = sys.argv[i + 1]
            valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]
            if openai_tts_voice not in valid_voices:
                print(f"Error: --openai-tts-voice must be one of: {', '.join(valid_voices)}")
                sys.exit(1)
            i += 2
        elif arg == "--openai-tts-model" and i + 1 < len(sys.argv):
            openai_tts_model = sys.argv[i + 1]
            valid_tts_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
            if openai_tts_model not in valid_tts_models:
                print(f"Error: --openai-tts-model must be one of: {', '.join(valid_tts_models)}")
                sys.exit(1)
            i += 2
        # Local LLM preset configurations
        elif arg == "--preset" and i + 1 < len(sys.argv):
            preset = sys.argv[i + 1]
            if preset == "conservative":
                local_llm_temperature = 0.2
                local_llm_top_p = 0.8
                local_llm_top_k = 30
                local_llm_repetition_penalty = 1.15
                print(f"Applied conservative preset: temp=0.2, top_p=0.8, top_k=30, rep_penalty=1.15")
            elif preset == "balanced":
                local_llm_temperature = 0.3
                local_llm_top_p = 0.85
                local_llm_top_k = 40
                local_llm_repetition_penalty = 1.1
                print(f"Applied balanced preset: temp=0.3, top_p=0.85, top_k=40, rep_penalty=1.1")
            elif preset == "creative":
                local_llm_temperature = 0.5
                local_llm_top_p = 0.9
                local_llm_top_k = 50
                local_llm_repetition_penalty = 1.05
                print(f"Applied creative preset: temp=0.5, top_p=0.9, top_k=50, rep_penalty=1.05")
            else:
                print(f"Error: --preset must be 'conservative', 'balanced', or 'creative'")
                sys.exit(1)
            i += 2
        # Individual local LLM parameters
        elif arg == "--local-temp" and i + 1 < len(sys.argv):
            try:
                local_llm_temperature = float(sys.argv[i + 1])
                if not (0.1 <= local_llm_temperature <= 1.0):
                    print("Error: --local-temp must be between 0.1 and 1.0")
                    sys.exit(1)
            except ValueError:
                print("Error: --local-temp must be a valid float")
                sys.exit(1)
            i += 2
        elif arg == "--local-top-p" and i + 1 < len(sys.argv):
            try:
                local_llm_top_p = float(sys.argv[i + 1])
                if not (0.1 <= local_llm_top_p <= 1.0):
                    print("Error: --local-top-p must be between 0.1 and 1.0")
                    sys.exit(1)
            except ValueError:
                print("Error: --local-top-p must be a valid float")
                sys.exit(1)
            i += 2
        elif arg == "--local-top-k" and i + 1 < len(sys.argv):
            try:
                local_llm_top_k = int(sys.argv[i + 1])
                if not (1 <= local_llm_top_k <= 100):
                    print("Error: --local-top-k must be between 1 and 100")
                    sys.exit(1)
            except ValueError:
                print("Error: --local-top-k must be a valid integer")
                sys.exit(1)
            i += 2
        elif arg == "--local-max-tokens" and i + 1 < len(sys.argv):
            try:
                local_llm_max_tokens = int(sys.argv[i + 1])
                if not (50 <= local_llm_max_tokens <= 2048):
                    print("Error: --local-max-tokens must be between 50 and 2048")
                    sys.exit(1)
            except ValueError:
                print("Error: --local-max-tokens must be a valid integer")
                sys.exit(1)
            i += 2
        elif arg == "--local-rep-penalty" and i + 1 < len(sys.argv):
            try:
                local_llm_repetition_penalty = float(sys.argv[i + 1])
                if not (1.0 <= local_llm_repetition_penalty <= 1.5):
                    print("Error: --local-rep-penalty must be between 1.0 and 1.5")
                    sys.exit(1)
            except ValueError:
                print("Error: --local-rep-penalty must be a valid float")
                sys.exit(1)
            i += 2
        else:
            # Try to parse as start_step if it's a digit
            if arg.isdigit():
                start_step = int(arg)
            else:
                print(f"Warning: Unknown argument '{arg}', ignoring...")
            i += 1
    
    # Validate start step
    if start_step < 1 or start_step > 6:
        print("Error: start_step must be between 1 and 6")
        sys.exit(1)
    
    # Validate OpenAI requirements
    if (llm_type == "openai" or tts_type == "openai") and not openai_api_key:
        print("Error: OpenAI API key required when using OpenAI services")
        print("Provide --openai-api-key YOUR_API_KEY")
        sys.exit(1)
    
    # Validate OpenAI library availability
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        
    if (llm_type == "openai" or tts_type == "openai") and not OPENAI_AVAILABLE:
        print("Error: OpenAI library not installed but required for OpenAI services")
        print("Install with: pip install openai")
        sys.exit(1)
    
    # Validate input file
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found")
        sys.exit(1)
    
    # Check if output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_video))
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist")
        sys.exit(1)
    
    # Display configuration
    print("=" * 70)
    print("UNIFIED VIDEO DUBBING PIPELINE - CONFIGURATION")
    print("=" * 70)
    print(f"📁 Input video: {input_video}")
    print(f"📁 Output video: {output_video}")
    print(f"🚀 Start step: {start_step}")
    print(f"🧠 LLM type: {llm_type}")
    print(f"🎵 TTS type: {tts_type}")
    
    if llm_type == "local":
        print(f"🔧 Local LLM parameters:")
        print(f"   • Temperature: {local_llm_temperature}")
        print(f"   • Top-p: {local_llm_top_p}")
        print(f"   • Top-k: {local_llm_top_k}")
        print(f"   • Max tokens: {local_llm_max_tokens}")
        print(f"   • Repetition penalty: {local_llm_repetition_penalty}")
        print(f"   • Do sample: {local_llm_do_sample}")
    
    if openai_api_key:
        print(f"🔑 OpenAI API key: {'*' * (len(openai_api_key) - 4)}{openai_api_key[-4:]}")
    
    if tts_type == "openai":
        print(f"🎤 OpenAI TTS voice: {openai_tts_voice}")
        print(f"📦 OpenAI TTS model: {openai_tts_model}")
    
    print("=" * 70)
    
    # Pre-validate LLM requirements for the pipeline
    steps_requiring_llm = [2, 3]  # Translation and refinement
    llm_will_be_needed = False
    
    # Check if any LLM-requiring steps will be executed
    for step in range(start_step, 7):
        if step in steps_requiring_llm:
            # Check if this step is already completed
            step_files = {
                2: output_video.replace('.mp4', '_persian.srt'),  # Approximate check
                3: output_video.replace('.mp4', '_refined.srt')   # Approximate check
            }
            # For simplicity, assume LLM will be needed if starting from steps 2-3
            if step <= 3:
                llm_will_be_needed = True
                break
    
    # Early warning about LLM requirements
    if llm_will_be_needed:
        print(f"🔍 LLM will be required for this pipeline (steps {steps_requiring_llm})")
        if llm_type == "local":
            print("📋 Make sure local model is available and HuggingFace token is set if needed")
        elif llm_type == "openai":
            print("📋 OpenAI API will be used for translation and refinement")
    
    # Initialize dubber with all parameters
    try:
        dubber = VideoDubber(
            input_video, 
            output_video, 
            start_step=start_step,
            llm_type=llm_type,
            openai_api_key=openai_api_key,
            tts_type=tts_type,
            openai_tts_voice=openai_tts_voice,
            openai_tts_model=openai_tts_model,
            # Local LLM parameters
            local_llm_temperature=local_llm_temperature,
            local_llm_top_p=local_llm_top_p,
            local_llm_top_k=local_llm_top_k,
            local_llm_max_tokens=local_llm_max_tokens,
            local_llm_repetition_penalty=local_llm_repetition_penalty,
            local_llm_do_sample=local_llm_do_sample
        )
        print("✅ VideoDubber initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize VideoDubber: {e}")
        sys.exit(1)
    
    try:
        print("\n🚀 Starting pipeline...")
        success = dubber.run_full_pipeline()
        
        if success:
            print(f"\n🎉 SUCCESS! Dubbed video created successfully!")
            print(f"📁 Output file: {output_video}")
            print(f"📂 Work directory: {dubber.work_dir}")
            
            # Ask if user wants to clean up temporary files (with timeout)
            print("\n" + "="*50)
            cleanup_approved = dubber.timeout_prompt_with_approval(
                "🧹 Do you want to clean up temporary files to save disk space?", 
                None, 
                timeout_minutes=2
            )
            
            if cleanup_approved:
                keep_audio_approved = dubber.timeout_prompt_with_approval(
                    "🎵 Keep generated audio files? (Recommended: Yes for future reference)", 
                    None, 
                    timeout_minutes=1
                )
                dubber.cleanup_temp_files(keep_audio=keep_audio_approved)
                print("✅ Cleanup completed")
            else:
                print("📂 Temporary files preserved for future use")
            
            print(f"\n🎬 Final output: {output_video}")
            print("🎉 Video dubbing completed successfully!")
            
        else:
            print(f"\n❌ Pipeline failed. Check the error messages above.")
            print(f"📂 Work directory preserved for debugging: {dubber.work_dir}")
            print("\n💡 Troubleshooting tips:")
            print("  • Check if all required dependencies are installed")
            print("  • Verify API keys are correct and have sufficient credits")
            print("  • Check if input file is accessible and not corrupted")
            print("  • Review the error messages above for specific issues")
            print("  • Try running individual steps to isolate the problem")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        print(f"📂 Work directory preserved: {dubber.work_dir}")
        print("💡 You can resume from where you left off using the appropriate step number")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        print(f"📂 Work directory preserved for debugging: {dubber.work_dir}")
        print("\n🐛 Please report this error with the following details:")
        print(f"   • Error: {str(e)}")
        print(f"   • LLM type: {llm_type}")
        print(f"   • TTS type: {tts_type}")
        print(f"   • Start step: {start_step}")
        import traceback
        print(f"   • Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()