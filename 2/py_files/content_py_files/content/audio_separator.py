#!/usr/bin/env python3
"""
Audio Separation Script using Demucs
Separates vocals and background music from audio files (.mp3, .wav)
Usage: python audio_separator.py input_audio.mp3
"""

import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# Set environment variables to suppress ALSA errors
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip install -q tqdm")
    from tqdm import tqdm

try:
    import demucs.separate
    from demucs.pretrained import get_model
    from demucs.audio import save_audio
    from demucs.apply import apply_model
    import torchaudio
except ImportError:
    print("Installing required packages...")
    os.system("pip install -q demucs torchaudio")
    import demucs.separate
    from demucs.pretrained import get_model
    from demucs.audio import save_audio
    from demucs.apply import apply_model
    import torchaudio

from torch.cuda import is_available as is_cuda_available
from typing import Optional
import gc
import argparse
from pathlib import Path

def downsample_audio(audio, orig_sr, target_sr=16000):
    """Downsamples audio to a target sample rate and converts to mono."""
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio = resampler(audio)
    if audio.shape[0] > 1:  # Stereo to mono
        audio = torch.mean(audio, dim=0, keepdim=True)
    return audio

def load_audio_file(file_path: str, model_channels: int, model_samplerate: int):
    """Load an audio file and prepare it for Demucs processing."""
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.float()

        # Handle mono to stereo conversion if the model requires it
        if waveform.shape[0] == 1 and model_channels == 2:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > model_channels:
            waveform = waveform[:model_channels]

        # Resample if the sample rate doesn't match the model's
        if sample_rate != model_samplerate:
            resampler = torchaudio.transforms.Resample(sample_rate, model_samplerate)
            waveform = resampler(waveform)

        # Add a batch dimension for the model
        waveform = waveform.unsqueeze(0)
        return waveform, model_samplerate
    except Exception as e:
        raise Exception(f"Error loading audio file: {str(e)}")

def get_device():
    """Get the best available device (CUDA, MPS, or CPU) for processing."""
    if is_cuda_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def separate_audio(input_audio: str, output_dir: str = None):
    """Main function to separate vocals and background music from an audio file."""
    input_path = Path(input_audio)
    
    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file '{input_audio}' not found!")
        sys.exit(1)
        
    if input_path.suffix.lower() not in ['.mp3', '.wav']:
        print(f"Error: Unsupported file type. Please provide a .mp3 or .wav file.")
        sys.exit(1)

    # Setup output directory
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_separated"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Define output file paths
    vocal_audio_file = output_dir / f"{input_path.stem}_vocals.wav"
    background_audio_file = output_dir / f"{input_path.stem}_background.wav"

    # Check if separation has already been done
    if vocal_audio_file.exists() and background_audio_file.exists():
        print(f"'{vocal_audio_file.name}' and '{background_audio_file.name}' already exist. Skipping separation.")
        print(f"Output files are in: {output_dir}")
        return

    # Load Demucs model
    print("Loading Demucs model (htdemucs)...")
    try:
        model = get_model('htdemucs')
        model.eval()
        device = get_device()
        model.to(device)
        print(f"Model loaded successfully on '{device}'")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Load and process the input audio file
    print(f"Loading audio file: {input_path.name}...")
    try:
        wav, sr = load_audio_file(str(input_path), model.audio_channels, model.samplerate)
        print(f"Audio loaded. Shape: {wav.shape}, Sample Rate: {sr}")
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        sys.exit(1)

    # Separate audio sources
    print("Separating audio sources...")
    try:
        wav = wav.to(device)
        with torch.no_grad():
            # Use tqdm for a progress bar
            sources = apply_model(model, wav, device=device, shifts=1, split=True, overlap=0.25, progress=True)
        
        # Process the separated sources
        sources = sources.cpu().squeeze(0)
        source_names = ['drums', 'bass', 'other', 'vocals']
        outputs = {name: sources[i] for i, name in enumerate(source_names) if i < sources.shape[0]}
        print("Audio separation completed.")
    except Exception as e:
        print(f"Error during separation: {str(e)}")
        sys.exit(1)

    # Save the separated tracks
    print("Saving separated tracks...")
    
    # Save vocals track
    try:
        vocals = outputs['vocals']
        if vocals.dim() == 1:
            vocals = vocals.unsqueeze(0)
        vocals = downsample_audio(vocals, model.samplerate)
        torchaudio.save(str(vocal_audio_file), vocals, 16000)
        print(f"-> Vocals saved to: {vocal_audio_file.name}")
    except Exception as e:
        print(f"Error saving vocals: {str(e)}")

    # Save background music (combine all non-vocal tracks)
    try:
        background_sources = [audio for source, audio in outputs.items() if source != 'vocals']
        if background_sources:
            background = torch.stack(background_sources).sum(dim=0)
            if background.dim() == 1:
                background = background.unsqueeze(0)
            background = downsample_audio(background, model.samplerate)
            torchaudio.save(str(background_audio_file), background, 16000)
            print(f"-> Background music saved to: {background_audio_file.name}")
        else:
            print("Warning: No background sources found to combine.")
    except Exception as e:
        print(f"Error saving background music: {str(e)}")

    # Clean up memory
    print("Cleaning up memory...")
    del outputs, sources, wav, model
    if 'background' in locals():
        del background
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print("\n✨ Separation process finished successfully! ✨")
    print(f"📁 Output files are located in: {output_dir}")
    print("\nGenerated files:")
    print(f"  🎤 Vocals: {vocal_audio_file.name}")
    print(f"  🎹 Background: {background_audio_file.name}")

def main():
    """Defines the command-line interface for the script."""
    parser = argparse.ArgumentParser(
        description="Separate vocals and background music from an audio file (.mp3, .wav) using Demucs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_separator.py my_song.mp3
  python audio_separator.py ./path/to/my_audio.wav
  python audio_separator.py my_song.mp3 --output ./separated_tracks/
        """
    )
    
    parser.add_argument(
        "input_audio", 
        help="Path to the input audio file (.mp3 or .wav)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: creates a new folder next to the input file)",
        default=None
    )
    
    args = parser.parse_args()
    
    separate_audio(args.input_audio, args.output)

if __name__ == "__main__":
    main()
