#!/usr/bin/env python3
"""
SRT to Audio Converter
Generates a separate audio file for each line in an SRT file using a text-to-speech model.
Usage: python srt_to_audio.py input.srt
"""

import os
import sys
import argparse
import re
import time
from tqdm import tqdm

try:
    import sherpa_onnx
    import soundfile as sf
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them using: pip install sherpa-onnx soundfile")
    sys.exit(1)

def parse_srt_file(file_path):
    """Parses an SRT file into a list of subtitle segments."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: SRT file not found at '{file_path}'")
        return None

    blocks = re.split(r'\n\s*\n', content.strip())
    segments = []
    for i, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) >= 2 and '-->' in lines[1]:
            try:
                # Use line number `i + 1` as sequence if the SRT sequence is missing or invalid
                sequence = lines[0] if lines[0].isdigit() else str(i + 1)
                text = ' '.join(lines[2:]).strip()
                if text:
                    segments.append({'sequence': sequence, 'text': text})
            except (ValueError, IndexError):
                print(f"Skipping malformed SRT block:\n{block}")
    return segments

def text_to_speech_persian(text, tts_engine, output_filename):
    """
    Generates audio from a given text using the pre-configured TTS engine.
    """
    try:
        # Generate audio from text
        audio = tts_engine.generate(text, sid=0, speed=1.0)

        if len(audio.samples) == 0:
            print(f"Warning: TTS failed for text: '{text}'. No audio generated.")
            return False

        # Save the generated audio
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

def generate_audio_from_srt(srt_path, model_dir, output_dir):
    """
    Main function to process an SRT file and generate audio for each line.
    """
    # 1. Check if the model directory exists
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found at '{model_dir}'.")
        print("Please ensure you have cloned the model from Hugging Face and provided the correct path.")
        sys.exit(1)

    # 2. Configure and initialize the TTS model once
    print("Initializing Text-to-Speech engine...")
    model_path = f"{model_dir}/fa_IR-amir-medium.onnx"
    tokens_path = f"{model_dir}/tokens.txt"
    espeak_data_path = f"{model_dir}/espeak-ng-data"

    if not all(os.path.exists(p) for p in [model_path, tokens_path, espeak_data_path]):
        print(f"Error: One or more model files are missing from '{model_dir}'.")
        sys.exit(1)

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
        tts_engine = sherpa_onnx.OfflineTts(tts_config)
    except Exception as e:
        print(f"Failed to initialize TTS engine: {e}")
        sys.exit(1)
    
    print("TTS engine initialized successfully.")

    # 3. Parse the SRT file
    segments = parse_srt_file(srt_path)
    if segments is None:
        sys.exit(1)
    print(f"Found {len(segments)} subtitle segments to process.")

    # 4. Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Audio files will be saved in: {output_dir}")

    # 5. Process each segment and generate audio
    for segment in tqdm(segments, desc="Generating Audio"):
        line_text = segment['text']
        sequence_num = segment['sequence']
        output_filename = os.path.join(output_dir, f"line_{sequence_num}.wav")
        
        # Skip TTS if the file already exists
        if os.path.exists(output_filename):
            continue
            
        text_to_speech_persian(line_text, tts_engine, output_filename)
        # Add a small delay to avoid overwhelming the system
        time.sleep(0.1)

    print("\nAudio generation process completed!")

def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(
        description='Generate a WAV audio file for each line of an SRT file.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_srt', help='Path to the input SRT file.')
    parser.add_argument(
        '--model-dir',
        default="./vits-piper-fa_IR-amir-medium",
        help='Path to the directory containing the TTS model files.\n(default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir',
        help='Path to the output directory for saving WAV files.\n(default: [input_srt_name]_audio)'
    )
    
    args = parser.parse_args()

    # Determine the output directory
    output_dir = args.output_dir
    if not output_dir:
        base_name = os.path.splitext(os.path.basename(args.input_srt))[0]
        output_dir = f"{base_name}_audio"

    generate_audio_from_srt(args.input_srt, args.model_dir, output_dir)

if __name__ == "__main__":
    main()
