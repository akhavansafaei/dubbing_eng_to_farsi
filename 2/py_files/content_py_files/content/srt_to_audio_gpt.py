#!/usr/bin/env python3
"""
SRT to Audio Converter with OpenAI TTS
Generates a separate audio file for each line in an SRT file using OpenAI's text-to-speech API.
Usage: python srt_to_audio.py input.srt --api-key YOUR_API_KEY
"""

import os
import sys
import argparse
import re
import time
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them using: pip install openai")
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

def generate_tts_instruction_from_srt(srt_path, api_key, model="gpt-4o"):
    """Generate a global TTS instruction for the SRT content using OpenAI LLM."""
    API_KEY= 
    client = OpenAI(api_key=API_KEY)
    
    # Extract all text from SRT segments
    segments = parse_srt_file(srt_path)
    if not segments:
        return None
    
    # Combine all subtitle text
    full_text = ' '.join([segment['text'] for segment in segments])
    
    system_prompt = (
        "You are a voice director. Based on the full subtitle content, write a single global instruction "
        "for how it should be spoken using a TTS model. Focus on voice qualities such as tone, pace, "
        "clarity, and emotion. Format the output as a brief instruction paragraph suitable for a TTS API."
    )
    
    user_prompt = f"Here is the subtitle content:\n\n{full_text}\n\nNow write the instruction."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Failed to generate TTS instruction: {e}")
        return None

def text_to_speech_openai(text, client, output_filename, voice="alloy", model="tts-1", instructions=None):
    """
    Generates audio from a given text using OpenAI's TTS API.
    """
    try:
        # Prepare the request parameters
        params = {
            "model": model,
            "voice": voice,
            "input": text
        }
        
        # Add instructions if available (note: instructions parameter may not be available in all TTS models)
        if instructions and model == "tts-1-hd":
            params["instructions"] = instructions
        
        # Generate audio from text
        with client.audio.speech.with_streaming_response.create(**params) as response:
            response.stream_to_file(output_filename)
        
        return True
    except Exception as e:
        print(f"An error occurred during TTS generation for '{text[:50]}...': {e}")
        return False

def generate_audio_from_srt(srt_path, api_key, output_dir, voice="alloy", model="tts-1", generate_instruction=False):
    """
    Main function to process an SRT file and generate audio for each line using OpenAI TTS.
    """
    # 1. Initialize OpenAI client
    print("Initializing OpenAI TTS client...")
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)
    
    print("OpenAI TTS client initialized successfully.")

    # 2. Generate TTS instruction if requested
    tts_instruction = None
    if generate_instruction:
        print("Generating TTS instruction from SRT content...")
        tts_instruction = generate_tts_instruction_from_srt(srt_path, api_key)
        if tts_instruction:
            print(f"Generated TTS Instruction: {tts_instruction}")
        else:
            print("Failed to generate TTS instruction, proceeding without it.")

    # 3. Parse the SRT file
    segments = parse_srt_file(srt_path)
    if segments is None:
        sys.exit(1)
    print(f"Found {len(segments)} subtitle segments to process.")

    # 4. Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Audio files will be saved in: {output_dir}")

    # 5. Process each segment and generate audio
    successful_generations = 0
    for segment in tqdm(segments, desc="Generating Audio"):
        line_text = segment['text']
        sequence_num = segment['sequence']
        output_filename = os.path.join(output_dir, f"line_{sequence_num}.mp3")
        
        # Skip TTS if the file already exists
        if os.path.exists(output_filename):
            successful_generations += 1
            continue
        
        # Skip empty or very short texts
        if len(line_text.strip()) < 3:
            print(f"Skipping very short text: '{line_text}'")
            continue
            
        success = text_to_speech_openai(
            line_text, 
            client, 
            output_filename, 
            voice=voice, 
            model=model, 
            instructions=tts_instruction
        )
        
        if success:
            successful_generations += 1
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

    print(f"\nAudio generation process completed!")
    print(f"Successfully generated {successful_generations} out of {len(segments)} audio files.")

def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(
        description='Generate an MP3 audio file for each line of an SRT file using OpenAI TTS.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_srt', help='Path to the input SRT file.')
    parser.add_argument(
        '--api-key',
        required=True,
        help='OpenAI API key for TTS service.'
    )
    parser.add_argument(
        '--voice',
        default="ash",
        choices=["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"],
        help='Voice to use for TTS generation.\n(default: %(default)s)'
    )
    parser.add_argument(
        '--model',
        default="gpt-4o-mini-tts",
        choices=["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
        help='TTS model to use. gpt-4o-mini-tts is the latest mini-voice model; tts-1-hd has higher quality but is slower.\n(default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir',
        help='Path to the output directory for saving MP3 files.\n(default: [input_srt_name]_audio)'
    )
    parser.add_argument(
        '--generate-instruction',
        action='store_true',
        help='Generate a global TTS instruction based on the SRT content using GPT-4.'
    )
    
    args = parser.parse_args()

    # Determine the output directory
    output_dir = args.output_dir
    if not output_dir:
        base_name = os.path.splitext(os.path.basename(args.input_srt))[0]
        output_dir = f"{base_name}_audio"

    generate_audio_from_srt(
        args.input_srt, 
        args.api_key, 
        output_dir, 
        voice=args.voice, 
        model=args.model,
        generate_instruction=args.generate_instruction
    )

if __name__ == "__main__":
    main()