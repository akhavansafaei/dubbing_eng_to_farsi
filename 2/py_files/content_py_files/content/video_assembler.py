#!/usr/bin/env python3
"""
Video Assembler with Speed Adjustment
Processes video segments with individual speed adjustments, combines with TTS audio,
and adds background audio to create final video (without subtitle burning).

Usage: python video_assembler.py input_video srt_file csv_file audio_dir background_audio output_video
"""

import os
import sys
import argparse
import subprocess
import tempfile
import csv
import re
import json
from pathlib import Path
from tqdm import tqdm

def parse_srt_timestamp(timestamp):
    """Convert SRT timestamp to seconds"""
    # Format: HH:MM:SS,mmm
    time_pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    match = re.match(time_pattern, timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds

def parse_srt_file(srt_path):
    """Parse SRT file and extract timing information"""
    with open(srt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    blocks = re.split(r'\n\s*\n', content.strip())
    segments = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            sequence = int(lines[0])
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
            if not timestamp_match:
                continue
                
            start_time = timestamp_match.group(1)
            end_time = timestamp_match.group(2)
            text = '\n'.join(lines[2:]).strip()
            
            start_seconds = parse_srt_timestamp(start_time)
            end_seconds = parse_srt_timestamp(end_time)
            
            segments.append({
                'sequence': sequence,
                'start': start_time,
                'end': end_time,
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'duration_seconds': end_seconds - start_seconds,
                'text': text
            })
                
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping malformed SRT block: {e}")
            continue
    
    return segments

def parse_csv_file(csv_path):
    """Parse CSV file with speed adjustment data"""
    adjustments = {}
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            line_sequence = int(row['line_sequence'])
            
            # Parse speed ratios (remove 'x' suffix if present)
            voice_speedup = float(row['voice_speedup_ratio'].replace('x', ''))
            video_speed = float(row['video_speed_ratio'].replace('x', ''))
            
            adjustments[line_sequence] = {
                'original_duration_sec': float(row['original_duration_sec']),
                'audio_duration_sec': float(row['audio_duration_sec']),
                'voice_speedup_ratio': voice_speedup,
                'video_speed_ratio': video_speed,
                'final_segment_duration_sec': float(row['final_segment_duration_sec']),
                'time_shift_for_next_line_sec': float(row['time_shift_for_next_line_sec'])
            }
    
    return adjustments

def find_audio_file(audio_dir, sequence):
    """Find the corresponding audio file for a segment (supports both WAV and MP3)"""
    # List of patterns to try for both WAV and MP3 files
    patterns = [
        f"line_{sequence}.wav",
        f"line_{sequence}.mp3",
        f"line_{sequence:02d}.wav",
        f"line_{sequence:02d}.mp3",
        f"line_{sequence:03d}.wav",
        f"line_{sequence:03d}.mp3",
        f"line_{sequence:04d}.wav",
        f"line_{sequence:04d}.mp3",
        f"seg_{sequence}.wav",
        f"seg_{sequence}.mp3",
        f"audio_{sequence}.wav",
        f"audio_{sequence}.mp3",
        f"{sequence}.wav",
        f"{sequence}.mp3",
        f"{sequence:02d}.wav",
        f"{sequence:02d}.mp3",
        f"{sequence:03d}.wav",
        f"{sequence:03d}.mp3",
        f"{sequence:04d}.wav",
        f"{sequence:04d}.mp3"
    ]
    
    for pattern in patterns:
        file_path = os.path.join(audio_dir, pattern)
        if os.path.exists(file_path):
            return file_path
    
    return None

def seconds_to_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def get_video_duration(video_path):
    """Get video duration in seconds"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    return 0

def create_base_video_without_audio(input_video, output_path):
    """Create video without audio (simplified version)"""
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-an',  # Remove audio
        '-c:v', 'copy',  # Copy video without re-encoding
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error: {result.stderr}")
    
    return output_path

def create_mixed_audio_track(srt_segments, adjustments, audio_dir, background_audio, output_path):
    """Create mixed audio track with TTS voices positioned at correct times and background audio"""
    
    # Get background audio duration
    cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', background_audio]
    result = subprocess.run(cmd, capture_output=True, text=True)
    bg_duration = float(result.stdout.strip()) if result.returncode == 0 else 3600  # Default 1 hour
    
    # Find the total duration needed
    max_end_time = max(seg['end_seconds'] for seg in srt_segments)
    total_duration = max(max_end_time, bg_duration)
    
    # Build filter complex for audio mixing
    filter_complex_parts = []
    input_files = [background_audio]  # Background audio is input 0
    
    # Add TTS audio files as inputs
    tts_inputs = []
    missing_audio_files = []
    
    for segment in srt_segments:
        seq = segment['sequence']
        audio_file = find_audio_file(audio_dir, seq)
        if audio_file:
            tts_inputs.append({
                'file': audio_file,
                'start_time': segment['start_seconds'],
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
    
    # Start with background audio
    filter_complex_parts.append(f"[0:a]volume=-15dB[bg]")
    
    # Process each TTS audio
    audio_mix_inputs = ["[bg]"]
    
    for i, tts_input in enumerate(tts_inputs):
        seq = tts_input['sequence']
        input_idx = tts_input['input_index']
        start_time = tts_input['start_time']
        
        # Apply speed adjustment if needed
        if seq in adjustments:
            speed_ratio = adjustments[seq]['voice_speedup_ratio']
            if abs(speed_ratio - 1.0) > 0.01:
                filter_complex_parts.append(f"[{input_idx}:a]atempo={speed_ratio}[speed{i}]")
                current_audio = f"[speed{i}]"
            else:
                current_audio = f"[{input_idx}:a]"
        else:
            current_audio = f"[{input_idx}:a]"
        
        # Add delay to position audio at correct time
        delay_ms = int(start_time * 1000)
        filter_complex_parts.append(f"{current_audio}adelay={delay_ms}|{delay_ms}[tts{i}]")
        audio_mix_inputs.append(f"[tts{i}]")
    
    # Mix all audio inputs
    if len(audio_mix_inputs) > 1:
        mix_inputs = "".join(audio_mix_inputs)
        filter_complex_parts.append(f"{mix_inputs}amix=inputs={len(audio_mix_inputs)}:duration=first:dropout_transition=0[final_audio]")
        output_map = "[final_audio]"
    else:
        output_map = "[bg]"
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-y']
    
    # Add all input files
    for input_file in input_files:
        cmd.extend(['-i', input_file])
    
    # Add filter complex
    filter_complex = ";".join(filter_complex_parts)
    cmd.extend(['-filter_complex', filter_complex])
    
    # Map output and set duration
    cmd.extend([
        '-map', output_map,
        '-t', str(total_duration),
        '-c:a', 'pcm_s16le',
        output_path
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error creating mixed audio: {result.stderr}")
    
    return output_path

def combine_video_and_audio(video_path, audio_path, output_path):
    """Combine video with mixed audio track (no subtitle burning)"""
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,  # Video input
        '-i', audio_path,  # Audio input
        '-map', '0:v',     # Map video from first input
        '-map', '1:a',     # Map audio from second input
        '-c:v', 'copy',    # Copy video without re-encoding
        '-c:a', 'aac',     # Encode audio as AAC
        '-b:a', '128k',    # Audio bitrate
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error combining video and audio: {result.stderr}")
    
    return output_path

def main():
    if len(sys.argv) != 7:
        print("Usage: python video_assembler.py input_video srt_file csv_file audio_dir background_audio output_video")
        sys.exit(1)
    
    input_video = sys.argv[1]
    srt_file = sys.argv[2]
    csv_file = sys.argv[3]
    audio_dir = sys.argv[4]
    background_audio = sys.argv[5]
    output_video = sys.argv[6]
    
    # Validate input files
    for file_path, name in [(input_video, "Video"), (srt_file, "SRT"), 
                           (csv_file, "CSV"), (background_audio, "Background audio")]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file not found: {file_path}")
            sys.exit(1)
    
    if not os.path.isdir(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    print("Video Assembler with Speed Adjustment (MP3/WAV Support)")
    print("=" * 55)
    print(f"Input video: {input_video}")
    print(f"SRT file: {srt_file}")
    print(f"CSV file: {csv_file}")
    print(f"Audio directory: {audio_dir}")
    print(f"Background audio: {background_audio}")
    print(f"Output video: {output_video}")
    print()
    
    # Check what audio files are available in the directory
    audio_files = []
    for ext in ['*.wav', '*.mp3']:
        audio_files.extend(Path(audio_dir).glob(ext))
    
    print(f"Found {len(audio_files)} audio files in directory:")
    for audio_file in sorted(audio_files)[:10]:  # Show first 10 files
        print(f"  - {audio_file.name}")
    if len(audio_files) > 10:
        print(f"  ... and {len(audio_files) - 10} more files")
    print()
    
    # Parse input files
    print("Parsing SRT file...")
    srt_segments = parse_srt_file(srt_file)
    
    print("Parsing CSV file...")
    adjustments = parse_csv_file(csv_file)
    
    print(f"Found {len(srt_segments)} SRT segments and {len(adjustments)} CSV adjustments")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating base video (removing original audio)...")
        
        # Step 1: Create video without audio
        base_video = os.path.join(temp_dir, "base_video.mp4")
        create_base_video_without_audio(input_video, base_video)
        
        print("Creating mixed audio track...")
        
        # Step 2: Create mixed audio track with TTS positioned at correct times
        mixed_audio = os.path.join(temp_dir, "mixed_audio.wav")
        create_mixed_audio_track(srt_segments, adjustments, audio_dir, background_audio, mixed_audio)
        
        print("Combining video with mixed audio...")
        
        # Step 3: Combine video with mixed audio (no subtitle burning)
        combine_video_and_audio(base_video, mixed_audio, output_video)
    
    print(f"\nSuccess! Final video created: {output_video}")
    
    # Create processing report
    report_path = output_video.replace('.mp4', '_processing_report.json')
    report_data = {
        'input_video': input_video,
        'srt_file': srt_file,
        'csv_file': csv_file,
        'audio_dir': audio_dir,
        'background_audio': background_audio,
        'output_video': output_video,
        'segments_processed': len(srt_segments),
        'adjustments_applied': len(adjustments),
        'processing_method': 'positioned_audio_overlay_no_subtitles',
        'supported_audio_formats': ['wav', 'mp3'],
        'audio_files_found': len(audio_files)
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing report saved to: {report_path}")

if __name__ == "__main__":
    main()