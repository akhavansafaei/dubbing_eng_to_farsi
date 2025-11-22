#!/usr/bin/env python3
"""
SRT Timing Adjuster based on Audio Duration

This script adjusts the timing of an SRT file to match the duration of
corresponding audio files, applying voice speed-up and video slow-down
as needed, and shifting subsequent subtitles if necessary.

Usage:
    python srt_adjuster.py input.srt /path/to/audio_folder
"""

import os
import sys
import argparse
import re
import csv
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    print("Error: The 'soundfile' library is required. Please install it.")
    print("Install command: pip install soundfile")
    sys.exit(1)

def time_str_to_seconds(time_str):
    """Converts SRT time format (HH:MM:SS,ms) to seconds."""
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def seconds_to_time_str(seconds):
    """Converts seconds to SRT time format (HH:MM:SS,ms)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def find_audio_file(audio_folder, sequence_num):
    """Finds the audio file for a given sequence number, checking both WAV and MP3 extensions."""
    base_name = f"line_{sequence_num}"
    
    # Check for WAV file first
    wav_path = os.path.join(audio_folder, f"{base_name}.wav")
    if os.path.exists(wav_path):
        return wav_path
    
    # Check for MP3 file
    mp3_path = os.path.join(audio_folder, f"{base_name}.mp3")
    if os.path.exists(mp3_path):
        return mp3_path
    
    return None

def get_audio_duration(file_path):
    """Returns the duration of an audio file (WAV or MP3) in seconds."""
    if not file_path or not os.path.exists(file_path):
        return 0.0
    try:
        with sf.SoundFile(file_path) as f:
            return len(f) / f.samplerate
    except Exception as e:
        print(f"Warning: Could not read duration from '{file_path}'. Error: {e}")
        return 0.0

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
                sequence = lines[0] if lines[0].isdigit() else str(i + 1)
                start, end = lines[1].split(' --> ')
                text = ' '.join(lines[2:]).strip()
                if text:
                    segments.append({
                        'sequence': sequence,
                        'start': start.strip(),
                        'end': end.strip(),
                        'text': text
                    })
            except (ValueError, IndexError):
                print(f"Skipping malformed SRT block:\n{block}")
    return segments

def write_srt_file(segments, output_path):
    """Writes segments to an SRT file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(f"{segment['sequence']}\n")
            file.write(f"{segment['start']} --> {segment['end']}\n")
            file.write(f"{segment['text']}\n\n")

def write_report_file(report_data, output_path):
    """Writes the adjustment report to a CSV file."""
    if not report_data:
        return
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=report_data[0].keys())
        writer.writeheader()
        writer.writerows(report_data)

def adjust_srt_timing(srt_path, audio_folder, output_path, report_path, max_voice_speed, min_video_speed):
    """
    Main logic to adjust SRT timings based on audio durations.
    """
    segments = parse_srt_file(srt_path)
    if not segments:
        sys.exit(1)

    print(f"Processing {len(segments)} subtitle segments...")

    new_segments = []
    adjustment_report = []
    cumulative_offset = 0.0

    for segment in tqdm(segments, desc="Adjusting Timings"):
        sequence_num = segment['sequence']
        audio_file_path = find_audio_file(audio_folder, sequence_num)

        if not audio_file_path:
            print(f"Warning: Skipping line {sequence_num} - no audio file found (checked for line_{sequence_num}.wav and line_{sequence_num}.mp3).")
            continue

        actual_audio_duration = get_audio_duration(audio_file_path)
        if actual_audio_duration == 0.0:
            print(f"Warning: Skipping line {sequence_num} due to invalid audio file at '{audio_file_path}'.")
            continue
            
        # Apply cumulative offset from previous lines' adjustments
        start_sec = time_str_to_seconds(segment['start']) + cumulative_offset
        end_sec = time_str_to_seconds(segment['end']) + cumulative_offset
        predicted_duration = end_sec - start_sec

        voice_speedup_ratio = 1.0
        video_speed_ratio = 1.0  # 1.0 means normal speed, < 1.0 is slower

        if actual_audio_duration <= predicted_duration:
            # Audio fits perfectly, no adjustments needed
            final_duration = predicted_duration
        else:
            # --- Adjustment Step 1: Speed up voice ---
            voice_speedup_ratio = min(actual_audio_duration / predicted_duration, max_voice_speed)
            duration_after_voice_speedup = actual_audio_duration / voice_speedup_ratio
            
            if duration_after_voice_speedup <= predicted_duration:
                # Fits after speeding up voice, no more changes needed
                final_duration = predicted_duration
            else:
                # --- Adjustment Step 2: Slow down video ---
                # Max duration achievable by slowing down the video
                max_possible_duration = predicted_duration / min_video_speed

                if duration_after_voice_speedup <= max_possible_duration:
                    # Fits by slowing down video
                    video_speed_ratio = predicted_duration / duration_after_voice_speedup
                    final_duration = duration_after_voice_speedup
                else:
                    # --- Adjustment Step 3: Use max adjustments and shift timeline ---
                    # It still doesn't fit. Apply max slowdown and let the segment run long.
                    video_speed_ratio = min_video_speed
                    final_duration = max_possible_duration

        # Calculate the final end time for this segment
        final_end_sec = start_sec + final_duration
        
        # This is the extra time this segment took compared to its original slot.
        # This will be added to the cumulative offset for all subsequent segments.
        line_offset = final_duration - predicted_duration
        cumulative_offset += line_offset

        # Append data for new SRT file
        new_segments.append({
            'sequence': segment['sequence'],
            'start': seconds_to_time_str(start_sec),
            'end': seconds_to_time_str(final_end_sec),
            'text': segment['text']
        })

        # Get the file extension for the report
        file_extension = os.path.splitext(audio_file_path)[1].upper()

        # Append data for the report
        adjustment_report.append({
            "line_sequence": sequence_num,
            "audio_file_type": file_extension,
            "original_duration_sec": f"{predicted_duration:.3f}",
            "audio_duration_sec": f"{actual_audio_duration:.3f}",
            "voice_speedup_ratio": f"{voice_speedup_ratio:.2f}x",
            "video_speed_ratio": f"{video_speed_ratio:.2f}x",
            "final_segment_duration_sec": f"{final_duration:.3f}",
            "time_shift_for_next_line_sec": f"{line_offset:.3f}",
        })
        
    # Write the output files
    write_srt_file(new_segments, output_path)
    write_report_file(adjustment_report, report_path)
    
    print("\nSRT adjustment process completed!")
    print(f"Adjusted SRT file saved to: {output_path}")
    print(f"Adjustment report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Adjust SRT file timings based on audio durations.")
    parser.add_argument('input_srt', help="Path to the input SRT file.")
    parser.add_argument('audio_folder', help="Path to the folder containing the corresponding WAV or MP3 files.")
    parser.add_argument(
        '--max-voice-speed', type=float, default=1.25,
        help="Maximum speed-up ratio for the voice (e.g., 1.25 for 125%% speed). Default: 1.25."
    )
    parser.add_argument(
        '--min-video-speed', type=float, default=0.90,
        help="Minimum speed for the video (e.g., 0.90 for 90%% speed, which is a 10%% slowdown). Default: 0.90."
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input_srt):
        print(f"Error: Input SRT file not found: '{args.input_srt}'")
        sys.exit(1)
    if not os.path.isdir(args.audio_folder):
        print(f"Error: Audio folder not found: '{args.audio_folder}'")
        sys.exit(1)

    # Define output paths
    base_name = os.path.splitext(os.path.basename(args.input_srt))[0]
    output_srt_path = f"{base_name}_adjusted.srt"
    report_csv_path = f"{base_name}_adjustment_report.csv"

    adjust_srt_timing(
        args.input_srt,
        args.audio_folder,
        output_srt_path,
        report_csv_path,
        args.max_voice_speed,
        args.min_video_speed
    )

if __name__ == "__main__":
    main()