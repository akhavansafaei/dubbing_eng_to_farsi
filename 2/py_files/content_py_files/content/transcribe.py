#!/usr/bin/env python3
"""
Video Transcription Script using OpenAI Whisper
Supports multiple output formats: SRT, TXT, VTT, JSON, TSV, and more

Performance Guide:
Model      | Size  | Speed    | Accuracy | RAM Usage | Recommended Use
-----------|-------|----------|----------|-----------|----------------
tiny       | 39MB  | Fastest  | Good     | ~1GB      | Quick drafts, real-time
tiny.en    | 39MB  | Fastest  | Good     | ~1GB      | English only, fastest
base       | 74MB  | Fast     | Better   | ~1GB      | Balanced choice
base.en    | 74MB  | Fast     | Better   | ~1GB      | English only, balanced
small      | 244MB | Medium   | Good     | ~2GB      | Good quality/speed ratio
small.en   | 244MB | Medium   | Good     | ~2GB      | English only, good ratio
medium     | 769MB | Slow     | Better   | ~5GB      | High accuracy needs
medium.en  | 769MB | Slow     | Better   | ~5GB      | English only, high accuracy
large-v1   | 1.5GB | Slowest  | Best     | ~10GB     | Maximum accuracy
large-v2   | 1.5GB | Slowest  | Best     | ~10GB     | Maximum accuracy
large-v3   | 1.5GB | Slowest  | Best     | ~10GB     | Latest, maximum accuracy

Speed Tips:
- Use --fast flag for quickest results
- For English content, use .en models (2x faster)
- Use smaller models for drafts, large models for final transcripts
- Consider faster-whisper for better performance
"""

import argparse
import os
import sys
import time
from pathlib import Path
import json

# Attempt to import whisper, fallback to faster_whisper
try:
    import whisper
    USE_FASTER_WHISPER = False
except ImportError:
    try:
        from faster_whisper import WhisperModel
        USE_FASTER_WHISPER = True
        print("Using faster-whisper implementation")
    except ImportError:
        print("Error: Please install either 'openai-whisper' or 'faster-whisper' to proceed.")
        print("Install command: pip install openai-whisper")
        print("Or for better performance: pip install faster-whisper")
        sys.exit(1)

from typing import Optional, List, Dict, Any

class VideoTranscriber:
    def __init__(self):
        self.supported_models = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"
        ]
        self.supported_formats = ["srt", "txt", "vtt", "json", "tsv", "csv"]
        self.use_faster_whisper = USE_FASTER_WHISPER

    def load_model(self, model_name: str, device: Optional[str] = None):
        """Load the Whisper model."""
        print(f"Loading Whisper model: {model_name}...")
        try:
            if self.use_faster_whisper:
                # faster-whisper implementation
                device_type = "cuda" if device != "cpu" else "cpu"
                model = WhisperModel(model_name, device=device_type)
                print(f"Faster-Whisper model loaded successfully on device: {device_type}")
            else:
                # openai-whisper implementation
                model = whisper.load_model(model_name, device=device)
                print(f"OpenAI-Whisper model loaded successfully on device: {model.device}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def transcribe_video(self, model, video_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe the video file."""
        print(f"Transcribing: {video_path}")
        start_time = time.time()

        try:
            if self.use_faster_whisper:
                # faster-whisper implementation returns a generator
                segments, info = model.transcribe(video_path, **kwargs)
                result = {
                    'text': '',
                    'segments': [],
                    'language': info.language,
                    'language_probability': info.language_probability
                }
                
                # Segments need to be converted to a list of dicts
                segment_list = []
                for segment in segments:
                    segment_dict = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    }
                    segment_list.append(segment_dict)
                    result['text'] += segment.text
                result['segments'] = segment_list
            else:
                # openai-whisper implementation
                result = model.transcribe(video_path, **kwargs)

            end_time = time.time()
            print(f"Transcription completed in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)

    def format_time_srt(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def format_time_vtt(self, seconds: float) -> str:
        """Format time for VTT format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

    def save_srt(self, result: Dict[str, Any], output_path: str):
        """Save transcription in SRT format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start_time = self.format_time_srt(segment['start'])
                end_time = self.format_time_srt(segment['end'])
                text = segment['text'].strip()
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

    def save_vtt(self, result: Dict[str, Any], output_path: str):
        """Save transcription in VTT format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for segment in result['segments']:
                start_time = self.format_time_vtt(segment['start'])
                end_time = self.format_time_vtt(segment['end'])
                text = segment['text'].strip()
                f.write(f"{start_time} --> {end_time}\n{text}\n\n")

    def save_txt(self, result: Dict[str, Any], output_path: str):
        """Save transcription in plain text format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'].strip())

    def save_json(self, result: Dict[str, Any], output_path: str):
        """Save transcription in JSON format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def save_tsv(self, result: Dict[str, Any], output_path: str):
        """Save transcription in TSV format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("start\tend\ttext\n")
            for segment in result['segments']:
                start = f"{segment['start']:.2f}"
                end = f"{segment['end']:.2f}"
                text = segment['text'].strip().replace('\t', ' ')
                f.write(f"{start}\t{end}\t{text}\n")

    def save_csv(self, result: Dict[str, Any], output_path: str):
        """Save transcription in CSV format"""
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end', 'text'])
            for segment in result['segments']:
                writer.writerow([
                    f"{segment['start']:.2f}",
                    f"{segment['end']:.2f}",
                    segment['text'].strip()
                ])

    def save_transcription(self, result: Dict[str, Any], output_dir: str, base_name: str, formats: List[str]):
        """Save transcription in specified formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
            print(f"Saving {fmt.upper()} file: {output_path}")

            save_func = getattr(self, f"save_{fmt}", None)
            if save_func:
                save_func(result, output_path)
            else:
                print(f"Warning: Unknown format '{fmt}' requested. Skipping.")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py video.mp4
  python transcribe.py audio.mp3 --model large-v3 --language en
  python transcribe.py "My File.m4a" --output-dir ./transcripts --formats srt txt json
  python transcribe.py video.mp4 --task translate --temperature 0.2
        """
    )
    
    # Required arguments
    parser.add_argument("video_path", help="Path to the video or audio file to transcribe")
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"],
        help="Whisper model to use. Performance: tiny(fastest) < base < small < medium < large-v3(slowest) (default: base)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        default="./transcripts",
        help="Output directory for transcription files (default: ./transcripts)"
    )
    
    parser.add_argument(
        "--output-name", "-n",
        help="Base name for output files (default: video filename without extension)"
    )
    
    parser.add_argument(
        "--formats", "-f",
        nargs="+",
        default=["srt", "txt"],
        choices=["srt", "txt", "vtt", "json", "tsv", "csv"],
        help="Output formats (default: srt txt)"
    )
    
    # Whisper transcription parameters
    parser.add_argument(
        "--language", "-l",
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified"
    )
    
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform (default: transcribe)"
    )
    
    # FIX: Added the missing --verbose argument
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from Whisper (default: False). Not used with faster-whisper."
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)"
    )
    
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidates when temperature > 0 (default: 5)"
    )
    
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search (default: 5)"
    )
    
    parser.add_argument(
        "--patience",
        type=float,
        default=None,
        help="Patience for beam search"
    )
    
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Length penalty for beam search"
    )
    
    parser.add_argument(
        "--suppress-tokens",
        default="-1",
        help="Comma-separated list of token IDs to suppress (default: -1)"
    )
    
    parser.add_argument(
        "--initial-prompt",
        help="Initial prompt to condition the model"
    )
    
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        default=True,
        help="Condition on previous text (default: True)"
    )
    
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
        help="Don't condition on previous text"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision (default: True)"
    )
    
    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Don't use FP16 precision"
    )
    
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Compression ratio threshold (default: 2.4)"
    )
    
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Log probability threshold (default: -1.0)"
    )
    
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="No speech threshold (default: 0.6)"
    )
    
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Extract word-level timestamps"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video_path):
        print(f"Error: Input file '{args.video_path}' not found")
        sys.exit(1)
    
    # Determine output name
    base_name = args.output_name or Path(args.video_path).stem
    
    # Initialize transcriber
    transcriber = VideoTranscriber()
    
    # Set device
    device = None if args.device == "auto" else args.device
    
    # Load model
    model = transcriber.load_model(args.model, device=device)
    
    # Prepare transcription parameters, filtering for None values
    transcribe_params = {
        "task": args.task,
        "temperature": args.temperature,
        "best_of": args.best_of,
        "beam_size": args.beam_size,
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "initial_prompt": args.initial_prompt,
        "condition_on_previous_text": args.condition_on_previous_text,
        "fp16": args.fp16,
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "logprob_threshold": args.logprob_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "word_timestamps": args.word_timestamps,
        "language": args.language,
        "suppress_tokens": args.suppress_tokens,
        "verbose": args.verbose if not transcriber.use_faster_whisper else None,
    }

    # Clean out None parameters so defaults are used
    final_params = {k: v for k, v in transcribe_params.items() if v is not None}
    
    # Perform transcription
    result = transcriber.transcribe_video(model, args.video_path, **final_params)
    
    # Save transcription
    transcriber.save_transcription(result, args.output_dir, base_name, args.formats)
    
    print(f"\nTranscription complete!")
    print(f"Output files saved in: {os.path.abspath(args.output_dir)}")
    print(f"Formats: {', '.join(args.formats)}")
    
    # Print summary
    if 'segments' in result and result['segments']:
        total_duration = result['segments'][-1]['end']
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Number of segments: {len(result['segments'])}")
        print(f"Detected language: '{result.get('language')}'")

if __name__ == "__main__":
    main()
