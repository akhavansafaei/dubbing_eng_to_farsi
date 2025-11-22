#!/usr/bin/env python3
"""
Video to Audio Extractor
Extracts audio from video files with comprehensive options and error handling.
"""

import os
import sys
import argparse
from pathlib import Path
from moviepy.editor import VideoFileClip
import logging


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def extract_audio(input_file, output_file=None, **kwargs):
    """
    Extract audio from video file.
    
    Args:
        input_file (str): Path to input video file
        output_file (str): Path to output audio file (optional)
        **kwargs: Additional arguments for audio extraction
    
    Returns:
        str: Path to the extracted audio file
    """
    try:
        # Load video file
        logging.info(f"Loading video: {input_file}")
        video = VideoFileClip(input_file)
        
        # Check if video has audio
        if video.audio is None:
            raise ValueError("Video file has no audio track")
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            audio_format = kwargs.get('format', 'mp3')
            output_file = input_path.with_suffix(f'.{audio_format}')
        
        # Extract audio with specified parameters
        audio = video.audio
        
        # Apply audio modifications if specified
        if 'start_time' in kwargs or 'end_time' in kwargs:
            start = kwargs.get('start_time', 0)
            end = kwargs.get('end_time', audio.duration)
            audio = audio.subclip(start, end)
            logging.info(f"Trimmed audio from {start}s to {end}s")
        
        if 'volume' in kwargs:
            audio = audio.volumex(kwargs['volume'])
            logging.info(f"Applied volume multiplier: {kwargs['volume']}")
        
        # Prepare write_audiofile arguments
        write_args = {
            'filename': str(output_file),
            'verbose': kwargs.get('verbose', False),
            'logger': None if not kwargs.get('verbose', False) else 'bar'
        }
        
        # Audio quality settings
        if 'bitrate' in kwargs:
            write_args['bitrate'] = kwargs['bitrate']
        
        if 'fps' in kwargs:
            write_args['fps'] = kwargs['fps']
        
        if 'nbytes' in kwargs:
            write_args['nbytes'] = kwargs['nbytes']
        
        if 'codec' in kwargs:
            write_args['codec'] = kwargs['codec']
        
        if 'ffmpeg_params' in kwargs:
            write_args['ffmpeg_params'] = kwargs['ffmpeg_params']
        
        # Write audio file
        logging.info(f"Extracting audio to: {output_file}")
        audio.write_audiofile(**write_args)
        
        # Clean up
        audio.close()
        video.close()
        
        logging.info(f"Audio extraction completed: {output_file}")
        return str(output_file)
        
    except Exception as e:
        logging.error(f"Error extracting audio: {str(e)}")
        raise


def batch_extract(input_dir, output_dir=None, **kwargs):
    """
    Extract audio from multiple video files in a directory.
    
    Args:
        input_dir (str): Directory containing video files
        output_dir (str): Output directory for audio files
        **kwargs: Additional arguments for audio extraction
    
    Returns:
        list: List of extracted audio file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Setup output directory
    if output_dir is None:
        output_path = input_path / "extracted_audio"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    
    # Find video files
    video_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in video_extensions]
    
    if not video_files:
        logging.warning(f"No video files found in {input_dir}")
        return []
    
    extracted_files = []
    
    for video_file in video_files:
        try:
            audio_format = kwargs.get('format', 'mp3')
            output_file = output_path / f"{video_file.stem}.{audio_format}"
            
            logging.info(f"Processing: {video_file.name}")
            result = extract_audio(str(video_file), str(output_file), **kwargs)
            extracted_files.append(result)
            
        except Exception as e:
            logging.error(f"Failed to process {video_file.name}: {str(e)}")
            continue
    
    return extracted_files


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract audio from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python video_audio_extractor.py input.mp4
  
  # Specify output file and format
  python video_audio_extractor.py input.mp4 -o output.wav -f wav
  
  # Extract with custom bitrate and trim
  python video_audio_extractor.py input.mp4 -b 192k -s 30 -e 120
  
  # Batch extraction from directory
  python video_audio_extractor.py -d videos/ -od audio/ -f mp3
  
  # High quality extraction with custom codec
  python video_audio_extractor.py input.mp4 -c libmp3lame -b 320k
        """)
    
    # Input/Output arguments
    parser.add_argument('input', nargs='?', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output audio file path')
    parser.add_argument('-d', '--directory', help='Input directory for batch processing')
    parser.add_argument('-od', '--output-dir', help='Output directory for batch processing')
    
    # Audio format and quality
    parser.add_argument('-f', '--format', default='mp3', 
                       choices=['mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'],
                       help='Output audio format (default: mp3)')
    parser.add_argument('-b', '--bitrate', help='Audio bitrate (e.g., 128k, 192k, 320k)')
    parser.add_argument('-c', '--codec', help='Audio codec (e.g., libmp3lame, aac, flac)')
    parser.add_argument('--fps', type=int, help='Audio sample rate (Hz)')
    parser.add_argument('--nbytes', type=int, help='Number of bytes per sample')
    
    # Audio processing
    parser.add_argument('-s', '--start-time', type=float, help='Start time in seconds')
    parser.add_argument('-e', '--end-time', type=float, help='End time in seconds')
    parser.add_argument('--volume', type=float, default=1.0, 
                       help='Volume multiplier (default: 1.0)')
    
    # Advanced options
    parser.add_argument('--ffmpeg-params', nargs='+', 
                       help='Additional FFmpeg parameters')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    
    # Logging and verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress bars and non-error output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose and not args.quiet)
    
    # Validate arguments
    if not args.input and not args.directory:
        parser.error("Either input file or directory must be specified")
    
    if args.input and args.directory:
        parser.error("Cannot specify both input file and directory")
    
    # Prepare extraction arguments
    extract_kwargs = {
        'format': args.format,
        'verbose': args.verbose and not args.quiet,
    }
    
    # Add optional parameters
    if args.bitrate:
        extract_kwargs['bitrate'] = args.bitrate
    if args.codec:
        extract_kwargs['codec'] = args.codec
    if args.fps:
        extract_kwargs['fps'] = args.fps
    if args.nbytes:
        extract_kwargs['nbytes'] = args.nbytes
    if args.start_time is not None:
        extract_kwargs['start_time'] = args.start_time
    if args.end_time is not None:
        extract_kwargs['end_time'] = args.end_time
    if args.volume != 1.0:
        extract_kwargs['volume'] = args.volume
    if args.ffmpeg_params:
        extract_kwargs['ffmpeg_params'] = args.ffmpeg_params
    
    try:
        if args.directory:
            # Batch processing
            extracted_files = batch_extract(
                args.directory, 
                args.output_dir, 
                **extract_kwargs
            )
            
            if extracted_files:
                print(f"\nSuccessfully extracted audio from {len(extracted_files)} files:")
                for file_path in extracted_files:
                    print(f"  - {file_path}")
            else:
                print("No files were processed successfully.")
                
        else:
            # Single file processing
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
            
            # Check if output file exists
            if args.output and os.path.exists(args.output) and not args.overwrite:
                response = input(f"Output file '{args.output}' exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Operation cancelled.")
                    return
            
            output_file = extract_audio(args.input, args.output, **extract_kwargs)
            print(f"Audio extracted successfully: {output_file}")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()