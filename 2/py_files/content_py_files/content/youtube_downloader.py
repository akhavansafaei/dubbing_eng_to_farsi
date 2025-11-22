#!/usr/bin/env python3
"""
YouTube Video Downloader
A comprehensive script to download YouTube videos with various options.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed. Install it with: pip install yt-dlp")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos with various options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s -q 720p -o ~/Downloads "https://youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --audio-only --format mp3 "https://youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --list-formats "https://youtube.com/watch?v=dQw4w9WgXcQ"
        """
    )
    
    # Required argument
    parser.add_argument(
        "url",
        help="YouTube video URL to download"
    )
    
    # Quality and format options
    parser.add_argument(
        "-q", "--quality",
        choices=["144p", "240p", "360p", "480p", "720p", "1080p", "1440p", "2160p", "best", "worst"],
        default="best",
        help="Video quality (default: best)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["mp4", "mkv", "webm", "mp3", "m4a", "wav"],
        default="mp4",
        help="Output format (default: mp4)"
    )
    
    # Audio options
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only"
    )
    
    parser.add_argument(
        "--audio-quality",
        choices=["best", "worst", "128", "192", "256", "320"],
        default="best",
        help="Audio quality in kbps (default: best)"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: current directory)"
    )
    
    parser.add_argument(
        "--filename",
        help="Custom filename template (use yt-dlp format)"
    )
    
    # Subtitle options
    parser.add_argument(
        "--subtitles",
        action="store_true",
        help="Download subtitles"
    )
    
    parser.add_argument(
        "--subtitle-lang",
        default="en",
        help="Subtitle language code (default: en)"
    )
    
    parser.add_argument(
        "--auto-subtitles",
        action="store_true",
        help="Download auto-generated subtitles"
    )
    
    # Playlist options
    parser.add_argument(
        "--playlist",
        action="store_true",
        help="Download entire playlist"
    )
    
    parser.add_argument(
        "--playlist-start",
        type=int,
        help="Playlist video to start at (default: 1)"
    )
    
    parser.add_argument(
        "--playlist-end",
        type=int,
        help="Playlist video to end at"
    )
    
    # Information options
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List available formats and exit"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show video information and exit"
    )
    
    # Download options
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download, just show what would be downloaded"
    )
    
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_download",
        help="Continue partial downloads"
    )
    
    parser.add_argument(
        "--rate-limit",
        help="Maximum download rate (e.g., 50K, 4.2M)"
    )
    
    # Metadata options
    parser.add_argument(
        "--embed-metadata",
        action="store_true",
        help="Embed metadata in the video file"
    )
    
    parser.add_argument(
        "--embed-thumbnail",
        action="store_true",
        help="Embed thumbnail in the video file"
    )
    
    parser.add_argument(
        "--write-thumbnail",
        action="store_true",
        help="Write thumbnail image to disk"
    )
    
    parser.add_argument(
        "--write-description",
        action="store_true",
        help="Write video description to file"
    )
    
    # Verbose options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.audio_only and args.format in ["mp4", "mkv", "webm"]:
        args.format = "mp3"
    
    # Build yt-dlp options
    ydl_opts = {}
    
    # Output template
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        if args.filename:
            ydl_opts['outtmpl'] = str(output_path / args.filename)
        else:
            ydl_opts['outtmpl'] = str(output_path / '%(title)s.%(ext)s')
    elif args.filename:
        ydl_opts['outtmpl'] = args.filename
    
    # Format selection
    if args.audio_only:
        if args.audio_quality == "best":
            ydl_opts['format'] = 'bestaudio/best'
        elif args.audio_quality == "worst":
            ydl_opts['format'] = 'worstaudio/worst'
        else:
            ydl_opts['format'] = f'bestaudio[abr<={args.audio_quality}]/best[abr<={args.audio_quality}]'
    else:
        if args.quality == "best":
            ydl_opts['format'] = 'best[ext=mp4]/best'
        elif args.quality == "worst":
            ydl_opts['format'] = 'worst[ext=mp4]/worst'
        else:
            height = args.quality.replace('p', '')
            ydl_opts['format'] = f'best[height<={height}][ext=mp4]/best[height<={height}]'
    
    # Post-processing options
    postprocessors = []
    
    if args.format in ["mp3", "m4a", "wav"]:
        postprocessors.append({
            'key': 'FFmpegExtractAudio',
            'preferredcodec': args.format,
            'preferredquality': args.audio_quality if args.audio_quality.isdigit() else '192',
        })
    elif args.format in ["mkv", "webm"] and not args.audio_only:
        postprocessors.append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': args.format,
        })
    
    if args.embed_metadata:
        postprocessors.append({'key': 'FFmpegMetadata'})
    
    if args.embed_thumbnail:
        postprocessors.append({'key': 'EmbedThumbnail'})
    
    if postprocessors:
        ydl_opts['postprocessors'] = postprocessors
    
    # Subtitle options
    if args.subtitles:
        ydl_opts['writesubtitles'] = True
        ydl_opts['subtitleslangs'] = [args.subtitle_lang]
    
    if args.auto_subtitles:
        ydl_opts['writeautomaticsub'] = True
        ydl_opts['subtitleslangs'] = [args.subtitle_lang]
    
    # Thumbnail and description options
    if args.write_thumbnail:
        ydl_opts['writethumbnail'] = True
    
    if args.write_description:
        ydl_opts['writedescription'] = True
    
    # Playlist options
    if not args.playlist:
        ydl_opts['noplaylist'] = True
    else:
        if args.playlist_start:
            ydl_opts['playliststart'] = args.playlist_start
        if args.playlist_end:
            ydl_opts['playlistend'] = args.playlist_end
    
    # Download options
    if args.continue_download:
        ydl_opts['continuedl'] = True
    
    if args.rate_limit:
        ydl_opts['ratelimit'] = args.rate_limit
    
    # Verbosity options
    if args.quiet:
        ydl_opts['quiet'] = True
    elif args.verbose:
        ydl_opts['verbose'] = True
    
    # Information-only options
    if args.list_formats:
        ydl_opts['listformats'] = True
    
    if args.info:
        ydl_opts['dump_single_json'] = True
    
    if args.no_download:
        ydl_opts['simulate'] = True
    
    # Execute download
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.url])
        
        if not (args.list_formats or args.info or args.no_download):
            print(f"\n✅ Download completed successfully!")
            if args.output:
                print(f"📁 Files saved to: {args.output}")
    
    except yt_dlp.utils.DownloadError as e:
        print(f"❌ Download error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()