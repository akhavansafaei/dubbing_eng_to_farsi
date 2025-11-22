import os
import sys
import subprocess
import platform
from pathlib import Path

def install_requirements():
    """Install required packages"""
    try:
        import yt_dlp
        print("✓ yt-dlp already installed")
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        import yt_dlp
        print("✓ yt-dlp installed successfully")

def get_download_path():
    """Get appropriate download path based on platform"""
    system = platform.system()
    
    if system == "Windows":
        # Windows: Downloads folder
        downloads_path = Path.home() / "Downloads" / "YouTube_Downloads"
    else:
        # Linux/Colab: Current directory or content folder
        if 'google.colab' in sys.modules:
            downloads_path = Path("/content/YouTube_Downloads")
        else:
            downloads_path = Path.cwd() / "YouTube_Downloads"
    
    # Create directory if it doesn't exist
    downloads_path.mkdir(parents=True, exist_ok=True)
    return str(downloads_path)

def get_format_string(quality):
    """Get proper format string for different resolutions"""
    # Define standard resolutions with their proper dimensions
    resolution_formats = {
        '144p': 'best[height<=144]/best[width<=256]/best',
        '240p': 'best[height<=240]/best[width<=426]/best',
        '360p': 'best[height<=360]/best[width<=640]/best',
        '480p': 'best[height<=480]/best[width<=854]/best',
        '720p': 'best[height<=720]/best[width<=1280]/best',
        '1080p': 'best[height<=1080]/best[width<=1920]/best',
        '1440p': 'best[height<=1440]/best[width<=2560]/best',
        '2160p': 'best[height<=2160]/best[width<=3840]/best'  # 4K
    }
    
    if quality == 'best':
        return 'best[height<=1080][ext=mp4]/best[height<=1080]/best[ext=mp4]/best'
    elif quality == 'worst':
        return 'worst[ext=mp4]/worst'
    elif quality in resolution_formats:
        # Use the predefined format for standard resolutions
        base_format = resolution_formats[quality]
        # Add mp4 preference and fallbacks
        return f'{base_format}[ext=mp4]/{base_format}/best[ext=mp4]/best'
    else:
        # For custom resolutions like '720p', extract height
        try:
            height = quality.replace('p', '')
            return f'best[height<={height}][ext=mp4]/best[height<={height}]/best[ext=mp4]/best'
        except:
            return 'best[ext=mp4]/best'

def download_video(url, quality='best', audio_only=False, output_path=None):
    """
    Download YouTube video
    
    Args:
        url (str): YouTube video URL
        quality (str): Video quality ('best', 'worst', '144p', '240p', '360p', '480p', '720p', '1080p', '1440p', '2160p', etc.)
        audio_only (bool): Download only audio
        output_path (str): Custom output path
    """
    try:
        import yt_dlp
        
        if output_path is None:
            output_path = get_download_path()
        
        # Configure yt-dlp options with better error handling
        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'ignoreerrors': False,
            'no_warnings': False,
            'extractaudio': audio_only,
            'audioformat': 'mp3' if audio_only else None,
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,
            'keep_fragments': False,
            'abort_on_unavailable_fragment': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'cookiefile': None,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'referer': 'https://www.youtube.com/',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        }
        
        # Format selection with fallbacks
        if audio_only:
            ydl_opts['format'] = 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else:
            ydl_opts['format'] = get_format_string(quality)
        
        print(f"📁 Download location: {output_path}")
        print(f"🎯 Quality: {quality}")
        print(f"🎵 Audio only: {audio_only}")
        print(f"🔧 Format string: {ydl_opts['format']}")
        print("⏳ Starting download...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                print(f"📺 Title: {title}")
                print(f"⏱️ Duration: {duration // 60}:{duration % 60:02d}")
                
                # Show available formats for debugging
                formats = info.get('formats', [])
                print(f"📊 Available formats: {len(formats)} found")
                
                # Show some format examples
                for fmt in formats[-5:]:  # Show last 5 formats
                    height = fmt.get('height', 'N/A')
                    width = fmt.get('width', 'N/A')
                    ext = fmt.get('ext', 'N/A')
                    print(f"   • {width}×{height} ({ext})")
                
                # Download the video
                ydl.download([url])
                
                # Check if file was actually downloaded
                downloaded_files = list(Path(output_path).glob(f"*{title[:30]}*"))
                if downloaded_files:
                    print(f"✅ Download completed successfully!")
                    print(f"📄 File saved as: {downloaded_files[0].name}")
                    return True
                else:
                    print("❌ Download may have failed - no file found")
                    return False
                    
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                if "403" in error_msg or "Forbidden" in error_msg:
                    print("❌ HTTP 403 Forbidden error occurred.")
                    print("💡 This usually happens due to:")
                    print("   • YouTube rate limiting")
                    print("   • IP blocking")
                    print("   • Video restrictions")
                    print("🔄 Trying alternative approach...")
                    
                    # Try with different user agent and minimal options
                    fallback_opts = {
                        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
                        'format': 'worst' if not audio_only else 'worstaudio',
                        'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                        'extractor_retries': 2,
                        'retries': 2,
                    }
                    
                    if audio_only:
                        fallback_opts['postprocessors'] = [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '128',
                        }]
                    
                    try:
                        with yt_dlp.YoutubeDL(fallback_opts) as fallback_ydl:
                            fallback_ydl.download([url])
                        print("✅ Downloaded using fallback method (lower quality)")
                        return True
                    except Exception as fallback_error:
                        print(f"❌ Fallback also failed: {str(fallback_error)}")
                        return False
                else:
                    print(f"❌ Download error: {error_msg}")
                    return False
                    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("💡 Troubleshooting tips:")
        print("   • Check if the URL is valid")
        print("   • Try again in a few minutes")
        print("   • Some videos may be region-restricted")
        print("   • Consider using a VPN if persistent issues occur")
        return False

def download_playlist(playlist_url, quality='best', audio_only=False, output_path=None):
    """Download entire YouTube playlist"""
    try:
        import yt_dlp
        
        if output_path is None:
            output_path = get_download_path()
        
        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(playlist_title)s/%(title)s.%(ext)s'),
            'ignoreerrors': True,
            'no_warnings': False,
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'referer': 'https://www.youtube.com/',
        }
        
        if audio_only:
            ydl_opts['format'] = 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else:
            ydl_opts['format'] = get_format_string(quality)
        
        print(f"📁 Download location: {output_path}")
        print("⏳ Starting playlist download...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Get playlist info first
                info = ydl.extract_info(playlist_url, download=False)
                playlist_title = info.get('title', 'Unknown Playlist')
                entries = info.get('entries', [])
                
                print(f"📋 Playlist: {playlist_title}")
                print(f"🎬 Videos found: {len(entries)}")
                
                # Download playlist
                ydl.download([playlist_url])
                
            except yt_dlp.utils.DownloadError as e:
                if "403" in str(e):
                    print("❌ HTTP 403 error - trying fallback method...")
                    fallback_opts = ydl_opts.copy()
                    fallback_opts['format'] = 'worst' if not audio_only else 'worstaudio'
                    fallback_opts['retries'] = 1
                    
                    with yt_dlp.YoutubeDL(fallback_opts) as fallback_ydl:
                        fallback_ydl.download([playlist_url])
                    print("✅ Downloaded using fallback method")
                else:
                    raise e
            
        print("✅ Playlist download completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Try downloading individual videos if playlist fails")
        return False

def get_video_info(url):
    """Get video information without downloading"""
    try:
        import yt_dlp
        
        ydl_opts = {'quiet': True}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'upload_date': info.get('upload_date', 'Unknown'),
                'description': info.get('description', '')[:200] + '...' if info.get('description') else 'No description'
            }
    except Exception as e:
        print(f"❌ Error getting video info: {str(e)}")
        return None

def list_available_formats(url):
    """List all available formats for a video"""
    try:
        import yt_dlp
        
        ydl_opts = {'quiet': True}
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            formats = info.get('formats', [])
            
            print(f"\n📊 Available formats for: {info.get('title', 'Unknown')}")
            print("=" * 80)
            print(f"{'Format ID':<12} {'Resolution':<12} {'FPS':<6} {'Codec':<8} {'Ext':<6} {'Size':<10}")
            print("-" * 80)
            
            for fmt in formats:
                format_id = fmt.get('format_id', 'N/A')
                height = fmt.get('height', 'N/A')
                width = fmt.get('width', 'N/A')
                fps = fmt.get('fps', 'N/A')
                vcodec = fmt.get('vcodec', 'N/A')[:7]
                ext = fmt.get('ext', 'N/A')
                filesize = fmt.get('filesize', 0)
                
                resolution = f"{width}×{height}" if width != 'N/A' and height != 'N/A' else 'N/A'
                size_mb = f"{filesize/1024/1024:.1f}MB" if filesize else 'N/A'
                
                print(f"{format_id:<12} {resolution:<12} {fps:<6} {vcodec:<8} {ext:<6} {size_mb:<10}")
            
            return formats
    except Exception as e:
        print(f"❌ Error listing formats: {str(e)}")
        return []

def interactive_downloader():
    """Interactive command-line interface"""
    print("🎬 YouTube Downloader")
    print("=" * 50)
    
    while True:
        print("\n📋 Options:")
        print("1. Download single video")
        print("2. Download playlist")
        print("3. Get video info")
        print("4. List available formats")
        print("5. Test URL accessibility")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            url = input("Enter YouTube video URL: ").strip()
            if not url:
                print("❌ Please enter a valid URL")
                continue
            
            # First test if URL is accessible
            print("🔍 Testing URL accessibility...")
            info = get_video_info(url)
            if not info:
                print("❌ Cannot access video. It may be restricted or unavailable.")
                continue
                
            print("\n🎯 Quality options:")
            print("1. Best quality (default)")
            print("2. 1080p (1920×1080)")
            print("3. 720p (1280×720)")
            print("4. 480p (854×480)")
            print("5. 360p (640×360)")
            print("6. 240p (426×240)")
            print("7. Worst quality (most compatible)")
            print("8. Audio only (MP3)")
            
            quality_choice = input("Select quality (1-8): ").strip()
            
            quality_map = {
                '2': '1080p',
                '3': '720p',
                '4': '480p', 
                '5': '360p',
                '6': '240p',
                '7': 'worst',
                '8': 'best'
            }
            
            if quality_choice in quality_map:
                quality = quality_map[quality_choice]
                audio_only = quality_choice == '8'
            else:
                quality = 'best'
                audio_only = False
            
            success = download_video(url, quality, audio_only)
            if not success:
                print("\n💡 Troubleshooting suggestions:")
                print("   • Try 'worst quality' option for better compatibility")
                print("   • Some videos may be region-restricted")
                print("   • Wait a few minutes and try again")
            
        elif choice == '2':
            url = input("Enter YouTube playlist URL: ").strip()
            if not url:
                print("❌ Please enter a valid URL")
                continue
                
            print("\n🎯 Quality options:")
            print("1. Best quality (default)")
            print("2. 1080p (1920×1080)")
            print("3. 720p (1280×720)")
            print("4. 480p (854×480)")
            print("5. 360p (640×360)")
            print("6. 240p (426×240)")
            print("7. Worst quality (most compatible)")
            print("8. Audio only (MP3)")
            
            quality_choice = input("Select quality (1-8): ").strip()
            
            quality_map = {
                '2': '1080p',
                '3': '720p',
                '4': '480p',
                '5': '360p',
                '6': '240p',
                '7': 'worst',
                '8': 'best'
            }
            
            if quality_choice in quality_map:
                quality = quality_map[quality_choice]
                audio_only = quality_choice == '8'
            else:
                quality = 'best'
                audio_only = False
            
            download_playlist(url, quality, audio_only)
            
        elif choice == '3':
            url = input("Enter YouTube video URL: ").strip()
            if not url:
                print("❌ Please enter a valid URL")
                continue
                
            info = get_video_info(url)
            if info:
                print(f"\n📺 Title: {info['title']}")
                print(f"👤 Uploader: {info['uploader']}")
                print(f"⏱️ Duration: {info['duration'] // 60}:{info['duration'] % 60:02d}")
                print(f"👀 Views: {info['view_count']:,}")
                print(f"📅 Upload Date: {info['upload_date']}")
                print(f"📝 Description: {info['description']}")
        
        elif choice == '4':
            url = input("Enter YouTube video URL: ").strip()
            if not url:
                print("❌ Please enter a valid URL")
                continue
            
            list_available_formats(url)
                
        elif choice == '5':
            url = input("Enter YouTube video URL to test: ").strip()
            if not url:
                print("❌ Please enter a valid URL")
                continue
                
            print("🔍 Testing URL accessibility...")
            info = get_video_info(url)
            if info:
                print("✅ URL is accessible!")
                print(f"📺 Title: {info['title']}")
                print(f"⏱️ Duration: {info['duration'] // 60}:{info['duration'] % 60:02d}")
            else:
                print("❌ URL is not accessible or video is restricted")
                
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main function"""
    print("🔧 Setting up YouTube Downloader...")
    
    # Install requirements
    install_requirements()
    
    # Show system info
    print(f"💻 System: {platform.system()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Download path: {get_download_path()}")
    
    # Check if running in Colab
    if 'google.colab' in sys.modules:
        print("🔗 Running in Google Colab")
        
        # Example usage for Colab
        print("\n📋 Example usage:")
        print("# Download a single video at 480p")
        print("download_video('https://www.youtube.com/watch?v=VIDEO_ID', '480p')")
        print("\n# Download audio only")
        print("download_video('https://www.youtube.com/watch?v=VIDEO_ID', audio_only=True)")
        print("\n# Download playlist at 720p")
        print("download_playlist('https://www.youtube.com/playlist?list=PLAYLIST_ID', '720p')")
        
        # Ask if user wants to start interactive mode
        start_interactive = input("\nStart interactive downloader? (y/n): ").strip().lower()
        if start_interactive == 'y':
            interactive_downloader()
    else:
        # Start interactive mode for local/Windows usage
        interactive_downloader()

if __name__ == "__main__":
    main()