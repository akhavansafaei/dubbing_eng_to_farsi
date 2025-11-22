# Video Dubber - Usage Examples

## 🎬 Complete Usage Examples

### Example 1: Basic Educational Video Dubbing

```bash
# Input: English lecture video
# Output: Persian dubbed version

python video_dubber.py "machine_learning_lecture.mp4" "machine_learning_lecture_persian.mp4"
```

**Expected Output:**
```
============================================================
UNIFIED VIDEO DUBBING PIPELINE
============================================================

Step 1: Transcribing video: machine_learning_lecture.mp4
Loading Whisper base.en model...
OpenAI-Whisper model loaded successfully
Transcription completed in 45.23 seconds
SRT file saved: ./dubbing_work_machine_learning_lecture/transcripts/machine_learning_lecture.srt
Total duration: 1847.50 seconds
Number of segments: 156

✅ Step 1 completed: Video transcription

Step 2: Translating to Persian...
Found 156 subtitle segments.
Generating transcript summary...
Summary generated: This lecture covers fundamental concepts of machine learning including supervised learning, unsupervised learning...
Translating: 100%|██████████| 156/156 [08:34<00:00,  3.30it/s]
Persian translation saved: ./dubbing_work_machine_learning_lecture/machine_learning_lecture_persian.srt

✅ Step 2 completed: Persian translation

Step 3: Refining Persian translation...
Found 156 segments to refine.
Refining subtitles: 100%|██████████| 156/156 [12:45<00:00,  4.91s/it]
Refined Persian SRT saved: ./dubbing_work_machine_learning_lecture/machine_learning_lecture_refined.srt

✅ Step 3 completed: Persian refinement

Please check the refined Persian SRT file at: ./dubbing_work_machine_learning_lecture/machine_learning_lecture_refined.srt
Review the translation quality and make sure it's acceptable.

Is the Persian translation approved? (y/n): y

Step 4: Generating Persian TTS audio...
Initializing Text-to-Speech engine...
TTS engine initialized successfully.
Found 156 subtitle segments to process.
Generating Audio: 100%|██████████| 156/156 [18:23<00:00,  7.08s/it]
Audio generation completed!

✅ Step 4 completed: TTS audio generation

Step 5: Adjusting SRT timing based on audio duration...
Processing 156 subtitle segments...
Adjusting Timings: 100%|██████████| 156/156 [00:03<00:00, 52.33it/s]
Adjusted SRT file saved: ./dubbing_work_machine_learning_lecture/machine_learning_lecture_adjusted.srt
Adjustment report saved: ./dubbing_work_machine_learning_lecture/machine_learning_lecture_adjustment_report.csv

✅ Step 5 completed: Timing adjustment

Step 6: Creating final dubbed video...
Running FFmpeg command...
FFmpeg completed successfully
Final dubbed video created: machine_learning_lecture_persian.mp4

✅ Step 6 completed: Final video creation

============================================================
🎉 DUBBING PIPELINE COMPLETED SUCCESSFULLY!
Total processing time: 42.3 minutes
Output video: machine_learning_lecture_persian.mp4
Work directory: ./dubbing_work_machine_learning_lecture
============================================================

Do you want to clean up temporary files? (y/n): y
Keep audio files? (y/n): n
Removed audio directory: ./dubbing_work_machine_learning_lecture/audio
Cleanup completed

🎬 Success! Dubbed video saved as: machine_learning_lecture_persian.mp4
```

### Example 2: Batch Processing Multiple Videos

```bash
#!/bin/bash
# batch_dub.sh - Process multiple videos

# Create list of videos to process
videos=(
    "intro_to_ai.mp4"
    "neural_networks.mp4"
    "deep_learning.mp4"
)

# Process each video
for video in "${videos[@]}"; do
    echo "Processing: $video"
    
    # Extract filename without extension
    filename=$(basename "$video" .mp4)
    
    # Run dubbing
    python video_dubber.py "$video" "${filename}_persian.mp4"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully dubbed: $video"
    else
        echo "❌ Failed to dub: $video"
    fi
    
    echo "---"
done

echo "Batch processing completed!"
```

**Run the batch script:**
```bash
chmod +x batch_dub.sh
./batch_dub.sh
```

### Example 3: Processing with Custom Directory Structure

```bash
# Organize files in directories
mkdir -p input_videos output_videos

# Move input videos
mv *.mp4 input_videos/

# Process with full paths
python video_dubber.py "input_videos/presentation.mp4" "output_videos/presentation_persian.mp4"
```

### Example 4: Processing Long Videos (Segmented Approach)

For very long videos (>2 hours), consider splitting first:

```bash
# Split video into 30-minute segments
ffmpeg -i "long_documentary.mp4" -c copy -map 0 -segment_time 1800 -f segment -reset_timestamps 1 "segment_%03d.mp4"

# Process each segment
for segment in segment_*.mp4; do
    python video_dubber.py "$segment" "dubbed_$segment"
done

# Combine dubbed segments
ffmpeg -f concat -safe 0 -i <(for f in dubbed_segment_*.mp4; do echo "file '$PWD/$f'"; done) -c copy "long_documentary_persian.mp4"
```

## 🔧 Advanced Configuration Examples

### Example 5: Custom API Key Management

```python
# custom_dubber.py - Using the script as a library with custom settings

from video_dubber import VideoDubber
import os

def custom_dubbing_workflow():
    # Set up custom API keys
    api_keys = [
        "your_primary_api_key",
        "your_secondary_api_key",
        "your_tertiary_api_key"
    ]
    
    # Create keys file
    with open("keys.txt", "w") as f:
        for key in api_keys:
            f.write(f"{key}\n")
    
    # Initialize dubber
    dubber = VideoDubber("input.mp4", "output.mp4")
    
    # Run with custom error handling
    try:
        success = dubber.run_full_pipeline()
        if success:
            print("✅ Custom dubbing completed successfully!")
        else:
            print("❌ Custom dubbing failed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Custom cleanup
    dubber.cleanup_temp_files(keep_audio=False)

if __name__ == "__main__":
    custom_dubbing_workflow()
```

### Example 6: Processing with Quality Checks

```bash
#!/bin/bash
# quality_check_dub.sh - Dubbing with quality validation

input_video="$1"
output_video="$2"

if [ -z "$input_video" ] || [ -z "$output_video" ]; then
    echo "Usage: $0 input_video.mp4 output_video.mp4"
    exit 1
fi

echo "Starting quality-checked dubbing process..."

# Check input video properties
echo "Input video information:"
ffprobe -v quiet -show_format -show_streams "$input_video" | grep -E "(duration|bit_rate|width|height)"

# Run dubbing
python video_dubber.py "$input_video" "$output_video"

# Check if output was created
if [ -f "$output_video" ]; then
    echo "✅ Output video created successfully"
    
    # Check output video properties
    echo "Output video information:"
    ffprobe -v quiet -show_format -show_streams "$output_video" | grep -E "(duration|bit_rate|width|height)"
    
    # Compare durations
    input_duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$input_video")
    output_duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$output_video")
    
    echo "Duration comparison:"
    echo "Input: ${input_duration}s"
    echo "Output: ${output_duration}s"
    
    # Check if durations are similar (within 5% difference)
    python3 -c "
import sys
input_dur = float('$input_duration')
output_dur = float('$output_duration')
diff = abs(input_dur - output_dur) / input_dur * 100
if diff < 5:
    print('✅ Duration check passed (difference: {:.2f}%)'.format(diff))
    sys.exit(0)
else:
    print('⚠️  Duration check failed (difference: {:.2f}%)'.format(diff))
    sys.exit(1)
"
else
    echo "❌ Output video was not created"
    exit 1
fi
```

## 📁 File Management Examples

### Example 7: Organized Workflow

```bash
#!/bin/bash
# organized_workflow.sh - Complete project organization

project_name="my_video_project"
input_file="$1"

# Create project structure
mkdir -p "$project_name"/{input,output,logs,temp}

# Copy input to project directory
cp "$input_file" "$project_name/input/"
input_basename=$(basename "$input_file")

# Run dubbing with organized paths
cd "$project_name"
python ../video_dubber.py "input/$input_basename" "output/dubbed_$input_basename" 2>&1 | tee "logs/dubbing_log_$(date +%Y%m%d_%H%M%S).txt"

# Move work directory to temp
if [ -d "../dubbing_work_${input_basename%.*}" ]; then
    mv "../dubbing_work_${input_basename%.*}" "temp/"
fi

echo "Project organized in: $project_name/"
echo "├── input/           # Original video"
echo "├── output/          # Dubbed video"
echo "├── logs/            # Processing logs"
echo "└── temp/            # Temporary files"
```

### Example 8: Resume Interrupted Processing

```python
# resume_dubbing.py - Resume interrupted dubbing process

import os
import sys
from video_dubber import VideoDubber

def resume_dubbing(video_path, output_path):
    """Resume dubbing from where it left off"""
    
    dubber = VideoDubber(video_path, output_path)
    
    # Check what steps are already completed
    steps_completed = []
    
    if os.path.exists(dubber.srt_original):
        steps_completed.append("transcription")
        print("✅ Found existing transcription")
    
    if os.path.exists(dubber.srt_persian):
        steps_completed.append("translation")
        print("✅ Found existing Persian translation")
    
    if os.path.exists(dubber.srt_refined):
        steps_completed.append("refinement")
        print("✅ Found existing refined translation")
    
    if os.path.exists(dubber.audio_dir) and os.listdir(dubber.audio_dir):
        steps_completed.append("tts")
        print("✅ Found existing TTS audio files")
    
    if os.path.exists(dubber.srt_adjusted):
        steps_completed.append("timing")
        print("✅ Found existing timing adjustments")
    
    # Resume from next step
    if "transcription" not in steps_completed:
        print("Starting from Step 1: Transcription")
        dubber.run_full_pipeline()
    elif "translation" not in steps_completed:
        print("Resuming from Step 2: Translation")
        if dubber.translate_to_persian():
            dubber.refine_persian_srt()
            dubber.generate_tts_audio()
            dubber.adjust_srt_timing()
            dubber.create_final_dubbed_video()
    elif "refinement" not in steps_completed:
        print("Resuming from Step 3: Refinement")
        if dubber.refine_persian_srt():
            dubber.generate_tts_audio()
            dubber.adjust_srt_timing()
            dubber.create_final_dubbed_video()
    elif "tts" not in steps_completed:
        print("Resuming from Step 4: TTS Generation")
        if dubber.generate_tts_audio():
            dubber.adjust_srt_timing()
            dubber.create_final_dubbed_video()
    elif "timing" not in steps_completed:
        print("Resuming from Step 5: Timing Adjustment")
        if dubber.adjust_srt_timing():
            dubber.create_final_dubbed_video()
    else:
        print("Resuming from Step 6: Final Video Creation")
        dubber.create_final_dubbed_video()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resume_dubbing.py input_video.mp4 output_video.mp4")
        sys.exit(1)
    
    resume_dubbing(sys.argv[1], sys.argv[2])
```

## 🎯 Real-World Scenarios

### Example 9: Educational Content Creator

```bash
#!/bin/bash
# educational_content_pipeline.sh

# Scenario: Teacher wants to dub educational videos for Persian-speaking students

course_videos=(
    "lesson_01_introduction.mp4"
    "lesson_02_basics.mp4"
    "lesson_03_advanced.mp4"
)

# Create course directory
mkdir -p "persian_course"

for video in "${course_videos[@]}"; do
    echo "Processing educational video: $video"
    
    # Extract lesson number
    lesson_num=$(echo "$video" | grep -o '[0-9]\+' | head -1)
    
    # Dub the video
    python video_dubber.py "$video" "persian_course/درس_${lesson_num}_فارسی.mp4"
    
    # Create subtitle files for students
    work_dir="dubbing_work_$(basename "$video" .mp4)"
    if [ -d "$work_dir" ]; then
        cp "$work_dir/$(basename "$video" .mp4)_refined.srt" "persian_course/درس_${lesson_num}_زیرنویس.srt"
    fi
done

echo "Educational course dubbing completed!"
echo "Files created in: persian_course/"
```

### Example 10: Content Localization Company

```python
# localization_pipeline.py - Professional localization workflow

import os
import json
import csv
from datetime import datetime
from video_dubber import VideoDubber

class LocalizationPipeline:
    def __init__(self, project_name):
        self.project_name = project_name
        self.project_dir = f"projects/{project_name}"
        self.setup_project_structure()
        self.job_log = []
    
    def setup_project_structure(self):
        """Create professional project structure"""
        dirs = [
            f"{self.project_dir}/source",
            f"{self.project_dir}/output",
            f"{self.project_dir}/subtitles",
            f"{self.project_dir}/audio",
            f"{self.project_dir}/reports",
            f"{self.project_dir}/temp"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def process_video(self, video_path, client_name=""):
        """Process single video with full reporting"""
        start_time = datetime.now()
        
        # Setup paths
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = f"{self.project_dir}/output/{video_name}_persian.mp4"
        
        # Initialize dubber
        dubber = VideoDubber(video_path, output_path)
        
        # Process with error handling
        try:
            success = dubber.run_full_pipeline()
            
            if success:
                # Move files to organized structure
                self.organize_output_files(dubber, video_name)
                
                # Generate report
                self.generate_job_report(video_name, client_name, start_time, success=True)
                
                return True
            else:
                self.generate_job_report(video_name, client_name, start_time, success=False)
                return False
                
        except Exception as e:
            self.generate_job_report(video_name, client_name, start_time, success=False, error=str(e))
            return False
    
    def organize_output_files(self, dubber, video_name):
        """Organize output files in professional structure"""
        import shutil
        
        # Move subtitle files
        subtitle_files = [
            (dubber.srt_original, f"{self.project_dir}/subtitles/{video_name}_english.srt"),
            (dubber.srt_refined, f"{self.project_dir}/subtitles/{video_name}_persian.srt"),
            (dubber.srt_adjusted, f"{self.project_dir}/subtitles/{video_name}_adjusted.srt")
        ]
        
        for src, dst in subtitle_files:
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Move audio files
        if os.path.exists(dubber.audio_dir):
            shutil.copytree(dubber.audio_dir, f"{self.project_dir}/audio/{video_name}_audio", dirs_exist_ok=True)
        
        # Move reports
        if os.path.exists(dubber.adjustment_report):
            shutil.copy2(dubber.adjustment_report, f"{self.project_dir}/reports/{video_name}_timing_report.csv")
    
    def generate_job_report(self, video_name, client_name, start_time, success=True, error=None):
        """Generate detailed job report"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        report_data = {
            "video_name": video_name,
            "client_name": client_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": round(duration, 2),
            "success": success,
            "error": error or ""
        }
        
        self.job_log.append(report_data)
        
        # Save individual report
        report_file = f"{self.project_dir}/reports/{video_name}_job_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def generate_project_summary(self):
        """Generate project summary report"""
        if not self.job_log:
            return
        
        summary = {
            "project_name": self.project_name,
            "total_jobs": len(self.job_log),
            "successful_jobs": sum(1 for job in self.job_log if job["success"]),
            "failed_jobs": sum(1 for job in self.job_log if not job["success"]),
            "total_processing_time": sum(job["duration_minutes"] for job in self.job_log),
            "generated_at": datetime.now().isoformat(),
            "jobs": self.job_log
        }
        
        # Save summary
        summary_file = f"{self.project_dir}/project_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Project Summary:")
        print(f"  Total Jobs: {summary['total_jobs']}")
        print(f"  Successful: {summary['successful_jobs']}")
        print(f"  Failed: {summary['failed_jobs']}")
        print(f"  Total Time: {summary['total_processing_time']:.1f} minutes")

# Usage example
if __name__ == "__main__":
    # Initialize project
    pipeline = LocalizationPipeline("client_education_videos_2025")
    
    # Process multiple videos
    videos = [
        ("video1.mp4", "ABC Education"),
        ("video2.mp4", "ABC Education"),
        ("video3.mp4", "ABC Education")
    ]
    
    for video_path, client in videos:
        pipeline.process_video(video_path, client)
    
    # Generate final summary
    pipeline.generate_project_summary()
```

These examples demonstrate various real-world scenarios and use cases for the video dubbing script, from simple single-video processing to complex professional workflows with full project management and reporting capabilities.