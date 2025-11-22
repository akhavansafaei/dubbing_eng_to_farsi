#!/usr/bin/env python3

"""
Advanced SRT Refinement Script
Refines a Persian SRT file using global, local, and sequential context to improve fluency.
Then performs a final overlap refinement to ensure coherence across subtitle lines.
Usage: python srt_refiner.py input.srt context.txt [output.srt]
"""

import os
import sys
import time
import google.generativeai as genai
from tqdm import tqdm
import argparse
import re

# --- Configuration ---
# Loads API keys from 'keys.txt'.
try:
    with open("keys.txt", "r") as f:
        api_keys = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: `keys.txt` not found. Please create it and add your Google API keys.")
    api_keys = []

current_key_index = 5
LLM_STAGE1 = None  # For initial refinement
LLM_STAGE2 = None  # For final overlap refinement

def switch_api_key():
    """Cycles to the next API key in the list."""
    global current_key_index, LLM_STAGE1, LLM_STAGE2
    current_key_index = (current_key_index + 1) % len(api_keys)
    new_key = api_keys[current_key_index]
    os.environ["GOOGLE_API_KEY"] = new_key
    genai.configure(api_key=new_key)
    LLM_STAGE1 = genai.GenerativeModel('gemini-2.0-flash')
    LLM_STAGE2 = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    print(f"\nSwitched to API key ending in: ...{new_key[-4:]}")

def initialize_api():
    """Initializes the Gemini API with both models."""
    global LLM_STAGE1, LLM_STAGE2
    if not api_keys:
        return False
    os.environ["GOOGLE_API_KEY"] = api_keys[current_key_index]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    LLM_STAGE1 = genai.GenerativeModel('gemini-2.0-flash')
    LLM_STAGE2 = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    return True

def create_refinement_prompt(srt_line, full_context, original_prev, original_next, refined_prev):
    """Creates a multi-context prompt for refining an SRT line."""
   
    # Build context sections
    local_context_parts = []
    if original_prev:
        local_context_parts.append(f"Original Previous Line:\n{original_prev}")
    if original_next:
        local_context_parts.append(f"Original Next Line:\n{original_next}")
    local_context_section = "\n\n".join(local_context_parts) if local_context_parts else "No surrounding lines in the original SRT."

    refined_history_section = refined_prev if refined_prev else "This is the first line to be refined."

    return f"""
   

You are a Persian language expert specializing in subtitle editing for improved fluency. Refine the following subtitle line based on three levels of context:

---
CONTEXT 1: GLOBAL (The complete, ideal translation)
{full_context}
---
CONTEXT 2: LOCAL (The original surrounding subtitles)
{local_context_section}
---
CONTEXT 3: HISTORY (The immediately preceding line you just refined)
Previously Refined Line: {refined_history_section}
---

LINE TO REFINE:
{srt_line}
---

Instructions:
1. Rephrase the "LINE TO REFINE" to make it natural, grammatically correct, and cohesive with the "Previously Refined Line," without omitting any information. The length should remain almost identical to the original.
2. Ensure the refined line aligns in meaning with all three contexts.
3. **CRITICAL**: Do not significantly alter the duration or character count—maintain subtitle timing and readability.
4. Return **only** the refined Persian line, without any additional commentary.
5. If a sentence fragment belongs to the next subtitle line, move it accordingly to preserve semantic completeness and grammatical accuracy.
    """

def create_overlap_refinement_prompt(ten_lines):
    """Creates a prompt for final overlap refinement of 10 lines."""
    lines_text = "\n".join([f"LINE_{i+1}: {line}" for i, line in enumerate(ten_lines)])
   
    return f"""شما یک ویرایشگر حرفه‌ای زیرنویس فارسی هستید. در هر مرحله ۱۰ خط از یک متن ترجمه‌شده‌ی زیرنویس به شما داده می‌شود. این متن قرار است به صورت صوتی و توسط مدل تبدیل متن به گفتار (TTS) خوانده شود.

**مهم: دقیقاً همان تعداد خط ({len(ten_lines)} خط) را که دریافت کرده‌اید برگردانید.**

متن را با هدف خوانایی طبیعی و گویایی روان بازنویسی و اصلاح کنید.
لحن جمله‌ها باید طبیعی، محاوره‌ای ملایم، و مناسب گفتار باشد.
ساختار جمله‌ها را طوری تنظیم کنید که در گفتار واضح و راحت ادا شوند.
اشتباهات ناشی از تقسیم‌بندی زیرنویس‌ها را اصلاح کنید.
اگر عبارتی خوب و طبیعی است، تغییری در آن ندهید.
ویرایش‌های شما باید حداقلی اما مؤثر باشند.

**فرمت پاسخ:**
دقیقاً به همین فرمت پاسخ دهید:
LINE_1: [متن اصلاح شده خط ۱]
LINE_2: [متن اصلاح شده خط ۲]
LINE_3: [متن اصلاح شده خط ۳]
...
LINE_{len(ten_lines)}: [متن اصلاح شده خط {len(ten_lines)}]

زیرنویس‌های دریافتی:
{lines_text}

**نکات مهم:**
- دقیقاً همان تعداد خط را برگردانید که دریافت کردید
- از فرمت LINE_X استفاده کنید
- هیچ توضیح اضافی یا کامنت ندهید
- اگر خطی خالی است، آن را خالی برگردانید"""

def parse_overlap_response(response_text, expected_count):
    """
    Parses the LLM response for overlap refinement with multiple fallback strategies.
    Returns a list of refined lines or None if parsing fails completely.
    """
    if not response_text or response_text.strip() == "[Refinement Failed]":
        return None
        
    response_text = response_text.strip()
    refined_lines = []
    
    # Strategy 1: Parse LINE_X: format
    line_pattern = re.compile(r'^LINE_(\d+):\s*(.*)$', re.MULTILINE)
    matches = line_pattern.findall(response_text)
    
    if matches and len(matches) == expected_count:
        # Sort by line number and extract text
        matches.sort(key=lambda x: int(x[0]))
        refined_lines = [match[1].strip() for match in matches]
        return refined_lines
    
    # Strategy 2: Split by lines and remove numbering
    lines = response_text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove various numbering patterns
        line = re.sub(r'^LINE_\d+:\s*', '', line)
        line = re.sub(r'^\d+\.\s*', '', line)
        line = re.sub(r'^\d+\)\s*', '', line)
        line = re.sub(r'^\d+\s*[-–—]\s*', '', line)
        
        if line:
            processed_lines.append(line)
    
    if len(processed_lines) == expected_count:
        return processed_lines
    
    # Strategy 3: Try to extract exactly expected_count non-empty lines
    all_lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    # Remove common prefixes from all lines
    cleaned_lines = []
    for line in all_lines:
        # Remove numbering and prefixes
        clean_line = re.sub(r'^(LINE_\d+:|[۰-۹]+[\.:\-\)]|\d+[\.:\-\)]|[•\-\*])\s*', '', line)
        if clean_line:
            cleaned_lines.append(clean_line)
    
    if len(cleaned_lines) == expected_count:
        return cleaned_lines
    
    # Strategy 4: If we have more lines than expected, take the first expected_count
    if len(cleaned_lines) > expected_count:
        return cleaned_lines[:expected_count]
    
    # Strategy 5: If we have fewer lines, pad with empty strings
    if len(cleaned_lines) < expected_count:
        while len(cleaned_lines) < expected_count:
            cleaned_lines.append("")
        return cleaned_lines
    
    return None

def get_llm_response(prompt, stage=1, max_retries=5):
    """Gets a response from the appropriate LLM based on stage with retry logic."""
    # Select the appropriate model based on stage
    model = LLM_STAGE1 if stage == 1 else LLM_STAGE2
    model_name = "gemini-2.0-flash" if stage == 1 else "gemini-2.5-flash-preview-05-20"
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, request_options={'timeout': 120})
            return response.text.strip()
        except Exception as e:
            error_message = str(e).lower()
            print(f"\nError on attempt {attempt + 1} using {model_name}: {error_message}")
            if any(keyword in error_message for keyword in ["quota", "limit", "exceeded", "resource has been exhausted"]):
                print("Quota likely exceeded, switching API key...")
                switch_api_key()
                time.sleep(5)
            else:
                time.sleep(attempt + 5)

            if attempt == max_retries - 1:
                print(f"Failed to get a response after {max_retries} attempts.")
                return f"[Refinement Failed]"
    return "[Refinement Failed]"

def parse_srt_file(file_path):
    """Parses an SRT file into a list of subtitle segments."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2 and '-->' in lines[1]:
            try:
                segments.append({
                    'sequence': lines[0],
                    'start': lines[1].split(' --> ')[0].strip(),
                    'end': lines[1].split(' --> ')[1].strip(),
                    'text': '\n'.join(lines[2:]).strip()
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

def final_overlap_refinement(segments):
    """
    Performs final overlap refinement on the SRT segments using gemini-2.5-flash-preview-05-20.
    Processes 10 lines at a time with 5-line overlap.
    Each chunk reads the most up-to-date refined lines from previous processing.
    """
    print("\nStarting final overlap refinement using gemini-2.5-flash-preview-05-20...")
   
    # Extract just the text from segments for processing
    texts = [segment['text'] for segment in segments]
    refined_texts = texts.copy()  # Start with original texts
   
    # Process in overlapping chunks of 10 with 5-line steps
    start_idx = 0
    total_chunks = max(1, (len(texts) - 5) // 5 + 1) if len(texts) > 10 else 1
   
    with tqdm(total=total_chunks, desc="Final overlap refinement (Stage 2)") as pbar:
        while start_idx < len(texts):
            end_idx = min(start_idx + 10, len(texts))
            
            # CRITICAL: Use refined_texts (most up-to-date) instead of original texts
            current_chunk = refined_texts[start_idx:end_idx]
           
            # Skip if chunk is too small (less than 2 lines)
            if len(current_chunk) < 2:
                print(f"Skipping chunk {start_idx}-{end_idx-1}: too small ({len(current_chunk)} lines)")
                break
            
            chunk_size = len(current_chunk)
            print(f"\nProcessing chunk {start_idx}-{end_idx-1} ({chunk_size} lines)")
            
            # Get refined version of this chunk using Stage 2 model with retry logic
            max_chunk_retries = 5  # Increased retries
            refined_lines = None
            
            for retry in range(max_chunk_retries):
                prompt = create_overlap_refinement_prompt(current_chunk)
                refined_chunk_text = get_llm_response(prompt, stage=2)
                
                if refined_chunk_text and refined_chunk_text != "[Refinement Failed]":
                    # Parse the response back into individual lines
                    refined_lines = parse_overlap_response(refined_chunk_text, chunk_size)
                    
                    if refined_lines and len(refined_lines) == chunk_size:
                        print(f"✓ Successfully refined chunk {start_idx}-{end_idx-1}")
                        break
                    else:
                        actual_count = len(refined_lines) if refined_lines else 0
                        print(f"⚠ Retry {retry+1}/{max_chunk_retries}: Expected {chunk_size} lines, got {actual_count}")
                        print(f"Raw response preview: {refined_chunk_text[:200]}...")
                        if retry < max_chunk_retries - 1:
                            time.sleep(3)  # Wait before retry
                else:
                    print(f"⚠ Retry {retry+1}/{max_chunk_retries}: LLM response failed")
                    if retry < max_chunk_retries - 1:
                        time.sleep(3)
            
            # Apply refinements if successful, otherwise keep original
            if refined_lines and len(refined_lines) == chunk_size:
                for i, refined_line in enumerate(refined_lines):
                    actual_idx = start_idx + i
                    # Update refined_texts immediately for next chunk to use
                    refined_texts[actual_idx] = refined_line
                    
                print(f"✓ Applied refinements to lines {start_idx}-{end_idx-1}")
            else:
                print(f"⚠ Could not refine chunk {start_idx}-{end_idx-1} after {max_chunk_retries} retries.")
                print(f"⚠ Keeping original text for lines {start_idx}-{end_idx-1}")
                # Don't skip - keep the original text
           
            # Move to next chunk (5 lines forward for overlap)
            start_idx += 5
            pbar.update(1)
            time.sleep(0.5)  # Respect API rate limits
   
    # Update the segments with refined texts
    refined_segments = []
    for i, segment in enumerate(segments):
        refined_segments.append({**segment, 'text': refined_texts[i]})
   
    return refined_segments

def refine_srt_file(srt_path: str, context_path: str, output_path: str):
    """
    Refines an SRT file using global, local, and sequential context.
    Then performs final overlap refinement for coherence.
    """
    # 1. Read the full context file
    try:
        with open(context_path, 'r', encoding='utf-8') as f:
            full_context = f.read()
        print("Successfully loaded global context file.")
    except FileNotFoundError:
        print(f"Error: Context file not found at '{context_path}'")
        sys.exit(1)

    # 2. Parse the SRT file
    segments = parse_srt_file(srt_path)
    if not segments:
        print("No valid segments found in the SRT file. Exiting.")
        sys.exit(1)
    print(f"Found {len(segments)} segments to refine in {os.path.basename(srt_path)}.")

    # 3. Refine each segment sequentially using Stage 1 model (gemini-2.0-flash)
    print("Starting Stage 1 refinement using gemini-2.0-flash...")
    refined_segments = []
    refined_history = []
    for i, segment in enumerate(tqdm(segments, desc="Stage 1: Initial refinement")):
        if not segment['text'].strip():
            refined_text = ""
        else:
            # Gather all three types of context
            original_prev = segments[i-1]['text'] if i > 0 else None
            original_next = segments[i+1]['text'] if i < len(segments) - 1 else None
            refined_prev = refined_history[-1] if refined_history else None

            # Create prompt and get refined text using Stage 1 model
            prompt = create_refinement_prompt(
                segment['text'], full_context, original_prev, original_next, refined_prev
            )
            refined_text = get_llm_response(prompt, stage=1)
            time.sleep(0.5) # Respect API rate limits
       
        refined_segments.append({**segment, 'text': refined_text})
        refined_history.append(refined_text)

    # 4. Perform final overlap refinement using Stage 2 model (gemini-2.5-flash-preview-05-20)
    final_refined_segments = final_overlap_refinement(refined_segments)

    # 5. Write the final refined segments to the output file
    write_srt_file(final_refined_segments, output_path)
    print("\nComplete refinement process finished!")
    print(f"Stage 1 (Initial): gemini-2.0-flash")
    print(f"Stage 2 (Final overlap): gemini-2.5-flash-preview-05-20")
    print(f"Final refined SRT file saved as: {output_path}")

def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(description='Refine a Persian SRT file using multiple contexts and models.')
    parser.add_argument('input_srt', help='Path to the input Persian SRT file to be refined.')
    parser.add_argument('context_txt', help='Path to the .txt file containing the full, high-quality Persian translation.')
    parser.add_argument('output_srt', nargs='?', help='Path for the output refined SRT file (optional).')
   
    args = parser.parse_args()

    if not os.path.exists(args.input_srt):
        print(f"Error: Input SRT file not found: '{args.input_srt}'")
        sys.exit(1)
       
    if not os.path.exists(args.context_txt):
        print(f"Error: Context text file not found: '{args.context_txt}'")
        sys.exit(1)

    if not initialize_api():
        print("Error: API keys are not loaded. Please check your 'keys.txt' file.")
        sys.exit(1)

    output_path = args.output_srt or f"{os.path.splitext(args.input_srt)[0]}_refined.srt"

    refine_srt_file(args.input_srt, args.context_txt, output_path)

if __name__ == "__main__":
    main()