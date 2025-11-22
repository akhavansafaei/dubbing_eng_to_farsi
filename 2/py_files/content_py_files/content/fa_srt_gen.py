#!/usr/bin/env python3
"""
SRT to Persian Translator with Sequential Context
Usage: python srt_translator.py input.srt [output.srt] --context <N> --history <K>
"""

import os
import sys
import time
import google.generativeai as genai
from tqdm import tqdm
import re
import argparse

# --- Configuration ---
# Load API keys from 'keys.txt'.
try:
    with open("keys.txt", "r") as f:
        api_keys = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: `keys.txt` not found. Please create it and add your Google API keys.")
    api_keys = []

current_key_index = 0
LLM = None
SUMMARY = ""

def switch_api_key():
    """Cycles to the next API key in the list when a quota is hit."""
    global current_key_index, LLM
    current_key_index = (current_key_index + 1) % len(api_keys)
    new_key = api_keys[current_key_index]
    os.environ["GOOGLE_API_KEY"] = new_key
    genai.configure(api_key=new_key)
    LLM = genai.GenerativeModel('gemini-1.5-flash')
    print(f"\nSwitched to API key ending in: ...{new_key[-4:]}")

def initialize_api():
    """Initializes the Gemini API with the first key."""
    global LLM
    if not api_keys:
        return
    os.environ["GOOGLE_API_KEY"] = api_keys[current_key_index]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    LLM = genai.GenerativeModel('gemini-1.5-flash')

def create_summary_prompt(full_text):
    """Creates a prompt to generate a summary of the entire transcript."""
    return f"""
    Please provide a concise, comprehensive summary of the following English transcript.
    This summary will be used to provide overall context for a translation task.

    Transcript:
    {full_text}
    """

def create_translation_prompt(current_text, previous_english, next_english, previously_translated_persian, summary):
    """Creates a contextual prompt for translating a single subtitle line."""
    
    # Context from surrounding English lines
    surrounding_context = []
    if previous_english:
        surrounding_context.append(f"Previous English Lines:\n{previous_english}")
    if next_english:
        surrounding_context.append(f"Next English Lines:\n{next_english}")
    surrounding_context_section = "\n\n".join(surrounding_context) if surrounding_context else "No surrounding lines."

    # Context from previously translated Persian lines
    translated_context_section = previously_translated_persian if previously_translated_persian else "This is the first line to be translated."

    return f"""
    You are an expert translator for English to Persian (Farsi) subtitles. Your task is to translate the "CURRENT TEXT TO TRANSLATE" while maintaining consistency with the provided context.

    ---
    CONTEXT 1: OVERALL SUMMARY
    {summary}
    ---
    CONTEXT 2: SURROUNDING ENGLISH LINES
    {surrounding_context_section}
    ---
    CONTEXT 3: PREVIOUSLY TRANSLATED PERSIAN LINES
    {translated_context_section}
    ---

    CURRENT TEXT TO TRANSLATE:
    {current_text}

    Instructions:
    1.  Translate the "CURRENT TEXT TO TRANSLATE" into natural, fluent Persian.
    2.  Ensure your translation is consistent with the tone and terminology of the previously translated lines.
    3.  Keep the translation concise and easy to read for subtitles.
    4.  Return ONLY the Persian translation. Do not include any other text, labels, or explanations.
    """

def get_llm_response(prompt, max_retries=5):
    """Gets a response from the LLM, with retry logic and API key switching."""
    for attempt in range(max_retries):
        try:
            response = LLM.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_message = str(e).lower()
            print(f"\nError on attempt {attempt + 1}: {error_message}")
            if any(keyword in error_message for keyword in ["quota", "limit", "exceeded", "resource has been exhausted"]):
                print("Quota likely exceeded, switching API key...")
                switch_api_key()
                time.sleep(2)
            else:
                time.sleep(attempt + 1)

            if attempt == max_retries - 1:
                print(f"Failed to get a response after {max_retries} attempts.")
                return "[Translation Failed]"
    return "[Translation Failed]"

def parse_srt_file(file_path):
    """Parses an SRT file into a list of subtitle segments."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3 and '-->' in lines[1]:
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

def generate_summary(segments):
    """Generates a summary of the transcript for global context."""
    global SUMMARY
    print("Generating transcript summary...")
    full_text = ' '.join([segment['text'] for segment in segments])
    summary_prompt = create_summary_prompt(full_text[:10000]) # Limit for summary
    SUMMARY = get_llm_response(summary_prompt)
    if not SUMMARY or "failed" in SUMMARY.lower():
        print("Warning: Could not generate a summary. Proceeding without it.")
        SUMMARY = "No summary available."
    else:
        print(f"Summary generated: {SUMMARY[:120]}...")
    return SUMMARY

def translate_segments_sequentially(segments, context_size, history_size):
    """Translates segments sequentially, building upon previous translations."""
    if not segments:
        return []

    translated_segments = []
    translated_history = []
    
    print(f"Translating {len(segments)} segments sequentially...")
    
    # Start translation from the first segment (index 0)
    for i in tqdm(range(len(segments)), desc="Translating"):
        current_segment = segments[i]

        # Gather surrounding English context
        prev_english_start = max(0, i - context_size)
        previous_english_segments = segments[prev_english_start:i]
        previous_english_text = "\n".join([s['text'] for s in previous_english_segments])

        next_english_start = i + 1
        next_english_end = next_english_start + context_size
        next_english_segments = segments[next_english_start:next_english_end]
        next_english_text = "\n".join([s['text'] for s in next_english_segments])

        # Get previously translated Persian text based on history_size
        previously_translated_persian = "\n".join(translated_history[-history_size:])

        # Create the prompt and get the translation
        translation_prompt = create_translation_prompt(
            current_segment['text'],
            previous_english_text,
            next_english_text,
            previously_translated_persian,
            SUMMARY
        )
        persian_text = get_llm_response(translation_prompt)

        # Append the new translation to history and results
        translated_history.append(persian_text)
        translated_segments.append({**current_segment, 'text': persian_text})
        
        time.sleep(0.3)  # Respect API rate limits

    return translated_segments

def write_srt_file(segments, output_path):
    """Writes the translated segments to a new SRT file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(f"{segment['sequence']}\n")
            file.write(f"{segment['start']} --> {segment['end']}\n")
            file.write(f"{segment['text']}\n\n")

def main():
    """Main function to orchestrate the translation process."""
    parser = argparse.ArgumentParser(description='Translate SRT subtitles to Persian with sequential context.')
    parser.add_argument('input_srt', help='Path to the input SRT file.')
    parser.add_argument('output_srt', nargs='?', help='Path for the output SRT file (optional).')
    parser.add_argument(
        '--context', '-c', type=int, default=3,
        help='Number of previous/next English lines for context (default: 3).'
    )
    parser.add_argument(
        '--history', '-k', type=int, default=2,
        help='Number of previously translated Persian lines for history context (default: 2).'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_srt):
        print(f"Error: Input file '{args.input_srt}' not found.")
        sys.exit(1)

    if not api_keys:
        print("Error: API keys are not loaded. Please check your 'keys.txt' file.")
        sys.exit(1)

    output_path = args.output_srt or f"{os.path.splitext(args.input_srt)[0]}_persian.srt"

    print(f"Input file: {args.input_srt}")
    print(f"Output file: {output_path}")
    print(f"English context size: {args.context} lines")
    print(f"Translation history size: {args.history} lines")

    try:
        initialize_api()
        segments = parse_srt_file(args.input_srt)
        if not segments:
            print("No valid subtitle segments found.")
            sys.exit(1)
        print(f"Found {len(segments)} subtitle segments.")

        generate_summary(segments)
        translated_segments = translate_segments_sequentially(segments, args.context, args.history)
        write_srt_file(translated_segments, output_path)

        print("\nTranslation completed successfully!")
        print(f"Persian SRT file saved as: {output_path}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
