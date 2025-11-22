#!/usr/bin/env python3
"""
Text File Translator (English to Persian)
Usage: python text_translator.py /path/to/your/file.txt
"""

import os
import sys
import time
import google.generativeai as genai
from tqdm import tqdm
import argparse
import re

# --- Configuration ---
# This script loads API keys from 'keys.txt' in the same directory.
try:
    with open("keys.txt", "r") as f:
        api_keys = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("Error: `keys.txt` not found. Please create it and add your Google API keys.")
    api_keys = []

current_key_index = 0
LLM = None

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
        return False
    os.environ["GOOGLE_API_KEY"] = api_keys[current_key_index]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    LLM = genai.GenerativeModel('gemini-1.5-flash')
    return True

def create_translation_prompt(text_chunk):
    """Creates a prompt to translate a chunk of text."""
    return f"""
    You are an expert English to Persian (Farsi) translator.
    Translate the following English text into natural, fluent Persian.
    Maintain the original tone and meaning.

    English Text:
    ---
    {text_chunk}
    ---

    Persian Translation:
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
                return f"[Translation Failed for this chunk]"
    return "[Translation Failed for this chunk]"

def translate_text_file(input_file_path: str):
    """
    Reads a text file, translates its content to Persian, and saves it.

    Args:
        input_file_path (str): The full path to the input .txt file.
    """
    print(f"Starting translation for: {input_file_path}")

    # 1. Read the file content
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            full_text = file.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not full_text.strip():
        print("The file is empty. Nothing to translate.")
        return

    # 2. Split the text into chunks (paragraphs) to handle long texts
    # A chunk is one or more non-empty lines separated by one or more empty lines.
    chunks = re.split(r'\n\s*\n', full_text.strip())
    
    # 3. Translate each chunk
    translated_chunks = []
    print(f"Translating {len(chunks)} text chunks...")
    for chunk in tqdm(chunks, desc="Translating Text"):
        if not chunk.strip():
            continue
            
        prompt = create_translation_prompt(chunk)
        translated_chunk = get_llm_response(prompt)
        translated_chunks.append(translated_chunk)
        time.sleep(0.3) # To respect API rate limits

    # 4. Combine translated chunks and prepare the output file path
    final_translation = "\n\n".join(translated_chunks)
    
    base_dir = os.path.dirname(input_file_path)
    base_name = os.path.basename(input_file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(base_dir, f"{file_name_without_ext}_persian.txt")

    # 5. Write the translated content to the output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(final_translation)
        print("\nTranslation completed successfully!")
        print(f"Output saved to: {output_file_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")


def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(description='Translate a text file from English to Persian.')
    parser.add_argument('input_file', help='Path to the input .txt file.')
    args = parser.parse_args()

    if not initialize_api():
        print("Error: API keys not loaded. Please check your 'keys.txt' file.")
        sys.exit(1)

    translate_text_file(args.input_file)

if __name__ == "__main__":
    main()
