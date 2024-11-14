import re
import json
import sys
import os

def is_webvtt(file_path):
    """
    Checks if the given file is a WebVTT file by verifying the header.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            return first_line == "WEBVTT"
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def parse_timecode(timecode):
    """
    Parses a timecode string formatted as 'HH:MM:SS.MMM' and returns the time in seconds.
    """
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", timecode)
    if not match:
        raise ValueError(f"Invalid timecode format: '{timecode}'")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds

def extract_speaker_and_text(text):
    """
    Extracts the speaker name and the spoken text from a line.
    Assumes the format 'Speaker Name: Text'.
    """
    if ':' in text:
        speaker, dialogue = text.split(':', 1)
        return speaker.strip(), dialogue.strip()
    else:
        return None, text.strip()

def parse_webvtt(file_path):
    """
    Parses a WebVTT file and returns a list of JSON objects with cue information.
    """
    cues = []
    cue_id = 1
    timecode_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s-->\s(\d{2}:\d{2}:\d{2}\.\d{3})")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            
            # Skip empty lines
            if not line:
                index += 1
                continue
            
            # Skip the WEBVTT header
            if line == "WEBVTT":
                index += 1
                continue
            
            # Check if the line is a cue number
            if line.isdigit():
                index += 1
                if index >= len(lines):
                    break
                timecode_line = lines[index].strip()
            else:
                timecode_line = line
            
            # Match the timecode
            time_match = timecode_pattern.match(timecode_line)
            if time_match:
                start_time = parse_timecode(time_match.group(1))
                end_time = parse_timecode(time_match.group(2))
                duration = end_time - start_time
                
                index += 1
                if index >= len(lines):
                    text = ""
                else:
                    text = lines[index].strip()
                
                speaker, dialogue = extract_speaker_and_text(text)
                
                cue = {
                    "id": cue_id,
                    "length_of_time_spoken_seconds": round(duration, 3),
                    "text_context": dialogue,
                    "speaker_name": speaker if speaker else "Unknown"
                }
                
                cues.append(cue)
                cue_id += 1
            else:
                # If the line is not a cue number or timecode, skip it
                index += 1
        
        return cues
    except Exception as e:
        print(f"Error parsing WebVTT file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_webvtt.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    # Construct the full path to the file within the vtt_transcripts directory
    file_path = os.path.join('vtt_transcripts', filename)
    
    if not is_webvtt(file_path):
        print("The provided file is not a valid WebVTT file.")
        sys.exit(1)
    
    cues = parse_webvtt(file_path)
    
    # Output the JSON objects
    json_output = json.dumps(cues, indent=4)
    print(json_output)
    
    # Optionally, write the JSON to a file
    json_folder = 'json_transcripts'
    os.makedirs(json_folder, exist_ok=True)  # Ensure the folder exists
    output_file = os.path.join(json_folder, 'output_cues.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(json_output)
        print(f"\nJSON output written to '{output_file}'.")
    except Exception as e:
        print(f"Error writing JSON to file: {e}")

if __name__ == "__main__":
    main()
