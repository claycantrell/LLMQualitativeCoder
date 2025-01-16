import json
import csv

def json_to_csv(input_json_path, output_csv_path):
    """
    Converts JSON data to CSV format based on NVivo integration rules.

    Parameters:
    - input_json_path: Path to the input JSON file.
    - output_csv_path: Path to the output CSV file.
    """
    # Define the CSV headers as per NVivo's requirements
    csv_headers = [
        "Source Name",
        "Meaning Unit ID",
        "Meaning Unit String",
        "Code Name",
        "Code Justification",
        "Speaker",
        "Time Spoken"
    ]

    try:
        # Load JSON data from the input file
        with open(input_json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Open the CSV file for writing
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_headers, delimiter='\t')
            writer.writeheader()

            # Iterate through each meaning unit in the JSON data
            for mu in data.get('meaning_units', []):
                # Extract Source Name (using source_id as a placeholder)
                source_name = mu.get('source_id', 'Unknown Source')

                # Extract Meaning Unit ID
                meaning_unit_id = mu.get('meaning_unit_id', 'N/A')

                # Extract Meaning Unit String
                meaning_unit_string = mu.get('meaning_unit_string', '').replace('\n', ' ').strip()

                # Extract Speaker and Time Spoken from preliminary_segment
                preliminary_segment = mu.get('preliminary_segment', {})
                metadata = preliminary_segment.get('metadata', {})
                speaker = metadata.get('speaker_name', 'Unknown Speaker')
                time_spoken = metadata.get('length_of_time_spoken_seconds', '0')

                # Handle multiple codes by iterating through assigned_code_list
                assigned_codes = mu.get('assigned_code_list', [])
                if not assigned_codes:
                    # If no codes are assigned, write a row with empty code fields
                    writer.writerow({
                        "Source Name": source_name,
                        "Meaning Unit ID": meaning_unit_id,
                        "Meaning Unit String": meaning_unit_string,
                        "Code Name": "",
                        "Code Justification": "",
                        "Speaker": speaker,
                        "Time Spoken": time_spoken
                    })
                else:
                    for code in assigned_codes:
                        code_name = code.get('code_name', '')
                        code_justification = code.get('code_justification', '').replace('\n', ' ').strip()

                        # Write the row to CSV
                        writer.writerow({
                            "Source Name": source_name,
                            "Meaning Unit ID": meaning_unit_id,
                            "Meaning Unit String": meaning_unit_string,
                            "Code Name": code_name,
                            "Code Justification": code_justification,
                            "Speaker": speaker,
                            "Time Spoken": time_spoken
                        })

        print(f"Successfully converted '{input_json_path}' to '{output_csv_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_json_path}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_json_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Specify the input JSON file and output CSV file paths
    input_json = "outputs/teacher_transcript_output_20250116_122542.json"    # Replace with your actual JSON file path
    output_csv = "outputs/output.csv"    # Replace with your desired CSV file path

    json_to_csv(input_json, output_csv)
