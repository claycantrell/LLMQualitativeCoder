import json
import pandas as pd

def main():
    # Path to your JSON file
    json_file_path = 'outputs/amazon_output_20241216_165914.json'

    # Read the JSON data from the file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Extract 'meaning_units' from the JSON data
    meaning_units = data.get('meaning_units', [])

    if not meaning_units:
        print("No 'meaning_units' found in the JSON data.")
        return

    # Normalize 'meaning_units' into a pandas DataFrame
    df_meaning = pd.json_normalize(meaning_units, sep='_')

    # Display the DataFrame for verification
    print("Meaning Units DataFrame:")
    print(df_meaning.head(), "\n")

    # Extract all 'assigned_code_list' entries
    # We need to explode the 'assigned_code_list' to have one row per code
    df_codes = df_meaning.explode('assigned_code_list')

    # Now, normalize the 'assigned_code_list' dictionaries
    df_codes = pd.concat([df_codes.drop(['assigned_code_list'], axis=1),
                          df_codes['assigned_code_list'].apply(pd.Series)], axis=1)

    # Display the Codes DataFrame for verification
    print("Assigned Codes DataFrame:")
    print(df_codes.head(), "\n")

    # Extract unique code names
    unique_code_names = df_codes['code_name'].dropna().unique()

    # Sort the unique code names for better readability
    unique_code_names_sorted = sorted(unique_code_names)

    # Print the list of unique code names
    print("Unique Code Names:")
    for idx, code_name in enumerate(unique_code_names_sorted, start=1):
        print(f"{idx}. {code_name}")

    # Optionally, save the unique code names to a text file
    output_file = 'outputs/unique_code_names.txt'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for code_name in unique_code_names_sorted:
                f.write(f"{code_name}\n")
        print(f"\nUnique code names have been written to '{output_file}'.")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")

if __name__ == "__main__":
    main()
