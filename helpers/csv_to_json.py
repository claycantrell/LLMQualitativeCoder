import pandas as pd
import argparse
import os
import sys

def csv_to_json(csv_file_path, json_file_path, orient='records', lines=False, indent=4):
    """
    Converts a CSV file to a JSON file using Pandas.

    Parameters:
    - csv_file_path: Path to the input CSV file.
    - json_file_path: Path where the output JSON file will be saved.
    - orient: Type of JSON format. Common options include 'records', 'split', 'index', 'columns', and 'values'.
    - lines: If True, writes JSON objects separated by newlines (useful for large files).
    - indent: Specifies the indentation level for pretty-printing JSON.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read CSV file: {csv_file_path}")

        # Convert DataFrame to JSON
        df.to_json(json_file_path, orient=orient, lines=lines, indent=indent)
        print(f"Successfully wrote JSON file: {json_file_path}")

    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    - args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Convert CSV to JSON using Pandas.')

    parser.add_argument('csv_file', help='Path to the input CSV file.')
    parser.add_argument('json_file', help='Path to the output JSON file.')
    parser.add_argument('--orient', default='records', choices=['split', 'records', 'index', 'columns', 'values'],
                        help='A string indicating the format of the JSON output.')
    parser.add_argument('--lines', action='store_true',
                        help='If set, write the JSON output with one record per line.')
    parser.add_argument('--indent', type=int, default=4,
                        help='Indentation level for the JSON output.')

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Validate input file
    if not os.path.isfile(args.csv_file):
        print(f"Error: The file {args.csv_file} does not exist.")
        sys.exit(1)

    # Convert CSV to JSON
    csv_to_json(
        csv_file_path=args.csv_file,
        json_file_path=args.json_file,
        orient=args.orient,
        lines=args.lines,
        indent=args.indent
    )

if __name__ == '__main__':
    main()
