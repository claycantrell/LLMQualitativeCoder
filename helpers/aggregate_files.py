import os
from pathlib import Path
import argparse

def is_binary(file_path):
    """
    Checks if a file is binary.
    
    Args:
        file_path (Path): Path object representing the file.
        
    Returns:
        bool: True if the file is binary, False otherwise.
    """
    try:
        with file_path.open('rb') as file:
            chunk = file.read(1024)
            if b'\0' in chunk:
                return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return True
    return False

def aggregate_files(output_filename='aggregated_files.txt', include_subdirectories=False, directory='.'):
    """
    Aggregates all files in the specified directory into a single text file.
    
    Args:
        output_filename (str): Name of the output text file.
        include_subdirectories (bool): Whether to include files from subdirectories.
        directory (str): Directory to aggregate files from.
    """
    current_dir = Path(directory).resolve()
    output_file = current_dir / output_filename
    
    # Open the output file in write mode
    with output_file.open('w', encoding='utf-8') as outfile:
        # Determine the search pattern based on whether to include subdirectories
        if include_subdirectories:
            files = current_dir.rglob('*')
        else:
            files = current_dir.glob('*')
        
        for file_path in files:
            # Skip directories and the output file itself
            if file_path.is_dir() or file_path == output_file:
                continue
            
            # Check if the file is binary
            if is_binary(file_path):
                print(f"Skipping binary file: {file_path.relative_to(current_dir)}")
                continue
            
            try:
                with file_path.open('r', encoding='utf-8') as infile:
                    content = infile.read()
                
                # Write a header with the file name
                outfile.write(f'\n\n===== {file_path.relative_to(current_dir)} =====\n\n')
                outfile.write(content)
                
                print(f'Added: {file_path.relative_to(current_dir)}')
            
            except UnicodeDecodeError:
                print(f"Skipping non-text file due to encoding issues: {file_path.relative_to(current_dir)}")
            except Exception as e:
                print(f'Error reading {file_path.relative_to(current_dir)}: {e}')
    
    print(f'\nAll eligible files have been aggregated into "{output_filename}".')

def main():
    parser = argparse.ArgumentParser(description='Aggregate all text-based files in a directory into a single .txt file.')
    parser.add_argument('--output', type=str, default='aggregated_files.txt', help='Name of the output text file.')
    parser.add_argument('--recursive', action='store_true', help='Include files from subdirectories.')
    parser.add_argument('--directory', type=str, default='.', help='Directory to aggregate files from.')
    
    args = parser.parse_args()
    
    aggregate_files(output_filename=args.output, include_subdirectories=args.recursive, directory=args.directory)

if __name__ == '__main__':
    main()
