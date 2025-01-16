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

def aggregate_python_files(output_filename='aggregated_python_files.txt', include_subdirectories=True, package_directory='src/transcriptanalysis', output_dir='outputs'):
    """
    Aggregates all Python (.py) files within the specified package directory into a single text file.
    
    Args:
        output_filename (str): Name of the output text file.
        include_subdirectories (bool): Whether to include files from subdirectories within the package.
        package_directory (str): Path to the package directory to aggregate files from.
        output_dir (str): Directory where the aggregated output will be stored.
    """
    current_dir = Path(package_directory).resolve()
    output_directory = Path(output_dir).resolve()
    
    # Ensure the output directory exists; if not, create it
    output_directory.mkdir(parents=True, exist_ok=True)
    
    output_file = output_directory / output_filename
    
    # Open the output file in write mode
    with output_file.open('w', encoding='utf-8') as outfile:
        # Determine the search pattern based on whether to include subdirectories
        if include_subdirectories:
            files = current_dir.rglob('*.py')  # Recursively find .py files
        else:
            files = current_dir.glob('*.py')   # Find .py files in the specified directory only
        
        # Exclude __pycache__ directories
        files = [f for f in files if '__pycache__' not in f.parts]
        
        for file_path in files:
            # Skip directories and the output file itself (in case it's inside the package directory)
            if file_path.is_dir() or file_path == output_file:
                continue
            
            # Check if the file is binary (optional, since we're filtering by .py extension)
            if is_binary(file_path):
                print(f"Skipping binary file: {file_path.relative_to(current_dir)}")
                continue
            
            try:
                with file_path.open('r', encoding='utf-8') as infile:
                    content = infile.read()
                
                # Write a header with the file name
                relative_path = file_path.relative_to(current_dir)
                outfile.write(f'\n\n===== {relative_path} =====\n\n')
                outfile.write(content)
                
                print(f'Added: {relative_path}')
            
            except UnicodeDecodeError:
                print(f"Skipping non-text file due to encoding issues: {file_path.relative_to(current_dir)}")
            except Exception as e:
                print(f'Error reading {file_path.relative_to(current_dir)}: {e}')
    
    print(f'\nAll eligible Python files have been aggregated into "{output_file}".')

def main():
    parser = argparse.ArgumentParser(description='Aggregate all Python (.py) files in a package directory into a single .txt file.')
    parser.add_argument('--output', type=str, default='aggregated_python_files.txt', help='Name of the output text file.')
    parser.add_argument('--recursive', action='store_true', help='Include files from subdirectories within the package.')
    parser.add_argument('--package', type=str, default='src/transcriptanalysis', help='Package directory to aggregate Python files from.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to store the aggregated output file.')
    
    args = parser.parse_args()
    
    aggregate_python_files(
        output_filename=args.output,
        include_subdirectories=args.recursive,
        package_directory=args.package,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
