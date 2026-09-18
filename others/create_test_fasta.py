#!/usr/bin/env python3
"""
Script to create a 0.1MB test FASTA file from the SILVA database.
"""

import os
import sys
from pathlib import Path

def create_test_fasta(input_file, output_file, target_size_mb=0.1):
    """
    Create a test FASTA file of specified size from a larger FASTA file.
    
    Args:
        input_file (str): Path to the input SILVA FASTA file
        output_file (str): Path to the output test FASTA file
        target_size_mb (float): Target size in megabytes
    """
    target_size_bytes = int(target_size_mb * 1024 * 1024)  # Convert MB to bytes
    current_size = 0
    sequence_count = 0
    
    print(f"Creating {target_size_mb}MB test file from {input_file}")
    print(f"Target size: {target_size_bytes:,} bytes")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            current_sequence = []
            header = None
            
            for line in infile:
                if current_size >= target_size_bytes:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('>'):
                    # Write previous sequence if exists
                    if header and current_sequence:
                        sequence_data = header + '\n' + ''.join(current_sequence) + '\n'
                        outfile.write(sequence_data)
                        current_size += len(sequence_data.encode('utf-8'))
                        sequence_count += 1
                        
                        if sequence_count % 100 == 0:
                            print(f"Processed {sequence_count:,} sequences, "
                                  f"current size: {current_size:,} bytes "
                                  f"({current_size/1024/1024:.2f} MB)")
                    
                    # Start new sequence
                    header = line
                    current_sequence = []
                else:
                    # Add sequence line
                    current_sequence.append(line)
            
            # Write the last sequence if we haven't reached target size
            if header and current_sequence and current_size < target_size_bytes:
                sequence_data = header + '\n' + ''.join(current_sequence) + '\n'
                outfile.write(sequence_data)
                current_size += len(sequence_data.encode('utf-8'))
                sequence_count += 1
    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found!")
        return False
    except Exception as e:
        print(f"Error processing file: {e}")
        return False
    
    print(f"\nTest file created successfully!")
    print(f"Output file: {output_file}")
    print(f"Final size: {current_size:,} bytes ({current_size/1024/1024:.2f} MB)")
    print(f"Number of sequences: {sequence_count:,}")
    
    return True

def main():
    # Define file paths, relative to the project root (this script lives in others/)
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "raw" / "SILVA_138.1_SSURef_NR99_tax_silva.fasta"
    output_file = project_root / "data" / "processed" / "silva_test_0.1mb.fasta"

    # Create processed directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        sys.exit(1)
    
    # Create the test file
    success = create_test_fasta(input_file, output_file, target_size_mb=0.1)
    
    if success:
        print(f"\nSuccessfully created 0.1MB test FASTA file!")
        print(f"Location: {output_file}")
    else:
        print("Failed to create test file!")
        sys.exit(1)

if __name__ == "__main__":
    main()
