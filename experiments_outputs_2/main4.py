import os
import json
import csv
import re

def process_files_to_csv():
    """
    Scans the current directory for specific JSON files, extracts data,
    and writes it to a CSV file.
    """
    # Regex to match filenames like 'cifar10_resnet50_adam_0.5.json'
    # and capture the numeric part.
    filename_pattern = re.compile(r"cifar100_vgg19_adam_([\d\.]+)\.json")
    
    # List to hold the extracted data
    extracted_data = []

    # Get the current directory path
    current_directory = os.getcwd()
    print(f"Scanning for files in: {current_directory}\n")

    # Iterate over all files in the current directory
    for filename in os.listdir(current_directory):
        match = filename_pattern.match(filename)
        
        # Check if the filename matches the expected pattern
        if match:
            # The saliency number is the first captured group
            saliency = match.group(1)
            file_path = os.path.join(current_directory, filename)
            
            try:
                with open(file_path, 'r') as f:
                    # Load the JSON content from the file
                    data_array = json.load(f)
                    
                    # Ensure it's a list and not empty
                    if isinstance(data_array, list) and data_array:
                        # Get the last number from the array
                        remaining = data_array[-1]
                        
                        # Append the results as a dictionary
                        extracted_data.append({
                            "Saliency": float(saliency),
                            "Remaining": remaining
                        })
                        print(f"Processed '{filename}': Saliency={saliency}, Remaining={remaining}")
                    else:
                        print(f"Warning: File '{filename}' does not contain a valid, non-empty array.")

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{filename}'.")
            except Exception as e:
                print(f"An unexpected error occurred while processing '{filename}': {e}")

    if not extracted_data:
        print("\nNo matching files were found or processed. CSV file will not be created.")
        return

    # Sort the data by the 'Saliency' value
    extracted_data.sort(key=lambda x: x['Saliency'])

    # Define the output CSV file name
    output_filename = "saliency_results.csv"
    
    try:
        # Write the extracted data to the CSV file
        with open(output_filename, 'w', newline='') as csvfile:
            # Define the column headers
            fieldnames = ["Saliency", "Remaining"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header row
            writer.writeheader()
            # Write the data rows
            writer.writerows(extracted_data)
        
        print(f"\nSuccessfully created '{output_filename}' with {len(extracted_data)} rows.")

    except Exception as e:
        print(f"\nAn error occurred while writing the CSV file: {e}")

if __name__ == "__main__":
    process_files_to_csv()
