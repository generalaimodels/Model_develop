import csv
import importlib
from typing import List, Dict

def get_functions_starting_with_letter(module_name: str, letter: str) -> List[str]:
    """
    Extracts a list of function names from the specified module that start with the specified letter.
    
    :param module_name: Name of the module to inspect.
    :param letter: The letter with which the function names should start.
    :return: List of function names starting with the letter.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error importing module: {e}")
        return []
    
    # Get all attributes from the module and filter for functions starting with the specified letter
    func = [f for f in dir(module) if f.startswith(letter)]
    return func

def save_to_csv(data: Dict[str, List[str]], filename: str = "functions1.csv") -> None:
    """
    Saves a dictionary to a CSV file with 27 columns: 's_no' and one for each letter of the alphabet.
    
    :param data: Dictionary where keys are letters and values are lists of function names.
    :param filename: Name of the CSV file to save the data to.
    """
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        headers = ['s_no'] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        writer.writerow(headers)  # Writing the header

        # Find the longest list to determine the number of rows needed
        max_rows = max(len(lst) for lst in data.values())

        # Writing the data, padding with empty strings if a list is shorter than max_rows
        for i in range(max_rows):
            row = [i + 1]  # s_no starts with 1
            for letter in headers[1:]:
                func_list = data.get(letter, [])
                row.append(func_list[i] if i < len(func_list) else '')
            writer.writerow(row)

def main():
    module_name = 'sys'  # Example module
    functions_by_letter = {}

    # Fetch functions for each letter and store them in the dictionary
    for letter in [chr(i) for i in range(ord('a'), ord('z') + 1)]:
        functions_by_letter[letter] = get_functions_starting_with_letter(module_name, letter)
    
    save_to_csv(functions_by_letter)

if __name__ == "__main__":
    main()