import os

def print_directory_structure(path, indent='', last=True):
    """
    Print directory structure in a tree-like format
    """

    if not os.path.exists(path):
        print("Error: Path not found.")
        return

    if os.path.isfile(path):
        print(indent + '├─ ' + os.path.basename(path))
        return

    files = sorted(os.listdir(path))

    for i, entry in enumerate(files):
        full_path = os.path.join(path, entry)
        is_last = i == len(files) - 1
        if os.path.isdir(full_path):
            if is_last:
                print(indent + '└─ ' + entry)
            else:
                print(indent + '├─ ' + entry)
            print_directory_structure(full_path, indent + '   ', is_last)
        else:
            if is_last:
                print(indent + '└─ ' + entry)
            else:
                print(indent + '├─ ' + entry)

if __name__ == "__main__":
    directory_path = "C:/Users/hemanthk.LAP53-FJS.000/OneDrive/Desktop/hemanth/Hemanth/Deep_learning/"
    print("Directory Structure:")
    print_directory_structure(directory_path)