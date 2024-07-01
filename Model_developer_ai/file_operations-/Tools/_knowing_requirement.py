import pkg_resources

def generate_requirements_file(output_filename="requirements.txt"):
    """
    Creates a requirements file listing the user's currently installed Python packages.

    Args:
        output_filename (str, optional): The name of the file to generate. Defaults to "requirements.txt".

    Returns:
        None
    """

    installed_packages = sorted(["%s==%s" % (i.key, i.version) for i in pkg_resources.working_set])

    with open(output_filename, "w", encoding="utf-8") as file:
        # Write all dependencies in a requirements-compliant format
        for package in installed_packages:
            file.write(package + "\n")

    print(f"Generated requirements file: {output_filename}")

if __name__ == "__main__":
    generate_requirements_file()