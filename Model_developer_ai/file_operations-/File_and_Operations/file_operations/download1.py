# Import the required modules
import requests
import re
import os


# Define the query and the maximum number of files to download
query = "LLMs large language Surveys "
max_files = 100

# Use Bing search to get the web search results for the query
search_url = f"https://www.bing.com/search?q={query}"
response = requests.get(search_url)
html = response.text

# Use regular expressions to extract the PDF file URLs from the HTML
pdf_pattern = r"https?://\S+\.pdf"
pdf_urls = re.findall(pdf_pattern, html)

# Create a folder to store the downloaded files
folder_name = query.replace(" ", "_")
os.mkdir(folder_name)

# Loop through the PDF URLs and download them one by one
for i, pdf_url in enumerate(pdf_urls):
    # Break the loop if the maximum number of files is reached
    if i >= max_files:
        break
    # Get the file name from the URL
    file_name = pdf_url.split("/")[-1]
    # Download the file and save it in the folder
    file_path = os.path.join(folder_name, file_name)
    # Set the timeout to 10 seconds for both connection and reading
    file_response = requests.get(pdf_url, timeout=30)
    # file_response = requests.get(pdf_url)
    with open(file_path, "wb") as f:
        f.write(file_response.content)
    # Print a message to indicate the progress
    print(f"Downloaded {i+1} of {max_files} files: {file_name}")
