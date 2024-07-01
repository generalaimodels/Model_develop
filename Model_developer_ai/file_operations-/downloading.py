import arxiv
import os
import urllib.request
import re
from datetime import date, timedelta
def download_papers(query, max_results, save_dir, search_in=('title', 'summary')):
    """
    This function downloads papers from arXiv based on the provided query.

    Parameters:
    - query (str): The search query for the papers.
    - max_results (int): The maximum number of results to return.
    - save_dir (str): The directory where the papers will be saved.
    - search_in (tuple): Where to search for keywords, options are 'title', 'summary', or both.
    """
    # Get the current date
    today = date.today()
# Get the date 30 days ago
    start_date = today - timedelta(days=30)
    # Construct the default API client
    client = arxiv.Client()

    # Create a search object
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        # date_range=arxiv.Search.date_range(start=start_date, end=today)
    )

    # Get the results as a list
    results = list(client.results(search))

    # Convert the query into a list of phrases and keywords
    phrases = re.findall(r'"([^"]+)"', query)
    keywords = [kw for kw in re.split(r'"[^"]+"', query) if kw.strip() != '']
    keywords = list(set(keywords + phrases))  # Combine phrases and keywords, remove duplicates

    filtered_results = []  # A list to store the filtered results

    for result in results:  # Loop through the results
        text_to_search = ' '.join(filter(None, [result.title if 'title' in search_in else '', 
                                                result.summary if 'summary' in search_in else '']))
        match_count = sum(bool(re.search(r'\b' + re.escape(keyword) + r'\b', text_to_search, re.IGNORECASE)) for keyword in keywords)

        # Adjust the required matches to be the minimum of 3 or the number of keywords
        required_matches = min(3, len(keywords))

        if match_count >= required_matches:  # If the result matches at least the required keywords
            filtered_results.append((result, match_count))  # Add it to the filtered list

    # Sort the filtered results by the number of keyword matches in descending order
    filtered_results.sort(key=lambda x: x[1], reverse=True)

    # Check if the folder exists
    if not os.path.exists(save_dir):
        # If not, create the folder
        os.makedirs(save_dir)

    # Download the filtered and sorted results
    for result, match_count in filtered_results[:max_results]:
        try:
            # Get the paper id
            paper_id = result.entry_id.split('/')[-1]
            # Get the paper url
            paper_url = result.pdf_url
            # Get the paper file name
            paper_file = os.path.join(save_dir, paper_id + '.pdf')
            # Download the paper
            urllib.request.urlretrieve(paper_url, paper_file)
            print(f"Downloaded {paper_id} with {match_count} keyword matches")
        except Exception as e:
            print(f"Failed to download {paper_id}: {e}")

# Example usage
save_dir = "C:/Users/hemanthk.LAP53-FJS.000/OneDrive/Desktop/hemanth/Hemanth\Deep_learning/Hemath_paper3/"
query = '"Google DeepMind"  "Meta ai research"'
No_of_papers = 100

download_papers(query=query, max_results=No_of_papers, save_dir=save_dir, search_in=('title', 'summary'))