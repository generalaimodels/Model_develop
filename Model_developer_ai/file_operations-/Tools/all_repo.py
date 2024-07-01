import requests
from typing import List

def get_all_repos(url: str) -> List[str]:
    """
    Fetches the list of all repositories from the given GitHub URL.

    Args:
    url (str): The URL of the GitHub user or organization.

    Returns:
    List[str]: A list of repository names.
    """
    repos = []
    page = 1
    while True:
        response = requests.get(url, params={'page': page})
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            repos.extend([repo['name'] for repo in data])
            page += 1
        else:
            break
    return repos

if __name__ == "__main__":
    github_url = "https://api.github.com/users/Stability-AI/repos"
    repo_list = get_all_repos(github_url)
    with open("repositories_Stability-AI.txt", "w") as file:
        for repo in repo_list:
            file.write(repo + "\n")