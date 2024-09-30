import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# Load the JSON files
with open('data/filtered-papers-with-abstracts.json', 'r') as f:
    papers_data = json.load(f)

with open('data/pwc/links-between-papers-and-code.json', 'r') as f:
    links_data = json.load(f)


# Create a dictionary to map GitHub repos to papers using the paper URL
github_links_dict = {link['paper_url']: link['repo_url'] for link in links_data}

# Function to format multiline text with triple quotes for CSV and JSON
def format_multiline_text(text):
    if text:
        # Replace any existing triple quotes in the text to avoid conflicts
        text = text.replace('"""', "'")
    else:
        text = ''
    # Wrap the entire text in triple quotes
    return f'"""{text}"""'

# Function to scrape the README file directly from a GitHub repo page
def fetch_raw_readme(repo_url):
    try:
        # Construct the base URL for the raw files
        repo_name = repo_url.replace("https://github.com/", "")
        
        # Possible README filenames with different capitalizations
        possible_readme_files = [
            "README.md", "Readme.md", "readme.md", 
            "README.MD", "ReadMe.md", "readMe.md", 
            "README", "Readme", "readme"
        ]

        for readme_file in possible_readme_files:
            raw_readme_url = f"https://raw.githubusercontent.com/{repo_name}/main/{readme_file}"
            response = requests.get(raw_readme_url)
            
            # Check for successful response
            if response.status_code == 200:
                return response.text

            # If "main" branch doesn't exist, try the "master" branch
            raw_readme_url = f"https://raw.githubusercontent.com/{repo_name}/master/{readme_file}"
            response = requests.get(raw_readme_url)
            if response.status_code == 200:
                return response.text
        
        print(f"README not found for {repo_url}")
        return None

    except Exception as e:
        print(f"Error fetching README from {repo_url}: {e}")
        return None

# Initialize CSV and JSON filenames
csv_filename = 'data/merged_papers_methods_with_github_readmes.csv'
json_filename = 'data/merged_papers_methods_with_github_readmes.json'

# Initialize CSV storage
if not os.path.exists(csv_filename):
    # Create the CSV file if it doesn't exist
    pd.DataFrame(columns=[
        'paper_title', 'abstract', 'tasks', 'method_full_name', 'method_description',
        'main_collection_name', 'main_collection_description', 'main_collection_area', 
        'github_repo', 'readme_content'
    ]).to_csv(csv_filename, index=False, escapechar='\\')

# Initialize JSON storage (write the opening bracket if the file doesn't exist)
if not os.path.exists(json_filename):
    with open(json_filename, 'w', encoding='utf-8') as f:
        f.write('[\n')

# Flag to track if this is the first paper (to avoid adding a comma at the start)
first_paper = True

# Process each paper one by one and save it
for paper in papers_data:
    paper_url = paper['paper_url']
    
    # Extract the relevant fields
    paper_title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    github_link = github_links_dict.get(paper_url, 'No GitHub link available')

    # Try to scrape the README content from the GitHub repository
    readme_content = None
    if github_link != 'No GitHub link available':
        readme_content = fetch_raw_readme(github_link)
        if readme_content is None:
            readme_content = 'README not available'
    
    formatted_abstract = format_multiline_text(abstract)
    formatted_readme = format_multiline_text(readme_content)

    main_collection_area = None
    for method in paper['methods']:
        if 'main_collection' in method:
            if method['main_collection'] and 'area' in method['main_collection']:
                main_collection_area = method['main_collection']['area']
                break

    # Add the merged data to the list
    new_record = {
        'paper_title': paper_title,
        'abstract': abstract,
        'main_collection_area': main_collection_area,
        'github_repo': github_link,
        'github_readme_content': readme_content,
    }

    df = pd.DataFrame([new_record])
    df.to_csv(csv_filename, mode='a', header=False, index=False, escapechar='\\')
    print(f"Paper '{paper_title}' saved to CSV.")

    # Append the data to the JSON file without loading the whole JSON
    with open(json_filename, 'a', encoding='utf-8') as f:
        if not first_paper:
            # Add a comma before the next record if it's not the first one
            f.write(',\n')
        json.dump(new_record, f, indent=4)
        first_paper = False

    print(f"Paper '{paper_title}' saved to JSON.")
