import json
import somef
import os
import subprocess
from time import sleep


output_folder = "data/somef"
os.makedirs(output_folder, exist_ok=True) 

github_token = ""

# Function to get the description from the Somef library
def get_somef_description(github_url, repo_name):
    print(github_url)
    try:
        # Create the output file path for this repo
        output_file_path = os.path.join(output_folder, f"{repo_name}.json")

        # Run Somef using the command line
        somef_command = f"somef describe -r {github_url} -o {output_file_path} -t 0.9 -p"
        subprocess.run(somef_command, shell=True, check=True)
        
        # Load the result from the saved output file
        with open(output_file_path, "r") as f:
            somef_data = json.load(f)
        
        # Get all the descriptions and concatenate them
        descriptions = []
        if "description" in somef_data:
            for desc in somef_data["description"]:
                descriptions.append(desc["result"]["value"])

        # Return concatenated descriptions
        return " ".join(descriptions)
    
    except Exception as e:
        print(f"Error processing {github_url}: {e}")
        return ""

# Load the JSON containing the list of repositories
with open("data/title_abstract_readme_clean.json", "r") as f:
    repo_list = json.load(f)

# Open the output JSON file in append mode to write item by item
with open("data/title_abstract_readme_somef_clean.json", "a") as output_file:
    output_file.write("[\n") 
    for index, item in enumerate(repo_list):
        # Get the GitHub repo URL
        github_url = item.get("github_repo")
        title = item.get("paper_title").lower().replace(' ', '_')
        
        if github_url:
            # Get the concatenated description from Somef
            somef_descriptions = get_somef_description(github_url, title)
            # Add the "somef_descriptions" field to the item
            item["somef_descriptions"] = somef_descriptions
        
        # Write the item to the output JSON file
        json.dump(item, output_file, indent=4)
        
        
        # Add a comma after every item except the last one
        if index < len(repo_list) - 1:
            output_file.write(",\n")
        else:
            output_file.write("\n") 

        sleep(1)
    
    output_file.write("]\n")

print("Processing completed.")
