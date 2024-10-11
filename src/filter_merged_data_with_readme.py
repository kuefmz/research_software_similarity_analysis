import json

def filter_json(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Filter items that have a non-empty 'github_readme_content'
    filtered_data = [item for item in data if item.get('github_readme_content')]
    
    # Write filtered items one by one to avoid data loss
    with open(output_file, 'w') as f:
        for item in filtered_data:
            json.dump(item, f)
            f.write('\n')

    print(f"Filtered data saved to {output_file}")

# Example usage
input_file = 'data/merged_papers_methods_with_github_readmes.json'
output_file = 'data/title_abstract_readme.json'
filter_json(input_file, output_file)
