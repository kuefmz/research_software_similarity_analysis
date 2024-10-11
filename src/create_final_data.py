import json


print('Load data')
with open('data/final_dataset.json', 'r') as f:
    data = json.load(f)


# Function to remove dictionaries where 'somef_descriptions' is empty and remove 'preprocessed_readme_content'
def process_data(data):
    processed_data = []
    for d in data:
        # Remove 'preprocessed_readme_content' if it exists
        if 'preprocessed_readme_content' in d:
            del d['preprocessed_readme_content']
        # Only keep the dictionary if 'somef_descriptions' is not empty
        if d.get('somef_descriptions') and d.get('github_keywords') and d.get('github_repo_title'):
                    processed_data.append(d)
    return processed_data

# Process the data
filtered_data = process_data(data)

print(len(filtered_data))
# Save the result to a new JSON file
output_file = 'data/filtered_data_complete.json'
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"Filtered data saved to {output_file}")
