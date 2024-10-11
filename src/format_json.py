import json

# Path to your JSON file
json_file_path = 'data/title_abstract_readme.json'

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Write the JSON back with indentation
with open(json_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print(f"JSON file '{json_file_path}' has been formatted with indentation.")
