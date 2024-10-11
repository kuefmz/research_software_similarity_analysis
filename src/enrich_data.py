import json
import somef
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re, unicodedata
import contractions
from nltk import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer


# Preprocessor class (from the script you provided)
class Preprocessor:
    def __init__(self, data: pd.DataFrame, TEXT: str = 'Text') -> None:
        self.data = data
        self.TEXT = TEXT

    def denoise_text(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = contractions.fix(text)
        text.replace("""404: Not Found""", '')
        return text

    def remove_stop_words(self, text: str) -> list:
        stop_words = stopwords.words('english')
        stop_words += ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'and']
        stop_words += ['network', 'install', 'run', 'file', 'use', 'result', 'paper', 'python', 'using', 'code', 'model', 'train', 'implementation', 'use']
        stop_words += ['data', 'dataset', 'example', 'build', 'learn', 'download', 'obj']
        return [word for word in text if word not in stop_words]

    def remove_codeblocks(self, text: str) -> str:
        return re.sub('```.*?```', ' ', text)

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]|\_', ' ', text)

    def remove_non_ascii(self, words: list) -> list:
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def remove_links2(self, text: str) -> str:
        return ' '.join([token for token in text.split(' ') if 'http' not in token])

    def lemmatize_verbs(self, words: list) -> list:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word, pos='v') for word in words]

    def lemmatize_nouns(self, words: list) -> list:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word, pos='n') for word in words]

    def lemmatize_adjectives(self, words: list) -> list:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word, pos='a') for word in words]

    def remove_short_and_number_words(self, text: list) -> list:
        return [word for word in text if word.isdigit() == False and len(word) > 2]

    def preprocess(self, text: str) -> str:
        text = self.remove_codeblocks(text)
        text = self.remove_links2(text)
        text = self.denoise_text(text)
        text = self.remove_punctuation(text)
        text = text.lower()
        words = word_tokenize(text)
        words = self.remove_non_ascii(words)
        words = self.lemmatize_verbs(words)
        words = self.lemmatize_nouns(words)
        words = self.lemmatize_adjectives(words)
        words = self.remove_stop_words(words)
        words = self.remove_short_and_number_words(words)
        return ' '.join(words)


# Function to use the Somef library to extract the description from the GitHub repo
def get_somef_description(github_url):
    try:
        # Use Somef library to analyze the repository
        model = somef.Somef()
        model.load(github_url)
        results = model.extract_metadata(technique="GitHubAPI")
        
        # Concatenate all "description" result values
        description_parts = [desc['result']['value'] for desc in results.get('description', [])]
        concatenated_description = " ".join(description_parts)
        return concatenated_description
    except Exception as e:
        print(f"Error processing {github_url}: {e}")
        return None


# Load the input JSON file
input_file = 'data/title_abstract_readme.json'
output_file = 'data/title_abstract_readme_clean.json'

with open(input_file, 'r') as f:
    data = json.load(f)

# Process each item and add the Somef description and preprocessed README if available
with open(output_file, 'w') as out_f:
    for item in data:
        github_repo = item.get('github_repo')
        readme_content = item.get('github_readme_content')

        # Get Somef description
        if github_repo and github_repo != "No GitHub link available":
            print(f"Processing GitHub repo: {github_repo}")
            somef_description = get_somef_description(github_repo)
            if somef_description:
                item['somef_description'] = somef_description
        else:
            item['somef_description'] = "No GitHub repo available"

        # Preprocess the README content
        if readme_content:
            preprocessor = Preprocessor(pd.DataFrame([{ 'Text': readme_content }]))
            preprocessed_readme = preprocessor.preprocess(readme_content)
            item['preprocessed_readme_content'] = preprocessed_readme
        else:
            item['preprocessed_readme_content'] = "No README content available"

        # Write each item one by one to the output file
        out_f.write(json.dumps(item, indent=4) + '\n')

print("Processing complete. Output written to", output_file)

