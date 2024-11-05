# Which Software Attributes Should be Used for Research Software Classification? An Empirical Analysis

This repository contains the implementation for the paper titled, *"Which Software Attributes Should be Used for Research Software Classification? An Empirical Analysis."* The code and methods in this study explore the effectiveness of various non-code attributes in classifying research software into community-defined categories. Using datasets from Papers with Code, this study evaluates the performance of attributes such as abstracts, README files, descriptions, and keywords through clustering and classification techniques.

## Repository Structure

- **src/**: Contains the main implementation scripts for clustering and classification tasks.
- **notebooks/**: Jupyter notebooks replicating the code and analyses from the `src` folder, allowing for step-by-step exploration.
- **plots/**: Stores the generated visuals, including bar charts and t-SNE plots used in the study.

## Requirements

This repository uses a `poetry` environment to manage dependencies. To install the required packages, use the following commands:

```bash
poetry install
```

## Usage

- Data Preparation: Prepare the data from Papers with Code, ensuring all necessary attributes (e.g., abstracts, README files) are collected.
- Generate Embeddings: Use the scripts in the src folder to create embeddings for each attribute type.
- Run Clustering and Classification: Execute scripts in the src folder to assess clustering performance and classification accuracy of various attributes.
- Visualize Results: Generate visualizations for analysis; generated visuals will be saved in the plots folder.

## Key Findings

This study highlights the importance of linking research software to publications, as attributes like abstracts provide richer context and yield better classification results compared to repository-specific metadata.

## License

This repository is licensed under the Apache 2.0 License. See the LICENSE file for more details.