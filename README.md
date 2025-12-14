# GEOFENCE-LLM v3.5

GEOFENCE-LLM is a robust prompt firewall and security analysis tool designed for Large Language Models (LLMs). It leverages internal state analysis (trajectory extraction) to detect and block adversarial inputs, focusing on feature interaction and separability to distinguish between safe and harmful prompts.

## Features

- **Trajectory Extraction**: Extracts numerical representations of LLM internal states (Llama-3.2-3B-Instruct).
- **Feature Analysis**: Performs correlation analysis and generates 2D scatter plots for feature visualization.
- **Separability Tests**: Evaluates feature effectiveness using Logistic Regression and Decision Tree classifiers.
- **Data Preparation**: Automated pipelines for cleaning and processing security datasets.
- **Environment Support**: optimized for Apple Silicon (MPS) and standard CUDA environments.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/GEOFENCE-LLM.git
    cd GEOFENCE-LLM
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation
Prepare the dataset for analysis:
```bash
python prepare_data.py
```

### 2. Feature Analysis
Run the feature interaction and separability analysis:
```bash
# Example command
python feature_analysis.py
```

## Project Structure

- `prepare_data.py`: Script for data cleaning and preparation.
- `feature_analysis.py`: Script for feature correlation and model testing.
- `data/`: Directory for datasets.
- `models/`: Directory for model checkpoints or configs.

## Roadmap

- [x] Data Pipeline Setup
- [x] Initial Trajectory Extraction
- [x] Feature Correlation Analysis
- [ ] Advanced Adversarial Detection
- [ ] Real-time Firewall Integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
