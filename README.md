# Project Analyzer

This project provides an automated way to scan and summarize the structure and contents of any given codebase or folder. It leverages IBM Watsonx to generate textual summaries of both individual files and the overall project, resulting in a developer guide that can help new contributors quickly understand how the project is organized.

---

## Overview

- **Scans a folder** to gather information on all files and subdirectories.
- **Excludes** certain folders by default (e.g., `.git`, `node_modules`, `__pycache__`, etc.).
- **Generates summaries** of each file and directory by making requests to the IBM Watsonx text generation service.
- **Produces** a `findings.json` file and text logs of initial summaries.
- **Automatically creates** a final developer guide in Markdown format, combining all the summaries.

---

## Requirements

1. **Python 3** (3.7+ recommended).
2. **pip** (for installing required Python packages).

---

# Setting Up Virtual Environment

## 1. Create a Virtual Environment
```bash
python3 -m venv venv #For Mac

python -m venv venv #For Windows
```

## 2. Activate the Virtual Environment
```bash
source .venv/bin/activate #For Mac

.venv/Scripts/Acticvate #For Windows
```

## Installation

1. **Clone or download** this repository:

    ```bash
    git clone https://github.com/Hahmedsyed/ibmhackathon.git
    ```

2. **Move into the Project Directory:

     ```bash
    cd ibmhackathon
    ```

3. **Install dependencies** by running:

    ```bash
    pip install -r requirements.txt
    ```

    This includes packages such as `requests`, `python-dotenv`, `gradio`, etc., which the script depends on.

---

## .env File Setup

Create a file named `.env` in the same directory as `project_analyzer.py` with the following contents (or equivalent values for your environment):

```env
IBM_API_KEY=Your_IBM_Cloud_API_key
REGION_ENDPOINT=https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29
PROJECT_ID=Your_Watsonx_project_ID
MODEL_ID=ibm/granite-3-8b-instruct
```

## How to Run

After setting up the environment variables in your `.env` file and installing the dependencies, you can run the analyzer script as follows:

```bash
python project_analyzer.py -targetFolder /path/to/target/folder --chatbot  
```

## What to Expect

- **Initial-Summaries**: The script will create a file named `initial-summaries_<timestamp>.txt` in the same directory as the script, containing summaries of files and directories.
- **Findings**: A `findings.json` file will be created inside a timestamped subfolder in `findings/<timestamp>/findings.json`. This JSON contains detailed results of the analysis.
- **Developer Guide**: A markdown file `guidebook_<timestamp>.md` will be generated, combining all summaries into a single guide.
- **Locally Hosted Chatbot**: A chatbot will be hosted locally, The model is hosted on: `http://localhost:7860/`, the user can use this to ask questions about the selected Code Repository.

---

## Notes

- If any of the required environment variables (`IBM_API_KEY`, `REGION_ENDPOINT`) are missing, the script will raise an exception.
- The script will skip binary or unreadable files (e.g., images, compiled files).
- Excluded folders (like `.git`, `node_modules`, etc.) are not analyzed to keep the summaries concise.

---

## Additional Configuration

- To add or remove folders from the exclusion list, modify the `EXCLUSION_LIST` in `project_analyzer.py`.
- You can change the default text generation parameters (e.g., `max_new_tokens`) in the `generate_text` function if needed.
- If you have a different Watsonx model or region endpoint, update the `.env` file accordingly.

---

## Demo Video
[Watch the Demo Video](./project_analyzer.mp4)
