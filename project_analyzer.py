import os
import logging
import json
import requests
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Configure Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------------------------------------------
# Exclusion List
# -------------------------------------------------------------------
EXCLUSION_LIST = [
    '.git', '.venv', 'node_modules', '__pycache__', '.DS_Store',
    'pb_data', 'pb_public', 'migrations'
]

# -------------------------------------------------------------------
# 1. IBM Cloud IAM Token Retrieval
# -------------------------------------------------------------------
def get_iam_token(api_key):
    """
    Exchanges your IBM Cloud API key for a short-lived IAM access token.
    """
    url = "https://iam.cloud.ibm.com/identity/token"
    payload = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=payload, headers=headers)
    response.raise_for_status()
    return response.json()["access_token"]

# -------------------------------------------------------------------
# 2. Watsonx Generate Text Function
# -------------------------------------------------------------------
def generate_text(
    token,
    system_prompt,
    user_prompt,
    region_endpoint,
    project_id,
    model_id="ibm/granite-3-8b-instruct"
):
    """
    Sends a system + user prompt to the Watsonx endpoint and 
    returns the 'generated_text' field from the first result.
    """

    # Combine system and user roles with special tokens
    combined_input = (
        f"<|start_of_role|>system<|end_of_role|>{system_prompt}"
        "<|end_of_text|>\n"
        f"<|start_of_role|>user<|end_of_role|>{user_prompt}"
        "<|start_of_role|>assistant<|end_of_role|>"
    )

    payload = {
        "input": combined_input,
        "model_id": model_id,
        "project_id": project_id,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 512
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(region_endpoint, json=payload, headers=headers)

    if response.status_code != 200:
        logging.error("Non-200 response: %s %s", response.status_code, response.text)
        return None

    data = response.json()
    results = data.get("results", [])
    if not results:
        return None
    return results[0].get("generated_text", "").strip()

# -------------------------------------------------------------------
# 3. ProjectAnalyzer: scans a folder, calls Watsonx to summarize
# -------------------------------------------------------------------
class ProjectAnalyzer:
    def __init__(self, project_dir):
        """
        1) Load .env to get: IBM_API_KEY, REGION_ENDPOINT, PROJECT_ID, MODEL_ID
        2) Set up directory structure for analysis results
        3) Acquire IAM token
        """
        load_dotenv()

        self.ibm_api_key = os.getenv("IBM_API_KEY")
        self.region_endpoint = os.getenv("REGION_ENDPOINT")
        self.project_id = os.getenv("PROJECT_ID", "7086d01d-42be-4dcd-bf4e-50e18594f225")
        self.model_id = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")

        if not self.ibm_api_key or not self.region_endpoint:
            raise ValueError("IBM_API_KEY or REGION_ENDPOINT missing from environment / .env")

        # Target project directory is now passed in
        self.project_dir = Path(project_dir)

        self.script_dir = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.findings_dir = self.script_dir / 'findings' / self.timestamp
        self.findings_dir.mkdir(parents=True, exist_ok=True)

        self.findings_path = self.findings_dir / 'findings.json'
        self.initial_summaries_path = self.script_dir / f'initial-summaries_{self.timestamp}.txt'

        # Acquire IAM token
        self.token = get_iam_token(self.ibm_api_key)

        # Initialize the findings JSON
        self._init_findings_file()

        logging.info("Initialized ProjectAnalyzer:")
        logging.info(f"- Project directory: {self.project_dir}")
        logging.info(f"- Script directory: {self.script_dir}")
        logging.info(f"- Findings directory: {self.findings_dir}")
        logging.info(f"- Initial summaries: {self.initial_summaries_path}")
        logging.info(f"- Findings JSON: {self.findings_path}")

    def _init_findings_file(self):
        """Initialize the findings JSON file with a basic structure."""
        initial_structure = {
            'root_summary': '',
            'directories': {},
            'files': {}
        }
        self._write_findings(initial_structure)

    def _write_findings(self, data):
        """Write/update the findings JSON file."""
        with open(self.findings_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def _append_to_summaries(self, content):
        """Append text content to the initial summaries text file."""
        with open(self.initial_summaries_path, 'a', encoding='utf-8') as f:
            f.write(f"{content}\n\n")

    def _read_findings(self):
        """Load the current findings JSON."""
        with open(self.findings_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _update_findings(self, key, value):
        """
        Update a portion of the JSON data (key can be a tuple for nested).
        e.g. ('files', 'path/to/file') = "some summary"
        """
        findings = self._read_findings()
        if isinstance(key, tuple):
            current = findings
            for k in key[:-1]:
                current = current.setdefault(k, {})
            current[key[-1]] = value
        else:
            findings[key] = value
        self._write_findings(findings)

    def is_excluded(self, path):
        """Check if a file/dir is in the excluded list."""
        return any(excluded in str(path) for excluded in EXCLUSION_LIST)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def analyze_root(self):
        """
        Summarize the entire root directory's structure,
        listing subfolders & files.
        """
        logging.info("Analyzing root directory...")
        root_contents = [
            str(f.relative_to(self.project_dir)) for f in self.project_dir.rglob('*')
            if not self.is_excluded(f)
        ]
        root_contents_str = '\n'.join(root_contents)

        system_prompt = "You are an AI assistant that summarizes a project."
        user_prompt = (
            f"Project directory: {self.project_dir}\n\n"
            f"Files and directories:\n{root_contents_str}\n\n"
            "Based on these names, what is the main language used, and what is the project's purpose?"
        )

        summary = generate_text(
            token=self.token,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            region_endpoint=self.region_endpoint,
            project_id=self.project_id,
            model_id=self.model_id
        )

        if not summary:
            summary = "No response received."

        self._update_findings('root_summary', summary)
        self._append_to_summaries(f"Project Overview:\n{summary}")
        logging.info("Root analysis complete")

    def analyze_file(self, file_path):
        """
        Summarize an individual file's purpose, classes, functions, etc.
        """
        rel_path = str(Path(file_path).relative_to(self.project_dir))
        logging.info(f"Analyzing file: {rel_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, OSError):
            logging.warning(f"Skipping binary or unreadable file: {rel_path}")
            return None

        system_prompt = "You are an AI assistant that analyzes a source code file."
        user_prompt = (
            f"File path: {rel_path}\n\n"
            f"Content:\n{content}\n\n"
            "Please summarize this file's purpose, main functions/classes, and how it fits into the project."
        )

        summary = generate_text(
            token=self.token,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            region_endpoint=self.region_endpoint,
            project_id=self.project_id,
            model_id=self.model_id
        )

        if not summary:
            summary = "No response received."

        self._update_findings(('files', rel_path), summary)
        self._append_to_summaries(f"File: {rel_path}\n{summary}")
        logging.info(f"Completed analysis of file: {rel_path}")
        return summary

    def analyze_directory(self, dir_path):
        """
        Summarize each file in a directory, then the directory as a whole.
        """
        if self.is_excluded(dir_path):
            return

        rel_path = str(Path(dir_path).relative_to(self.project_dir))
        logging.info(f"Analyzing directory: {rel_path}")

        directory_path = Path(dir_path)
        if not directory_path.is_dir():
            return

        files = [
            f for f in directory_path.iterdir()
            if f.is_file() and not self.is_excluded(f)
        ]
        if not files:
            return

        file_summaries = []
        for file in files:
            file_summary = self.analyze_file(file)
            if file_summary:
                file_summaries.append(f"{file.name}: {file_summary}")

        if file_summaries:
            system_prompt = "You are an AI assistant that analyzes code directories."
            user_prompt = (
                f"Directory path: {rel_path}\n\n"
                f"File Summaries:\n{''.join(file_summaries)}\n\n"
                "What is the purpose of this directory, and how do these files work together?"
            )

            summary = generate_text(
                token=self.token,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                region_endpoint=self.region_endpoint,
                project_id=self.project_id,
                model_id=self.model_id
            )
            if not summary:
                summary = "No response received."

            self._update_findings(('directories', rel_path), summary)
            self._append_to_summaries(f"Directory: {rel_path}\n{summary}")
            logging.info(f"Completed analysis of directory: {rel_path}")

    def analyze_project(self):
        """
        Phase 1: Analyze the root, subdirectories, and files, storing results.
        """
        logging.info(f"Starting analysis of project: {self.project_dir}")
        self.analyze_root()

        for root, dirs, _files in os.walk(self.project_dir):
            dirs[:] = [
                d for d in dirs
                if not self.is_excluded(Path(root) / d)
            ]
            self.analyze_directory(root)

        logging.info("Project analysis complete")
        return self.initial_summaries_path, self.findings_path

    def generate_developer_guide(self):
        """
        Phase 2: Summarize everything into a final developer guide.
        """
        logging.info("Generating developer guide...")

        with open(self.findings_path, 'r', encoding='utf-8') as f:
            findings = json.load(f)
        with open(self.initial_summaries_path, 'r', encoding='utf-8') as f:
            initial_summaries = f.read()

        system_prompt = "You are an AI assistant that creates developer guides."
        user_prompt = f"""
Based on the collected analysis below, create a short developer guide in markdown:

Initial Summaries:
{initial_summaries}

JSON Findings:
{json.dumps(findings, indent=2)}

Guide Outline:
1. Executive Summary
2. Project Architecture
3. Setup & Installation
4. Code Organization
5. Core Concepts
6. Development Workflow
7. API Reference
8. Common Tasks
"""

        guide_text = generate_text(
            token=self.token,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            region_endpoint=self.region_endpoint,
            project_id=self.project_id,
            model_id=self.model_id
        )
        if not guide_text:
            guide_text = "No response received."

        guidebook_path = self.script_dir / f'guidebook_{self.timestamp}.md'
        with open(guidebook_path, 'w', encoding='utf-8') as f:
            f.write(guide_text)

        logging.info(f"Developer guide created at: {guidebook_path}")
        return guidebook_path

# -------------------------------------------------------------------
# Argument Parsing
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a project folder.")
    parser.add_argument(
        "-targetFolder",
        required=True,
        help="Path to the target folder to analyze."
    )
    return parser.parse_args()

# -------------------------------------------------------------------
# 4. Main Entry Point
# -------------------------------------------------------------------
def main():
    args = parse_args()
    analyzer = ProjectAnalyzer(project_dir=args.targetFolder)
    summaries_path, findings_path = analyzer.analyze_project()
    guidebook_path = analyzer.generate_developer_guide()

    logging.info("---------------------------------------------------------")
    logging.info("Analysis complete.")
    logging.info(f"Initial Summaries: {summaries_path}")
    logging.info(f"Findings (JSON):   {findings_path}")
    logging.info(f"Developer Guide:   {guidebook_path}")
    logging.info("---------------------------------------------------------")


if __name__ == "__main__":
    main()
