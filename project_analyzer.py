import os
import logging
import json
import requests
import argparse
from pathlib import Path
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Constants and Configuration
# -------------------------------------------------------------------
EXCLUSION_LIST = [
    ".git", ".venv", "node_modules", "__pycache__", ".DS_Store",
    "pb_data", "pb_public", "migrations"
]

# -------------------------------------------------------------------
# IBM Cloud IAM Token Retrieval
# -------------------------------------------------------------------
def get_iam_token(api_key: str) -> str:
    """
    Exchanges an IBM Cloud API key for a short-lived IAM access token.
    """
    url = "https://iam.cloud.ibm.com/identity/token"
    payload = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data["access_token"]

# -------------------------------------------------------------------
# Watsonx Generate Text Function
# -------------------------------------------------------------------
def generate_text(
    token: str,
    system_prompt: str,
    user_prompt: str,
    region_endpoint: str,
    project_id: str,
    model_id: str = "ibm/granite-3-8b-instruct"
) -> str or None:
    """
    Sends a system + user prompt to the Watsonx endpoint,
    returning the 'generated_text' from the first result.
    """
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
# ProjectAnalyzer: Scans a folder, summarizes code, and generates docs
# -------------------------------------------------------------------
class ProjectAnalyzer:
    def __init__(self, project_dir: str):
        load_dotenv()  # Load environment variables from a .env file if present

        self.ibm_api_key = os.getenv("IBM_API_KEY")
        self.region_endpoint = os.getenv("REGION_ENDPOINT")
        self.project_id = os.getenv("PROJECT_ID")
        self.model_id = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")

        if not self.ibm_api_key or not self.region_endpoint:
            raise ValueError("IBM_API_KEY or REGION_ENDPOINT missing from environment / .env")

        self.project_dir = Path(project_dir)
        self.script_dir = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.findings_dir = self.script_dir / "findings" / self.timestamp
        self.findings_dir.mkdir(parents=True, exist_ok=True)

        self.findings_path = self.findings_dir / "findings.json"
        self.initial_summaries_path = self.script_dir / f"initial-summaries_{self.timestamp}.txt"

        self.token = get_iam_token(self.ibm_api_key)
        self._init_findings_file()

        logging.info("Initialized ProjectAnalyzer:")
        logging.info(f"- Project directory: {self.project_dir}")
        logging.info(f"- Script directory: {self.script_dir}")
        logging.info(f"- Findings directory: {self.findings_dir}")
        logging.info(f"- Initial summaries: {self.initial_summaries_path}")
        logging.info(f"- Findings JSON: {self.findings_path}")

    def _init_findings_file(self):
        """Create the findings.json with an initial structure."""
        initial_structure = {
            "root_summary": "",
            "directories": {},
            "files": {}
        }
        self._write_findings(initial_structure)

    def _write_findings(self, data: dict):
        """Overwrite findings.json with the given dictionary."""
        with open(self.findings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _append_to_summaries(self, content: str):
        """Append text content to the initial summaries .txt file."""
        with open(self.initial_summaries_path, "a", encoding="utf-8") as f:
            f.write(f"{content}\n\n")

    def _read_findings(self) -> dict:
        """Load the current findings from findings.json."""
        with open(self.findings_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _update_findings(self, key, value: str):
        """
        Update a portion of the JSON data (key can be a tuple for nested).
        E.g. ('files', 'path/to/file') -> "some summary"
        """
        findings = self._read_findings()
        if isinstance(key, tuple):
            current = findings
            for part in key[:-1]:
                current = current.setdefault(part, {})
            current[key[-1]] = value
        else:
            findings[key] = value
        self._write_findings(findings)

    def is_excluded(self, path: Path) -> bool:
        """Check if a file/dir is in the excluded list."""
        return any(excluded in str(path) for excluded in EXCLUSION_LIST)

    def analyze_root(self):
        """Summarize the entire root directory's structure."""
        logging.info("Analyzing root directory...")
        all_paths = list(self.project_dir.rglob("*"))
        root_contents = [
            str(f.relative_to(self.project_dir)) for f in all_paths
            if not self.is_excluded(f)
        ]
        root_contents_str = "\n".join(root_contents)

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
        ) or "No response received."

        self._update_findings("root_summary", summary)
        self._append_to_summaries(f"Project Overview:\n{summary}")
        logging.info("Root analysis complete")

    def analyze_file(self, file_path: Path) -> str or None:
        """Summarize an individual file's content."""
        rel_path = str(file_path.relative_to(self.project_dir))
        logging.info(f"Analyzing file: {rel_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
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
        ) or "No response received."

        self._update_findings(("files", rel_path), summary)
        self._append_to_summaries(f"File: {rel_path}\n{summary}")
        logging.info(f"Completed analysis of file: {rel_path}")
        return summary

    def analyze_directory(self, dir_path: Path):
        """Summarize each file in a directory, then summarize the directory as a whole."""
        if self.is_excluded(dir_path):
            return

        rel_path = str(dir_path.relative_to(self.project_dir))
        logging.info(f"Analyzing directory: {rel_path}")

        if not dir_path.is_dir():
            return

        files = [
            f for f in dir_path.iterdir()
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
            ) or "No response received."

            self._update_findings(("directories", rel_path), summary)
            self._append_to_summaries(f"Directory: {rel_path}\n{summary}")
            logging.info(f"Completed analysis of directory: {rel_path}")

    def analyze_project(self):
        """Analyzes the project, generating summaries for files and directories."""
        logging.info(f"Starting analysis of project: {self.project_dir}")
        self.analyze_root()

        for root, dirs, _ in os.walk(self.project_dir):
            dirs[:] = [
                d for d in dirs
                if not self.is_excluded(Path(root) / d)
            ]
            self.analyze_directory(Path(root))

        logging.info("Project analysis complete")
        return self.initial_summaries_path, self.findings_path

    def generate_developer_guide(self):
        """Generates a developer guide in markdown based on the findings."""
        logging.info("Generating developer guide...")

        with open(self.findings_path, "r", encoding="utf-8") as f:
            findings = json.load(f)
        with open(self.initial_summaries_path, "r", encoding="utf-8") as f:
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
        ) or "No response received."

        guidebook_path = self.script_dir / f"guidebook_{self.timestamp}.md"
        with open(guidebook_path, "w", encoding="utf-8") as f:
            f.write(guide_text)

        logging.info(f"Developer guide created at: {guidebook_path}")
        return guidebook_path

# -------------------------------------------------------------------
# Chatbot Logic
# -------------------------------------------------------------------
def load_context(analyzer: ProjectAnalyzer) -> str:
    """
    Returns the text from initial_summaries_path, or a fallback message
    if the file doesn't exist yet.
    """
    try:
        with open(analyzer.initial_summaries_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No summaries found. Please run analyze_project() first."

def chatbot_predict(
    message: str,
    chat_history: list,
    analyzer: ProjectAnalyzer,
    context_summary: str
) -> tuple[str, list]:
    """
    Builds a conversation from chat_history + current user message,
    calls generate_text() with 'context_summary' as part of the system prompt,
    and appends the result to chat_history.
    """
    conversation_text = ""
    for turn in chat_history:
        user_msg, bot_msg = turn
        conversation_text += f"\nUser: {user_msg}\nAssistant: {bot_msg}"

    conversation_text += f"\nUser: {message}\nAssistant:"

    system_prompt = (
        "You are an AI assistant that knows the following code summary:\n"
        f"{context_summary}\n\n"
        "Answer questions about this code. If not sure, provide your best guess.\n"
        "Current conversation:\n"
    )

    answer = generate_text(
        token=analyzer.token,
        system_prompt=system_prompt,
        user_prompt=conversation_text,
        region_endpoint=analyzer.region_endpoint,
        project_id=analyzer.project_id,
        model_id=analyzer.model_id
    ) or "No response received."

    chat_history.append((message, answer))
    return answer, chat_history

def start_chatbot(analyzer: ProjectAnalyzer):
    """
    Launches a Gradio Blocks UI for chatting about the codebase, using
    the text from 'initial_summaries_path' as context.
    """
    context_summary = load_context(analyzer)

    custom_css = """
    footer {display: none !important;}
    #share-btn {display: none !important;}
    """

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("## IntelliDoc: AI Code documentation")
        gr.Markdown(
            "Ask questions about your code base. The assistant uses the previously generated code summaries for context."
        )

        chatbot = gr.Chatbot(label="Code Chatbot")
        msg = gr.Textbox(label="Your question:")
        clear_btn = gr.Button("Clear Conversation")

        state = gr.State([])

        def on_submit(user_input, history):
            response, updated_history = chatbot_predict(
                user_input, history, analyzer, context_summary
            )
            return "", updated_history, updated_history

        msg.submit(on_submit, [msg, state], [msg, state, chatbot])
        clear_btn.click(lambda: [], None, chatbot)
        clear_btn.click(lambda: [], None, state)

        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# -------------------------------------------------------------------
# Command-Line Interface
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a project folder and optionally launch a chatbot."
    )
    parser.add_argument(
        "-targetFolder",
        required=True,
        help="Path to the target folder to analyze."
    )
    parser.add_argument(
        "--chatbot",
        action="store_true",
        help="Launch a local chatbot after analysis."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    analyzer = ProjectAnalyzer(project_dir=args.targetFolder)

    analyzer.analyze_project()
    analyzer.generate_developer_guide()

    if args.chatbot:
        logging.info("Starting local chatbot UI on http://localhost:7860 ...")
        start_chatbot(analyzer)
    else:
        logging.info("Analysis complete. Rerun with --chatbot to start an interactive UI.")

if __name__ == "__main__":
    main()
