import os
import re
import yaml
from openai import OpenAI

def _get_files(state):
    """Safely extract the file dictionary from the OpenEnv state object."""
    if hasattr(state, 'files'):
        return state.files
    elif isinstance(state, dict) and 'files' in state:
        return state['files']
    return {}

def grade_easy(state, *args, **kwargs) -> float:
    files = _get_files(state)
    if "SKILL.md" not in files: return 0.0
    
    text = files["SKILL.md"]
    yaml_match = re.search(r'^---\n(.*?)\n---', text, re.DOTALL | re.MULTILINE)
    if not yaml_match: return 0.0
    
    try:
        frontmatter = yaml.safe_load(yaml_match.group(1))
        name = frontmatter.get('name', '')
        if not re.match(r'^[a-z0-9\-]+$', name) or "claude" in name:
            return 0.5
        return 1.0 # Perfect Score
    except:
        return 0.0

def grade_medium(state, *args, **kwargs) -> float:
    files = _get_files(state)
    if "SKILL.md" not in files: return 0.0
    if "schema.md" not in files: return 0.2
    if "schema.md" not in files["SKILL.md"].lower(): return 0.6
    if "object" in files["SKILL.md"]: return 0.8
    return 1.0 # Perfect Score

def grade_hard(state, *args, **kwargs) -> float:
    files = _get_files(state)
    code = files.get("script.py", "")
    if not code: return 0.0
    if "pass" in code or "timeout = 47" in code: return 0.3
    
    try:
        # The true "Valid Signal" - LLM Judging the final file!
        client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.getenv("HF_TOKEN")
        )
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": "You are a strict code grader. Review this python code. Check if it avoids magic numbers and handles errors properly. Reply ONLY with a single float number between 0.0 and 1.0 representing the score."},
                {"role": "user", "content": code}
            ],
            max_tokens=10
        )
        match = re.search(r'0\.\d+|1\.0', response.choices[0].message.content.strip())
        return float(match.group()) if match else 0.5
    except:
        return 0.4