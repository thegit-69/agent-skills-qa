import os
import re
import yaml
from openai import OpenAI

def _clamp_score(value: float) -> float:
    return max(0.01, min(0.99, float(value)))

def _get_files(state):
    """Safely extract the file dictionary from the OpenEnv state object."""
    if hasattr(state, 'files'):
        return state.files
    elif isinstance(state, dict) and 'files' in state:
        return state['files']
    return {}

def grade_easy(state=None, *args, **kwargs) -> float:
    if state is None:
        return 0.01

    files = _get_files(state)
    if "SKILL.md" not in files:
        return 0.01
    
    text = files["SKILL.md"]
    yaml_match = re.search(r'^---\n(.*?)\n---', text, re.DOTALL | re.MULTILINE)
    if not yaml_match:
        return 0.01
    
    try:
        frontmatter = yaml.safe_load(yaml_match.group(1))
        if not isinstance(frontmatter, dict):
            return 0.01
        name = frontmatter.get('name', '')
        if not re.match(r'^[a-z0-9\-]+$', name) or "claude" in name:
            return _clamp_score(0.5)
        return _clamp_score(1.0)
    except Exception:
        return 0.01

def grade_medium(state=None, *args, **kwargs) -> float:
    if state is None:
        return 0.01

    files = _get_files(state)
    if "SKILL.md" not in files:
        return 0.01
    if "schema.md" not in files:
        return _clamp_score(0.2)
    if "schema.md" not in files["SKILL.md"].lower():
        return _clamp_score(0.6)
    if "object" in files["SKILL.md"]:
        return _clamp_score(0.8)
    return _clamp_score(1.0)

def grade_hard(state=None, *args, **kwargs) -> float:
    if state is None:
        return 0.01

    files = _get_files(state)
    code = files.get("script.py", "")
    if not code:
        return 0.01
    if "pass" in code or "timeout = 47" in code:
        return _clamp_score(0.3)
    
    try:
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
            max_tokens=10,
            temperature=0,
        )

        content = (response.choices[0].message.content or "").strip()
        match = re.search(r'(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)', content)
        if not match:
            return 0.01

        return _clamp_score(float(match.group(0)))
    except Exception:
        return 0.01