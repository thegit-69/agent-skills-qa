```markdown
---
title: Agent Skills QA Environment
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - agents
---
```
# Agent Skills QA Environment

A Reinforcement Learning sandbox designed to evaluate an LLM agent's ability to act as a Software Engineer. The agent is placed in an environment with broken Claude Agent `SKILL.md` or Python script files and must use file-system tools to read, debug, and fix them.

Features a mathematically robust **Potential-Based Reward System** to guide agents without succumbing to reward-hacking, and a strict absolute grader for final evaluation.

## 🎯 The Tasks (Auto-Randomized on Reset)
1. **Easy (YAML Validation):** The agent must read `SKILL.md`, detect illegal naming conventions (uppercase, reserved words like 'claude', length limits), and rewrite the frontmatter correctly.
2. **Medium (Progressive Disclosure):** The agent must read a bloated `SKILL.md`, extract a large JSON schema into a new `schema.md` file, and update the original file to link to it.
3. **Hard (Code Debugging):** The agent must read `script.py`, remove magic numbers (e.g., hardcoded timeouts), and fix lazy `except: pass` error handling. Evaluated dynamically by a `Qwen2.5-72B-Instruct` LLM Judge.

---

## 🚀 Quick Start (Local Inference)

The easiest way to test the environment is using the pre-configured baseline script. 

1. Create a `.env` file in the root directory with your Hugging Face token (required for the Hard Task's LLM Judge):
   ```env
   HF_TOKEN=your_hf_token_here
   LOCAL_IMAGE_NAME=agent_skills_image:latest
   ```

2. Build the Docker image (Note: The Dockerfile is in the root directory):
   ```bash
   docker build -t agent_skills_image:latest .
   ```

3. Run the baseline agent:
   ```bash
   uv run python inference.py
   ```

---

## 🛠️ Environment Details

### Action Schema (`AgentSkillsQaAction`)
The agent interacts with the environment using exactly 3 tools:
- `tool` (str): Must be `"read_file"`, `"write_file"`, or `"submit"`.
- `filepath` (str, optional): The target file (e.g., `"SKILL.md"`).
- `new_content` (str, optional): The new text to save when using `write_file`.

### Observation Schema (`AgentSkillsQaObservation`)
- `message` (str): Text feedback from the environment (e.g., file contents, success messages, or error warnings).
- `reward` (float): The partial reward earned on this specific step.
- `done` (bool): `True` if the agent submits or hits the step limit.

### Reward Shaping (Anti-Hack System)
Rewards are awarded using a Potential-Based delta system (Total max score = `1.0`).
- **Exploration:** `+0.10` for successfully reading a file.
- **Partial Progress:** Variable rewards (up to `0.70`) for writing correct code fixes to the files.
- **Submission:** The final `+0.20` is strictly reserved for calling the `submit` tool. 
- *Note: Spamming tools or rewriting broken code recalculates potential and strictly returns `0.00`.*

---

## 🧪 Development & Testing

We provide dedicated scripts to test both the real-time Environment (Phase 1) and the final Grader (Phase 2).

### 1. Test the Evaluation Grader Locally
Ensure your absolute grader logic mathematically aligns with the environment constraints without waiting for cloud pipelines:
```bash
uv run python test_grader.py
```

### 2. Manual Human-in-the-Loop Testing
Play the game yourself via the terminal to test reward limits and edge cases:
```bash
# Start the container with your secrets
docker run --name my_agent_env -p 8000:8000 --env-file .env agent_skills_image:latest

# In a new terminal, run the interactive prompt
uv run python human_test.py
```

### 3. OpenEnv UI
When the Docker container is running, navigate to `http://localhost:8000/web` in your browser to use the graphical OpenEnv testing sandbox.

---

## ☁️ Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the OpenEnv CLI. This automatically packages the environment, sets up the FastAPI server, and exposes the required evaluation endpoints.

```bash
# Validate your openenv.yaml configuration
uv run openenv validate

# Push to your Hugging Face Space
uv run openenv push --repo-id <your-hf-username>/<your-space-name>
```

---

## 📁 Project Structure

```text
agent_skills_qa/
├── Dockerfile                  # Container image definition (Root level)
├── openenv.yaml                # OpenEnv Hackathon manifest
├── pyproject.toml              # Dependencies
├── .env                        # Local secrets (Not tracked in git)
├── client.py                   # Pydantic <-> JSON Translator
├── models.py                   # Action and Observation Pydantic models
├── inference.py                # Baseline Qwen2.5 Agent loop
└── server/
    ├── app.py                  # FastAPI application & entry point
    ├── agent_skills_qa_environment.py  # Core Environment (The Teacher)
    └── grader.py               # Absolute Final Exam Grader (Phase 2)
```

# Contributions

### This project is done by **Dasarath C** and **Rohan V** for the Round-1 submission of Open-env hackathon conducted by Scalar school of technology.
