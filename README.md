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

# Agent Skills QA: A Multi-Step File Editing Benchmark for Autonomous SWE Agents

A Reinforcement Learning sandbox designed to evaluate an LLM agent's ability to act as a Software Engineer. The agent is placed in an environment with broken Claude Agent `SKILL.md` or Python script files and must use file-system tools to iteratively read, debug, and fix them.

## 💡 Why It Matters
A common failure mode in modern Autonomous SWE Agents is the **"Lazy Loop."** When faced with generating long, complex code corrections, LLMs often get intimidated and trap themselves in infinite loops (e.g., repeatedly reading the same file) or prematurely submitting failed code. 

**Agent Skills QA** specifically benchmarks an LLM's ability to break out of these loops, safely navigate strict action-sequence guardrails, and execute multi-step, multi-file modifications using a mathematically robust **Potential-Based Reward System**.

---

## 🧠 The RL Problem Formulation

To successfully train agents without succumbing to reward-hacking, this environment is built on strict Reinforcement Learning principles:

* **State:** The current contents of the target files within the container, represented to the agent via textual environment feedback (e.g., syntax errors, file contents).
* **Action Space (`AgentSkillsQaAction`):** A discrete, highly constrained tool space.
    * `read_file` (filepath): Discover the broken code.
    * `write_file` (filepath, new_content): Apply the fix.
    * `submit` (): Terminate the episode.
* **Reward Function (Anti-Sparse Shaping):** Evaluated dynamically via absolute logic and a `Qwen2.5-72B-Instruct` LLM Judge. To avoid the "sparse reward" problem, the environment grants intermediate partial credit (`+0.10` to `+0.70`) for successful state-discovery (reading) and partial state-transitions (writing), withholding the final `+0.20` until a successful `submit` action. Spamming tools or repeating actions strictly yields `0.00`.

---

## 🎯 The Tasks (Auto-Randomized on Reset)
1. **Easy (YAML Validation):** The agent must read `SKILL.md`, detect illegal naming conventions (uppercase, reserved words like 'claude', length limits), and rewrite the frontmatter correctly.
2. **Medium (Progressive Disclosure):** The agent must read a bloated `SKILL.md`, extract a large JSON schema into a new `schema.md` file, and update the original file to correctly link to the new schema.
3. **Hard (Code Debugging):** The agent must read `script.py`, remove magic numbers (e.g., hardcoded timeouts), and fix lazy `except: pass` error handling. Evaluated dynamically by the LLM Judge.

---

## 📈 Example Agent Trajectory
Our baseline inference script utilizes strict State-Machine Prompting to achieve perfect task execution. Below is a raw output log demonstrating the agent flawlessly solving the **Medium Task** (Multi-file editing) without falling into a loop:

```text
[START] task=medium env=agent_skills_qa model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"tool": "read_file", "filepath": "SKILL.md"} reward=0.10 done=false error=null
[STEP] step=2 action={"tool": "write_file", "filepath": "SKILL.md", "new_content": "# Skill Documentation..."} reward=0.40 done=false error=null
[STEP] step=3 action={"tool": "write_file", "filepath": "schema.md", "new_content": "# Schema Content..."} reward=0.30 done=false error=null
[STEP] step=4 action={"tool": "submit"} reward=0.20 done=true error=null
[END] success=true steps=4 score=1.00 rewards=0.10,0.40,0.30,0.20
```

---

## 🚀 Quick Start (Local Inference)

The easiest way to test the environment is using the pre-configured baseline script. 

1. Create a `.env` file in the root directory with your Hugging Face token (required for the Hard Task's LLM Judge):
   ```env
   HF_TOKEN=your_hf_token_here
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

This project is fully compliant with Hugging Face Spaces.

**Option 1: Standard Git (Recommended)**
```bash
git remote add hf [https://huggingface.co/spaces/](https://huggingface.co/spaces/)<your-hf-username>/<your-space-name>
git add .
git commit -m "Deploy to HF"
git push hf main --force
```

**Option 2: OpenEnv CLI**
```bash
uv run openenv validate
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
├── inference.py                # Baseline Qwen2.5 Agent loop (State-Machine Prompted)
└── server/
    ├── app.py                  # FastAPI application & entry point
    ├── agent_skills_qa_environment.py  # Core Environment (The Teacher)
    └── grader.py               # Absolute Final Exam Grader (Phase 2)
```

---

# Contributions

### This project is done by ***Dasarath C*** and ***Rohan V*** for the Round-1 submission of the OpenEnv Hackathon conducted by Scaler School of Technology.
```