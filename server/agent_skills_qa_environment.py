import os
import re
import yaml
import uuid
import random
from openai import OpenAI
from openenv.core.env_server import Environment

# Import from the local package so the environment works from this repository.
try:
    from ..models import (
        AgentSkillsQaAction,
        AgentSkillsQaObservation,
        AgentSkillsQaState,
        TASK_1_SKILL,
        TASK_2_SKILL,
        TASK_3_SKILL,
        TASK_3_CODE,
    )
except ModuleNotFoundError:
    from models import (
        AgentSkillsQaAction,
        AgentSkillsQaObservation,
        AgentSkillsQaState,
        TASK_1_SKILL,
        TASK_2_SKILL,
        TASK_3_SKILL,
        TASK_3_CODE,
    )

class AgentSkillsQaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def reset(self) -> AgentSkillsQaObservation:
        # Cycle through difficulties randomly to satisfy the 3-task hackathon rule
        self.difficulty = random.choice(["easy", "medium", "hard"])
        self.step_count = 0
        self.files = {}
        
        if self.difficulty == "easy":
            self.files["SKILL.md"] = TASK_1_SKILL
            prompt = "TASK (EASY): Fix the YAML frontmatter in SKILL.md. Name must be lowercase, no reserved words, max 64 chars."
            
        elif self.difficulty == "medium":
            self.files["SKILL.md"] = TASK_2_SKILL
            prompt = "TASK (MEDIUM): Implement 'Progressive Disclosure'. Move the giant schema out of SKILL.md into a new file called 'schema.md', and link to it in SKILL.md."
            
        else: # hard
            self.files["SKILL.md"] = TASK_3_SKILL
            self.files["script.py"] = TASK_3_CODE
            prompt = "TASK (HARD): Fix script.py. Remove magic numbers and fix the lazy 'except: pass' error handling."

        self._state = AgentSkillsQaState(
            episode_id=uuid.uuid4().hex,
            files=self.files,
            difficulty=self.difficulty,
            step_count=self.step_count
        )
        
        return AgentSkillsQaObservation(
            message=f"Environment Started.\n{prompt}\nUse 'read_file' to inspect the directory.",
            reward=0.0, done=False
        )

    def step(self, action: AgentSkillsQaAction) -> AgentSkillsQaObservation:
        self._state.step_count += 1
        
        # 1. INFINITE LOOP PROTECTION
        if self._state.step_count > 10:
            return AgentSkillsQaObservation(
                message="Max steps reached. Forcing failure.", 
                reward=0.0, done=True
            )

        # 2. TOOL EXECUTION
        if action.tool == "read_file":
            if not action.filepath or action.filepath not in self._state.files:
                return AgentSkillsQaObservation(message=f"Error: File '{action.filepath}' not found.", reward=0.0, done=False)
            return AgentSkillsQaObservation(message=f"Contents of {action.filepath}:\n\n{self._state.files[action.filepath]}", reward=0.0, done=False)

        elif action.tool == "write_file":
            if not action.filepath or not action.new_content:
                return AgentSkillsQaObservation(message="Error: Missing filepath or new_content.", reward=0.0, done=False)
            self._state.files[action.filepath] = action.new_content
            return AgentSkillsQaObservation(message=f"Success: '{action.filepath}' saved.", reward=0.0, done=False)

        elif action.tool == "submit":
            feedback, reward = self._calculate_reward()
            return AgentSkillsQaObservation(message=f"FINAL GRADE: {feedback}", reward=reward, done=True)
            
        else:
            return AgentSkillsQaObservation(message="Unknown tool.", reward=0.0, done=False)

    @property
    def state(self) -> AgentSkillsQaState:
        return self._state

    # ---------------------------------------------------------
    # THE GRADING LOGIC (The core of the hackathon)
    # ---------------------------------------------------------
    def _calculate_reward(self) -> tuple[str, float]:
        files = self._state.files
        if "SKILL.md" not in files:
            return "FAIL: SKILL.md was deleted.", 0.0

        # --- TASK 1: THE REGEX/YAML GRADER (HACK-PROOF) ---
        if self.difficulty == "easy":
            text = files["SKILL.md"]
            yaml_match = re.search(r'^---\n(.*?)\n---', text, re.DOTALL | re.MULTILINE)
            if not yaml_match:
                return "FAIL: Invalid YAML formatting.", 0.0
            try:
                frontmatter = yaml.safe_load(yaml_match.group(1))
                name = frontmatter.get('name', '')
                if not re.match(r'^[a-z0-9\-]+$', name) or "claude" in name:
                    return f"FAIL: Name '{name}' breaks character or reserved word rules.", 0.5
                return "PASS: YAML frontmatter is perfect.", 1.0
            except:
                return "FAIL: YAML parser crashed.", 0.0

        # --- TASK 2: PROGRESSIVE DISCLOSURE GRADER ---
        elif self.difficulty == "medium":
            if "schema.md" not in files:
                return "FAIL: You did not create schema.md.", 0.2
            if "schema.md" not in files["SKILL.md"].lower():
                return "FAIL: schema.md was created, but not linked in SKILL.md.", 0.6
            if "object" in files["SKILL.md"]:
                return "FAIL: You forgot to remove the schema from SKILL.md.", 0.8
            return "PASS: Progressive Disclosure implemented perfectly.", 1.0

        # --- TASK 3: THE LLM JUDGE (SEMANTIC GRADER) ---
        elif self.difficulty == "hard":
            code = files.get("script.py", "")
            if "pass" in code or "timeout = 47" in code:
                return "FAIL: You left the magic number or lazy exception in the code.", 0.3
            
            # The Ultimate "Valid Signal" - Calling the LLM to grade the code logic
            llm_score = self._run_llm_judge(code)
            if llm_score >= 0.9:
                return f"PASS: LLM Judge approved the code quality (Score: {llm_score}).", 1.0
            else:
                return f"FAIL: LLM Judge found logical flaws (Score: {llm_score}).", llm_score
                
        return "ERROR: Unknown state.", 0.0

    def _run_llm_judge(self, code: str) -> float:
        """Acts as the Semantic Grader to prevent brittle regex hacks."""
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
                max_tokens=10
            )
            score_text = response.choices[0].message.content.strip()
            # Extract the float using regex just in case the LLM yaps
            match = re.search(r'0\.\d+|1\.0', score_text)
            return float(match.group()) if match else 0.5
        except Exception as e:
            print(f"LLM Judge Failed: {e}")
            # Fallback heuristic if network fails during judging
            return 0.8 if "Exception" in code else 0.4