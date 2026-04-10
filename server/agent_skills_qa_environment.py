import os
import re
import yaml
import uuid
import random
from openai import OpenAI
from openenv.core.env_server import Environment

from agent_skills_qa.models import (
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

    def reset(self, task: str | None = None, **kwargs) -> AgentSkillsQaObservation:
        requested_task = (task or "").strip().lower()
        if requested_task in {"easy", "medium", "hard"}:
            self.difficulty = requested_task
        else:
            self.difficulty = random.choice(["easy", "medium", "hard"])
        self.step_count = 0
        self.files = {}
        
        # --- TRACKING VARIABLES FOR REWARD SHAPING ---
        self._read_files = set()
        self.current_potential = 0.0 
        
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
        
        if self._state.step_count > 10:
            return AgentSkillsQaObservation(
                message="Max steps reached. Forcing failure.", 
                reward=0.0, done=True
            )

        if action.tool == "read_file":
            if not action.filepath or action.filepath not in self._state.files:
                return AgentSkillsQaObservation(message=f"Error: File '{action.filepath}' not found.", reward=0.0, done=False)
            
            self._read_files.add(action.filepath)
            reward, msg = self._process_dynamic_reward(f"Contents of {action.filepath}:\n\n{self._state.files[action.filepath]}")
            return AgentSkillsQaObservation(message=msg, reward=reward, done=False)

        elif action.tool == "write_file":
            if not action.filepath or action.new_content is None:
                return AgentSkillsQaObservation(message="Error: Missing filepath or new_content.", reward=0.0, done=False)
            
            self._state.files[action.filepath] = action.new_content
            reward, msg = self._process_dynamic_reward(f"Success: '{action.filepath}' saved.")
            return AgentSkillsQaObservation(message=msg, reward=reward, done=False)

        elif action.tool == "submit":
            final_grade, feedback = self._calculate_final_grade()
            # The submit reward is exactly what's left over from the partial rewards
            submit_reward = max(0.0, final_grade - self.current_potential)
            self.current_potential = final_grade
            return AgentSkillsQaObservation(message=f"FINAL GRADE: {feedback} (Score: {final_grade})", reward=submit_reward, done=True)
            
        else:
            return AgentSkillsQaObservation(message="Unknown tool.", reward=0.0, done=False)

    @property
    def state(self) -> AgentSkillsQaState:
        return self._state

    # ---------------------------------------------------------
    # POTENTIAL-BASED REWARD SHAPING (Blocks Reward Hacking)
    # ---------------------------------------------------------
    def _process_dynamic_reward(self, base_message: str) -> tuple[float, str]:
        """Calculates the current 'potential' of the environment and awards the delta."""
        new_potential = self._calculate_current_potential()
        delta_reward = max(0.0, new_potential - self.current_potential)
        
        if delta_reward > 0:
            self.current_potential = new_potential
            return delta_reward, f"{base_message}\n[Partial Progress Reward: +{delta_reward:.2f}]"
        
        return 0.0, base_message

    def _calculate_current_potential(self) -> float:
        """Evaluates how close the files are to the final goal (Returns 0.0 to 0.8)"""
        potential = 0.0
        files = self._state.files

        # 1. Reward for exploring (reading files at least once)
        if len(self._read_files) > 0:
            potential += 0.1

        # 2. TASK 1 DYNAMICS (Regex live-checking)
        if self.difficulty == "easy" and "SKILL.md" in files:
            text = files["SKILL.md"]
            yaml_match = re.search(r'^---\n(.*?)\n---', text, re.DOTALL | re.MULTILINE)
            if yaml_match:
                potential += 0.2  # Valid YAML block exists
                try:
                    frontmatter = yaml.safe_load(yaml_match.group(1))
                    name = frontmatter.get('name', '')
                    if name and len(name) <= 64:
                        potential += 0.2
                    if name and re.match(r'^[a-z0-9\-]+$', name) and "claude" not in name:
                        potential += 0.3
                except: pass

        # 3. TASK 2 DYNAMICS (Progressive Disclosure)
        elif self.difficulty == "medium":
            if "schema.md" in files:
                potential += 0.3
            if "SKILL.md" in files and "schema.md" in files["SKILL.md"].lower():
                potential += 0.2
            if "SKILL.md" in files and "object" not in files["SKILL.md"]:
                potential += 0.2

        # 4. TASK 3 DYNAMICS (Light static analysis to save API calls)
        elif self.difficulty == "hard":
            code = files.get("script.py", "")
            if code:
                if "47" not in code:
                    potential += 0.3
                if "pass" not in code:
                    potential += 0.4
                    
        # Cap potential at 0.8. The last 0.2 is strictly reserved for clicking "submit".
        return min(potential, 0.8)

    # ---------------------------------------------------------
    # FINAL GRADING LOGIC
    # ---------------------------------------------------------
    def _calculate_final_grade(self) -> tuple[float, str]:
        files = self._state.files
        if "SKILL.md" not in files:
            return 0.0, "FAIL: SKILL.md was deleted."

        if self.difficulty == "easy":
            if self._calculate_current_potential() >= 0.8:
                return 1.0, "PASS: YAML frontmatter is perfect."
            return self.current_potential, "FAIL: Formatting rules violated."

        elif self.difficulty == "medium":
            if self._calculate_current_potential() >= 0.8:
                return 1.0, "PASS: Progressive Disclosure implemented perfectly."
            return self.current_potential, "FAIL: Schema not correctly extracted and linked."

        elif self.difficulty == "hard":
            code = files.get("script.py", "")
            # Final LLM Evaluation only runs on submit to save 20min timeout limits
            llm_score = self._run_llm_judge(code)
            return llm_score, "LLM Judge Evaluation complete."
            
        return 0.0, "ERROR: Unknown state."

    def _run_llm_judge(self, code: str) -> float:
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
            match = re.search(r'0\.\d+|1\.0', score_text)
            return float(match.group()) if match else 0.5
        except Exception as e:
            print(f"LLM Judge Failed: {e}")
            return 0.8 if "Exception" in code else 0.4