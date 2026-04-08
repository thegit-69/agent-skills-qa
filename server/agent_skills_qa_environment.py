import yaml
import uuid
from openenv.core.env_server import Environment
from ..models import AgentSkillsQaAction, AgentSkillsQaObservation, AgentSkillsQaState

class AgentSkillsQaEnvironment(Environment):
    
    def reset(self, task_name: str = "medium") -> AgentSkillsQaObservation:
        self.difficulty = task_name.lower()
        self.step_count = 0
        self.files = {}
        
        # Setup the virtual file system based on difficulty
        if self.difficulty == "easy":
            self.files["SKILL.md"] = "name: CalculatorSkill\nversion:1.0.0\nentrypoint: calc.py\n---\nRuns math."
            self.files["calc.py"] = "def add(a, b): return a + b"
            error_msg = "Initialization failed: Could not parse SKILL.md. Check YAML formatting."
            
        elif self.difficulty == "medium":
            self.files["SKILL.md"] = "name: DataSkill\nversion: 1.0.0\nentrypoint: data_loader.py\n---\nLoads data."
            error_msg = "Initialization failed: Entrypoint script referenced in SKILL.md is missing from the directory."
            
        else: # hard
            self.files["SKILL.md"] = "name: MathSkill\nversion: 1.0.0\nentrypoint: script.py\n---\nAdds two numbers."
            self.files["script.py"] = "def execute(a, b):\n    return a - b"
            error_msg = "Initialization successful, but automated tests are failing. Use 'run_test' to debug."

        self._state = AgentSkillsQaState(
            episode_id=uuid.uuid4().hex,
            files=self.files,
            difficulty=self.difficulty,
            step_count=self.step_count
        )
        
        return AgentSkillsQaObservation(
            message=f"Environment Started. Task: {self.difficulty.upper()}.\nSystem Alert: {error_msg}\nUse 'read_file' to inspect the directory files."
        )

    def step(self, action: AgentSkillsQaAction) -> AgentSkillsQaObservation:
            self._state.step_count += 1
            
            # Prevent infinite loops
            if self._state.step_count > 15:
                return AgentSkillsQaObservation(
                    message="Max steps reached. Forcing submission.", 
                    reward=0.0, 
                    done=True
                )

            # TOOL: Read File
            if action.tool == "read_file":
                if not action.filepath or action.filepath not in self._state.files:
                    return AgentSkillsQaObservation(
                        message=f"Error: File '{action.filepath}' not found.", 
                        reward=0.0, 
                        done=False
                    )
                return AgentSkillsQaObservation(
                    message=f"Contents of {action.filepath}:\n\n{self._state.files[action.filepath]}", 
                    reward=0.0, 
                    done=False
                )

            # TOOL: Write File
            elif action.tool == "write_file":
                if not action.filepath or not action.new_content:
                    return AgentSkillsQaObservation(
                        message="Error: Both filepath and new_content are required.", 
                        reward=0.0, 
                        done=False
                    )
                
                self._state.files[action.filepath] = action.new_content
                return AgentSkillsQaObservation(
                    message=f"Success: '{action.filepath}' was updated/created.", 
                    reward=0.0, 
                    done=False
                )

            # TOOL: Run Test
            elif action.tool == "run_test":
                test_output = self._simulate_test_run()
                return AgentSkillsQaObservation(
                    message=f"Test Suite Output:\n{test_output}", 
                    reward=0.0, 
                    done=False
                )

            # TOOL: Submit
            elif action.tool == "submit":
                reward = self._calculate_reward()
                return AgentSkillsQaObservation(
                    message=f"Final Submission Graded. Score: {reward}", 
                    reward=reward, 
                    done=True
                )
                
            else:
                return AgentSkillsQaObservation(
                    message="Unknown tool.", 
                    reward=0.0, 
                    done=False
                )
            
    @property
    def state(self) -> AgentSkillsQaState:
        return self._state

    def _is_yaml_valid(self, text: str) -> bool:
        try:
            header = text.split("---")[0]
            yaml.safe_load(header)
            return True
        except:
            return False

    def _simulate_test_run(self) -> str:
        if "SKILL.md" not in self._state.files:
            return "FAIL: SKILL.md is missing entirely."
            
        if not self._is_yaml_valid(self._state.files["SKILL.md"]):
            return "FAIL: SKILL.md YAML header is malformed."
            
        if self.difficulty == "medium":
            if "data_loader.py" not in self._state.files:
                return "FAIL: FileNotFoundError - 'data_loader.py' not found."
            return "PASS: All tests passed."
            
        if self.difficulty == "hard":
            script_code = self._state.files.get("script.py", "")
            if "return a + b" not in script_code and "return a+b" not in script_code:
                return "FAIL: AssertionError in script.py - Expected execute(2, 2) to equal 4, got 0."
            return "PASS: All tests passed."
            
        return "PASS: YAML is valid."

    def _calculate_reward(self) -> float:
        files = self._state.files
        if "SKILL.md" not in files or not self._is_yaml_valid(files["SKILL.md"]):
            return 0.0
            
        score = 0.4 
        
        if self.difficulty == "easy":
            score = 1.0 
            
        elif self.difficulty == "medium":
            if "data_loader.py" in files and len(files["data_loader.py"]) > 5:
                score = 1.0 
            else:
                score = 0.5 
                
        elif self.difficulty == "hard":
            script_code = files.get("script.py", "")
            if "return a + b" in script_code or "return a+b" in script_code:
                score = 1.0 
            elif "script.py" in files:
                score = 0.6 
                
        return score