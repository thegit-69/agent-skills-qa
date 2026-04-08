from pydantic import BaseModel, Field
from typing import Optional, Dict

class AgentSkillsQaAction(BaseModel):
    tool: str = Field(
        ..., 
        description="The tool to use. Options: 'read_file', 'write_file', 'run_test', 'submit'."
    )
    filepath: Optional[str] = Field(
        default=None, 
        description="The name of the file to interact with (e.g., 'SKILL.md' or 'script.py'). Required for read_file and write_file."
    )
    new_content: Optional[str] = Field(
        default=None, 
        description="The complete text to save. ONLY use this when tool is 'write_file'."
    )

class AgentSkillsQaObservation(BaseModel):
    message: str = Field(
        ..., 
        description="Feedback from the system. Contains file contents, test results, or error messages."
    )
    # The Web UI requires these two fields to exist so it can render the dashboard!
    reward: float = Field(
        default=0.0, 
        description="The current reward."
    )
    done: bool = Field(
        default=False, 
        description="Whether the episode has finished."
    )

class AgentSkillsQaState(BaseModel):
    episode_id: str = Field(
        default="episode_1",
        description="Unique identifier for the current playthrough."
    )
    files: Dict[str, str] = Field(
        ..., 
        description="Virtual file system tracking the current state of all files in the directory."
    )
    difficulty: str = Field(
        ..., 
        description="The current task level: 'easy', 'medium', or 'hard'."
    )
    step_count: int = Field(
        default=0, 
        description="Tracks how many moves the agent has made to prevent infinite loops."
    )