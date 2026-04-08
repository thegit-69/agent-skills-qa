from pydantic import BaseModel, Field
from typing import Optional, Dict

# ---------------------------------------------------------
# THE CURRICULUM (The broken files to fix)
# ---------------------------------------------------------
TASK_1_SKILL = """---
name: BAD_NAME_WITH_CAPS_AND_claude_WORD
description: This description is okay but the name is terrible and breaks the rules.
---
# Simple Skill
Just a basic skill.
"""

TASK_2_SKILL = """---
name: data-processor
description: Processes data.
---
# Data Processor
This is a massive file. Here is the giant 500-line database schema:
{ "type": "object", "properties": { "id": { "type": "string" }, "name": { "type": "string" } } }
"""

TASK_3_SKILL = """---
name: calc-skill
description: Does math safely.
---
# Math Skill
"""

TASK_3_CODE = """
def divide_numbers(a, b):
    # Voodoo constant / Magic Number
    timeout = 47 
    try:
        return a / b
    except:
        # Lazy error handling / punting to the agent
        pass 
"""

# ---------------------------------------------------------
# OPENENV SCHEMAS
# ---------------------------------------------------------
class AgentSkillsQaAction(BaseModel):
    tool: str = Field(
        ..., 
        description="The tool to use. Options: 'read_file', 'write_file', 'submit'."
    )
    filepath: Optional[str] = Field(
        default=None, 
        description="The name of the file (e.g., 'SKILL.md'). Required for read and write."
    )
    new_content: Optional[str] = Field(
        default=None, 
        description="The text to save. ONLY use when tool is 'write_file'."
    )

class AgentSkillsQaObservation(BaseModel):
    message: str = Field(..., description="Feedback from the system.")
    reward: float = Field(default=0.0, description="The current reward [0.0 to 1.0].")
    done: bool = Field(default=False, description="Whether the episode has finished.")

class AgentSkillsQaState(BaseModel):
    episode_id: str = Field(default="episode_1")
    files: Dict[str, str] = Field(..., description="Virtual file system.")
    difficulty: str = Field(..., description="The current task level.")
    step_count: int = Field(default=0)