from pydantic import BaseModel, Field

class AgentSkillsQaAction(BaseModel):
    action_type: str = Field(..., description="Must be 'edit_skill' or 'submit_final'")
    content: str = Field(..., description="The complete text of the rewritten SKILL.md file")

class AgentSkillsQaObservation(BaseModel):
    feedback: str = Field(default="", description="Parser errors or unit test results")
    current_skill_text: str = Field(default="", description="The current state of the file")
    done: bool = Field(default=False, description="Whether the episode has finished")
    reward: float = Field(default=0.0, description="The score from 0.0 to 1.0")
    metadata: dict = Field(default_factory=dict, description="Extra info")