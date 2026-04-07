# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AgentSkillsQaAction, AgentSkillsQaObservation
except ImportError:
    from models import AgentSkillsQaAction, AgentSkillsQaObservation


class AgentSkillsQaEnvironment(Environment):
    """
    Agent Skills QA Environment.
    Tests an agent's ability to debug and fix SKILL.md files.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # The 3 required tasks (Easy, Medium, Hard)
    TASKS = [
        # Task 1 (Easy): Syntax Error - Name has spaces, description missing
        "---\nname: PDF PROCESSOR\n# Missing description\n---\n# Instructions\nRun the tool.",
        
        # Task 2 (Medium): Progressive Disclosure - Too long, needs references.md
        "---\nname: weather-fetcher\ndescription: Fetches weather.\n---\n# Instructions\n[Pretend there are 10,000 words of heavy API documentation here that the agent needs to move to references.md]",
        
        # Task 3 (Hard): Logic Error - Ambiguous instructions
        "---\nname: csv-parser\ndescription: Parses CSVs.\n---\n# Instructions\nWrite a python script to parse CSV. Don't handle missing columns."
    ]

    def __init__(self):
        """Initialize the agent_skills_qa environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.current_task = ""

    def reset(self) -> AgentSkillsQaObservation:
        """
        Reset the environment and load a new broken skill.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Pick a random broken skill to fix
        self.current_task = random.choice(self.TASKS)

        return AgentSkillsQaObservation(
            feedback="Welcome to Agent Skills QA. Please fix the formatting and logic in the following SKILL.md file.",
            current_skill_text=self.current_task,
            done=False,
            reward=0.0,
        )

    def step(self, action: AgentSkillsQaAction) -> AgentSkillsQaObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0
        done = False
        feedback = ""

        # === GRADER LOGIC ===
        
        # 1. Check if the agent is just editing or submitting the final version
        if action.action_type not in ["edit_skill", "submit_final"]:
            feedback = f"Error: Invalid action_type '{action.action_type}'. Must be 'edit_skill' or 'submit_final'."
            return AgentSkillsQaObservation(
                feedback=feedback, current_skill_text=action.content, done=done, reward=0.0, metadata={"step": self._state.step_count}
            )

        # 2. Grade Task 2 (Progressive Disclosure)
        if "weather-fetcher" in self.current_task:
            # The agent should have replaced the massive text with a markdown link
            if "[Pretend there are 10,000 words" not in action.content:
                # Good! They removed the bloat.
                reward += 0.3
                feedback += "- Successfully removed bloated documentation.\n"
                
                # Check if they added a reference link
                if "](references.md)" in action.content or "](reference.md)" in action.content:
                    reward += 0.3
                    feedback += "- Successfully added reference link.\n"
                else:
                    feedback += "- Error: You removed the text but forgot to add a markdown link to 'references.md'.\n"
            else:
                feedback += "- Error: The SKILL.md file is still too large. You must extract the documentation.\n"

        # (We will add the logic for Task 1 and Task 3 here later)
        
        # === END GRADER LOGIC ===

        # Handle final submission
        if action.action_type == "submit_final":
            done = True
            feedback += f"\nEpisode complete. Final Python Score: {reward}/0.6"
            # (API Verification will go here later)
        elif self._state.step_count >= 15:
            done = True
            feedback += "\nMax steps reached."

        return AgentSkillsQaObservation(
            feedback=feedback,
            current_skill_text=action.content,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count}
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state