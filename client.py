from typing import Any, Dict
from dataclasses import dataclass
from openenv.core.env_client import EnvClient

# Import the State model as well now
from agent_skills_qa.models import AgentSkillsQaAction, AgentSkillsQaObservation, AgentSkillsQaState

@dataclass
class ClientResult:
    observation: AgentSkillsQaObservation
    reward: float
    done: bool
    info: dict

# Notice we also add AgentSkillsQaState as the third generic type here instead of Any!
class AgentSkillsQaEnv(EnvClient[AgentSkillsQaAction, AgentSkillsQaObservation, AgentSkillsQaState]):
    
    def _parse_result(self, data: Dict[str, Any]) -> ClientResult:
        """Maps the raw JSON from the Docker container to our Pydantic model."""
        observation = AgentSkillsQaObservation(
            message=data.get("message", "No message received"),
            reward=data.get("reward", 0.0),
            done=data.get("done", False)
        )
        return ClientResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
            info=data.get("info", {}),
        )

    def _parse_state(self, data: Dict[str, Any]) -> AgentSkillsQaState:
        """Maps the state JSON from the server into our State Pydantic model."""
        return AgentSkillsQaState(
            episode_id=data.get("episode_id", "unknown"),
            files=data.get("files", {}),
            difficulty=data.get("difficulty", "unknown"),
            step_count=data.get("step_count", 0)
        )

    def _step_payload(self, action: AgentSkillsQaAction) -> Dict[str, Any]:
        """Converts the Pydantic action into a raw dictionary to send over the network."""
        # .model_dump() is the Pydantic v2 way of converting a model to a dictionary
        return action.model_dump()