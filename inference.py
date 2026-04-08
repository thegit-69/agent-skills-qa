import asyncio
import os
import json
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

from agent_skills_qa.client import AgentSkillsQaEnv
from agent_skills_qa.models import AgentSkillsQaAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("agent_skills_qa_TASK", "agent_skills_qa")
BENCHMARK = os.getenv("agent_skills_qa_BENCHMARK", "agent_skills_qa")
IMAGE_NAME = os.getenv("IMAGE_NAME", "agent_skills_image:latest")

MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 1500
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert AI software engineer fixing broken Claude Agent SKILL.md files.
    You have 3 tools:
    1. {"tool": "read_file", "filepath": "SKILL.md"}
    2. {"tool": "write_file", "filepath": "SKILL.md", "new_content": "<fixed text>"}
    3. {"tool": "submit"}

    Reply ONLY with a raw JSON object. No markdown blocks, no thinking text.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def _single_line(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()

def extract_last_action_error(info: Optional[dict]) -> Optional[str]:
    if not info:
        return None
    raw_error = info.get("last_action_error")
    if raw_error is None:
        return None
    return str(raw_error)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = _single_line(error) if error else "null"
    done_val = str(done).lower()
    action_str = _single_line(action)
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, last_feedback: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step} of {MAX_STEPS}
        Past Actions: {history_block}
        
        Environment Feedback: 
        {last_feedback}

        CRITICAL DIRECTIVES:
        1. If the Feedback shows the contents of a file, you MUST immediately fix it and output a "write_file" JSON action. DO NOT read it again.
        2. If the Feedback says "Success: 'SKILL.md' saved", you MUST immediately output a "submit" JSON action.
        3. Only output RAW JSON.

        Example Write Action Format:
        {{
            "tool": "write_file", 
            "filepath": "SKILL.md", 
            "new_content": "---\\nname: valid-name\\ndescription: good\\n---\\n# Fixed File"
        }}
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, last_feedback: str, history: List[str]) -> tuple[AgentSkillsQaAction, str]:
    user_prompt = build_user_prompt(step, last_feedback, history)
    raw_text = "{}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        action_dict = json.loads(raw_text)
        return AgentSkillsQaAction(**action_dict), raw_text
    except Exception:
        # CRITICAL FALLBACK: If LLM fails JSON parsing, force a submit so it doesn't infinite loop!
        return AgentSkillsQaAction(tool="submit"), raw_text

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await AgentSkillsQaEnv.from_docker_image(IMAGE_NAME)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        last_feedback = result.observation.message

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, raw_json_str = get_model_action(client, step, last_feedback, history)
            result = await env.step(action)
            
            reward = result.reward or 0.0
            done = result.done
            error = extract_last_action_error(result.info)
            rewards.append(reward)
            steps_taken = step
            last_feedback = result.observation.message

            log_step(step=step, action=raw_json_str, reward=reward, done=done, error=error)
            
            # CRITICAL FIX: Truncate history so the LLM context window doesn't explode
            short_feedback = last_feedback[:100].replace('\n', ' ') + "..." if len(last_feedback) > 100 else last_feedback
            history.append(f"Step {step} - Action: {action.tool} -> Env: {short_feedback}")

            if done:
                break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())