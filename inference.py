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

# 1. EVALUATOR-COMPLIANT VARIABLES
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")

# The evaluator injects this dynamically during Phase 2. The fallback is for your local testing.
# Checks IMAGE_NAME first, then LOCAL_IMAGE_NAME, then falls back to your local tag
IMAGE_NAME = os.environ.get("IMAGE_NAME", os.environ.get("LOCAL_IMAGE_NAME", "agent_skills_image:latest"))

MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 1500
SUCCESS_SCORE_THRESHOLD = 0.5
BENCHMARK = "agent_skills_qa"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an elite autonomous Software Engineer agent. Your task is to fix broken Claude Agent `SKILL.md` files.
    You operate in an environment that gives partial rewards for correct steps. You have up to 10 steps to achieve a perfect score (1.0).

    AVAILABLE TOOLS:
    1. {"tool": "read_file", "filepath": "SKILL.md"}
       - Use this FIRST to see the broken code.
    2. {"tool": "write_file", "filepath": "SKILL.md", "new_content": "<entire_fixed_file_content_here>"}
       - Use this IMMEDIATELY after reading. You MUST include the completely fixed code in the "new_content" field.
    3. {"tool": "submit"}
       - Use this ONLY after you have successfully used write_file and received a success message.

    STRICT EXECUTION RULES:
    - RULE 1: NEVER use `read_file` twice in a row. Once you have the file contents, your next move MUST be `write_file`.
    - RULE 2: When using `write_file`, you must generate the ACTUAL repaired code. Do not use placeholders.
    - RULE 3: Do not use `submit` until you have written the fixed file.
    - RULE 4: Output ONLY a valid JSON object. No explanations, no markdown formatting (e.g., no ```json).

    EXAMPLE SUCCESS SEQUENCE:
    [Step 1] You output: {"tool": "read_file", "filepath": "SKILL.md"}
    [Env Feedback] "File contents: def broken_func():\\n  pass"
    [Step 2] You output: {"tool": "write_file", "filepath": "SKILL.md", "new_content": "def fixed_func():\\n    print('fixed')"}
    [Env Feedback] "Success: 'SKILL.md' updated."
    [Step 3] You output: {"tool": "submit"}
    """
).strip()
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", " ").replace("\r", " ").strip() if error else "null"
    done_val = str(done).lower()
    action_str = action.replace("\n", " ").replace("\r", " ").strip()
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, last_feedback: str, history: List[str]) -> str:
    history_block = "\n".join(history[-3:]) if history else "No previous actions."
    
    return textwrap.dedent(
        f"""
        --- CURRENT STATE ---
        Step: {step} of {MAX_STEPS}
        Past Actions: 
        {history_block}
        
        --- ENVIRONMENT FEEDBACK --- 
        {last_feedback}

        --- YOUR NEXT DIRECTIVE ---
        Analyze the feedback above:
        1. If it shows the broken file contents -> Output a "write_file" JSON with the fixed `new_content`.
        2. If it says the file was saved/updated -> Output the "submit" JSON.
        3. If you haven't read the file yet -> Output the "read_file" JSON.
        
        Output ONLY the raw JSON dictionary.
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
    except Exception as e:
        # Silently return the submit action, but embed the API error in the raw_text 
        # so it shows up safely in the 'action' or 'error' column of the [STEP] log
        return AgentSkillsQaAction(tool="submit"), f'{{"error": "{str(e)}"}}'

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = None

    # MASTER TRY BLOCK: Catches any fatal errors in initialization or the loop
    try:
        # 1. Define your fallback URL (in case the evaluator forgets to inject it)
        default_url = "https://kalki777-agent-skills-qa-env.hf.space"
        space_url = os.environ.get("OPENENV_BASE_URL") or os.environ.get("ENV_URL", default_url)

        # 2. Try Docker FIRST
        try:
            env = await AgentSkillsQaEnv.from_docker_image(
                IMAGE_NAME,
                env_vars={"HF_TOKEN": HF_TOKEN} # Injects secrets for local Hard Task testing
            )
        except Exception as docker_err:
            # 3. If Docker fails (Exit 125 in cloud), fallback to the Live URL
            if space_url:
                env = AgentSkillsQaEnv(base_url=space_url)
            else:
                raise RuntimeError(f"Docker failed and no URL available. {docker_err}")
                
        # 4. THE MULTI-TASK LOOP (Properly out-dented to run regardless of Docker or URL)
        for current_task in ["easy", "medium", "hard"]:
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)

            try:
                # OpenEnv allows passing kwargs to reset for task selection
                result = await env.reset(task=current_task)
                last_feedback = result.observation.message

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    action, raw_json_str = get_model_action(client, step, last_feedback, history)
                    result = await env.step(action)
                    
                    reward = result.reward or 0.0
                    done = result.done
                    error = str(result.info.get("last_action_error")) if (result.info and result.info.get("last_action_error")) else None
                    
                    rewards.append(reward)
                    steps_taken = step
                    last_feedback = result.observation.message

                    log_step(step=step, action=raw_json_str, reward=reward, done=done, error=error)
                    
                    short_feedback = last_feedback[:100].replace('\n', ' ') + "..." if len(last_feedback) > 100 else last_feedback
                    history.append(f"Step {step} - Action: {action.tool} -> Env: {short_feedback}")

                    if done:
                        break

                score = min(max(sum(rewards), 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

            except Exception as task_err:
                log_step(step=steps_taken + 1, action="{}", reward=0.0, done=True, error=f"Task Crash: {str(task_err)}")
                log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

    except Exception as global_err:
        print(f"[START] task=fatal_crash env={BENCHMARK} model={MODEL_NAME}")
        log_step(step=1, action="{}", reward=0.0, done=True, error=f"Fatal Init: {str(global_err)}")
        log_end(success=False, steps=0, score=0.0, rewards=[])

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as fatal_err:
        # If the absolute worst happens, still print the mandatory logs and exit 0
        print(f"[START] task=fatal_crash env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=1 action={{}} reward=0.0 done=true error=Outer Crash: {str(fatal_err)}", flush=True)
        print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)