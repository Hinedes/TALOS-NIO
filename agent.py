import os
import json
import re
from pathlib import Path
from smolagents import CodeAgent, OpenAIModel, tool

# 1. Define the single strict boundary (detect WSL vs Windows)
def _get_base_dir() -> Path:
    """Auto-detect whether running in WSL and return correct BASE_DIR."""
    try:
        # Check if running in WSL: /proc/version contains 'microsoft' or 'wsl'
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            if 'microsoft' in content or 'wsl' in content:
                # WSL: use Linux path
                return Path.home() / "TALOS"
    except FileNotFoundError:
        pass
    
    # Windows or unable to detect WSL: use Windows path
    return Path(r"C:\TALOS").resolve()

BASE_DIR = _get_base_dir().resolve()

def is_path_safe(requested_path: str) -> bool:
    try:
        target = Path(requested_path).resolve()
        return target.is_relative_to(BASE_DIR)
    except Exception:
        return False

# 2. Define locked-down tools
@tool
def read_safe(filepath: str) -> str:
    """Read a UTF-8 text file inside the allowed directory.

    Args:
        filepath: Absolute or relative path to the file to read.

    Returns:
        The file contents on success, or an error/access-denied message.
    """
    if not is_path_safe(filepath):
        return f"Access Denied: Cannot read {filepath}."
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_safe(filepath: str, content: str) -> str:
    """Write UTF-8 text to a file inside the allowed directory.

    Args:
        filepath: Absolute or relative path to the file to write.
        content: Full text payload to write to the target file.

    Returns:
        A success message or an error/access-denied message.
    """
    if not is_path_safe(filepath):
        return f"Access Denied: Cannot write to {filepath}."
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully updated {filepath}."
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def run_training() -> str:
    """Execute training from the sandbox root and return logs.

    Returns:
        The training stdout on success, or stderr/stdout details on failure.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["uv", "run", "train.py"], 
            capture_output=True, 
            text=True, 
            check=True, 
            cwd=str(BASE_DIR)
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Training crashed. Error log:\n{e.stderr}\nOutput log:\n{e.stdout}"


@tool
def parse_training_log(log_text: str) -> str:
    """Parse TALOS training logs and return structured metrics as JSON.

    Args:
        log_text: Raw output text returned by run_training.

    Returns:
        A JSON string containing parse status, best ATE, and latest ESKF ATE.
    """
    if not log_text or not log_text.strip():
        return json.dumps(
            {
                "ok": False,
                "error": "empty_log",
                "message": "No training output was provided.",
            },
            ensure_ascii=True,
        )

    best_ate_matches = re.findall(r"Best ATE\s*:\s*([0-9]+(?:\.[0-9]+)?)m", log_text)
    eskf_ate_matches = re.findall(r"ESKF ATE:\s*([0-9]+(?:\.[0-9]+)?)m", log_text)
    best_round_matches = re.findall(r"Achieved\s*:\s*Round\s*(\d+)", log_text)

    payload = {
        "ok": bool(best_ate_matches or eskf_ate_matches),
        "training_crashed": "Training crashed." in log_text,
        "best_ate_m": float(best_ate_matches[-1]) if best_ate_matches else None,
        "latest_eskf_ate_m": float(eskf_ate_matches[-1]) if eskf_ate_matches else None,
        "best_round": int(best_round_matches[-1]) if best_round_matches else None,
        "all_eskf_ate_m": [float(x) for x in eskf_ate_matches],
    }

    if not payload["ok"]:
        payload["message"] = "No ATE metrics found in training output."

    return json.dumps(payload, ensure_ascii=True)

# 3. Read your custom system prompt
with open(BASE_DIR / "system.txt", "r", encoding="utf-8") as f:
    custom_instructions = f.read()

# 4. Initialize the model with the mandatory parameters
model = OpenAIModel(
    model_id="OmniClaw-V2-Q8_0",
    api_base="http://192.168.176.1:8080/v1",
    api_key="sk-no-key-required"
)

# 5. Initialize the agent
agent = CodeAgent(
    tools=[read_safe, write_safe, run_training, parse_training_log],
    model=model,
    stream_outputs=True
)

# 6. Start the loop by injecting your rules
prompt = f"""
{custom_instructions}

YOUR MISSION:
1. Read `program.md` to understand your goals and constraints. 
2. Use `read_safe` to examine `talos_controller.py`.
3. Run `run_training` to establish your baseline ATE.
4. Parse every `run_training` output with `parse_training_log` and use only parsed JSON fields for decisions.
5. Modify ONLY `talos_controller.py` to lower the ATE.
6. If Slap_Rate exceeds 1.0%, you must adjust your thresholds or loss.
7. Iterate until you beat the 4.047 baseline.
"""

agent.run(prompt)