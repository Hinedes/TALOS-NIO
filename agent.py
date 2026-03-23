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


def _is_wsl() -> bool:
    """Return True when running inside WSL."""
    try:
        with open('/proc/version', 'r', encoding='utf-8') as f:
            content = f.read().lower()
            return 'microsoft' in content or 'wsl' in content
    except FileNotFoundError:
        return False


def _get_llama_api_base() -> str:
    """Resolve llama-server base URL with env override and WSL-aware fallback."""
    # Highest priority: explicit override for custom network setups.
    explicit_base = os.environ.get("LLAMA_API_BASE", "").strip()
    if explicit_base:
        return explicit_base

    if _is_wsl():
        # In WSL, the default route gateway is the Windows host.
        try:
            with open('/proc/net/route', 'r', encoding='utf-8') as f:
                next(f, None)
                for line in f:
                    fields = line.split()
                    if len(fields) >= 3 and fields[1] == '00000000':
                        gateway_hex = fields[2]
                        octets = [
                            str(int(gateway_hex[i:i + 2], 16))
                            for i in range(0, 8, 2)
                        ]
                        windows_host_ip = '.'.join(reversed(octets))
                        return f"http://{windows_host_ip}:8080/v1"
        except Exception:
            pass

        # Fallback for distros where this host alias is available.
        return "http://host.wsl.internal:8080/v1"

    return "http://127.0.0.1:8080/v1"

BASE_DIR = _get_base_dir().resolve()
CONTROLLER_FILE = (BASE_DIR / "talos_controller.py").resolve()
RESULTS_FILE = (BASE_DIR / "ea_results.tsv").resolve()
ATTEMPT_LOG_DIR = (BASE_DIR / "golden" / "ea_attempt_logs").resolve()

# Session state to enforce keep/discard behavior with rollback.
_SESSION = {
    "attempt": 0,
    "best_ate_m": None,
    "last_good_controller": None,
}

def is_path_safe(requested_path: str) -> bool:
    try:
        target = Path(requested_path).resolve()
        return target.is_relative_to(BASE_DIR)
    except Exception:
        return False


def _init_ledger() -> None:
    if RESULTS_FILE.exists():
        return
    header = "attempt\tbest_ate_m\tlatest_eskf_ate_m\tslap_rate_pct\tbest_round\tstatus\tnote\n"
    RESULTS_FILE.write_text(header, encoding='utf-8')


def _append_ledger_row(
    attempt: int,
    best_ate_m: float | None,
    latest_eskf_ate_m: float | None,
    slap_rate_pct: float | None,
    best_round: int | None,
    status: str,
    note: str,
) -> None:
    _init_ledger()
    row = (
        f"{attempt}\t"
        f"{'' if best_ate_m is None else f'{best_ate_m:.6f}'}\t"
        f"{'' if latest_eskf_ate_m is None else f'{latest_eskf_ate_m:.6f}'}\t"
        f"{'' if slap_rate_pct is None else f'{slap_rate_pct:.4f}'}\t"
        f"{'' if best_round is None else best_round}\t"
        f"{status}\t"
        f"{note.replace('\t', ' ').replace('\n', ' ').strip()}\n"
    )
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(row)


def _extract_slap_rate_pct(log_text: str) -> float | None:
    matches = re.findall(r"Slap_Rate:\s*([0-9]+(?:\.[0-9]+)?)%", log_text)
    if not matches:
        return None
    return float(matches[-1])


def _write_attempt_log(attempt: int, stdout_text: str, stderr_text: str, return_code: int) -> Path:
    ATTEMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ATTEMPT_LOG_DIR / f"attempt_{attempt:04d}.log"
    payload = [
        f"attempt={attempt}",
        f"return_code={return_code}",
        "--- STDOUT ---",
        stdout_text or "",
        "--- STDERR ---",
        stderr_text or "",
    ]
    log_path.write_text("\n".join(payload), encoding='utf-8')
    return log_path

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

    # AutoResearch-style mutation lock: only mutate the controller file.
    target = Path(filepath).resolve()
    if target != CONTROLLER_FILE:
        return f"Access Denied: Only {CONTROLLER_FILE} is mutable."

    try:
        if CONTROLLER_FILE.exists():
            _SESSION["last_good_controller"] = CONTROLLER_FILE.read_text(encoding='utf-8')
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


@tool
def run_scored_experiment(note: str = "") -> str:
    """Run one experiment, score it, and auto keep/discard with rollback.

    Rules:
    - First valid run establishes baseline and is kept.
    - Later runs are kept only if best_ate_m improves and slap_rate_pct <= 1.0 (if present).
    - Non-improving runs are automatically rolled back to last good controller.

    Args:
        note: Short description of mutation idea.

    Returns:
        JSON with attempt result and current best score.
    """
    import subprocess

    _SESSION["attempt"] += 1
    attempt = int(_SESSION["attempt"])

    try:
        result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE_DIR),
        )
        raw_log = result.stdout
        stderr_log = result.stderr or ""
        return_code = int(result.returncode)
    except subprocess.CalledProcessError as e:
        raw_log = f"Training crashed. Error log:\n{e.stderr}\nOutput log:\n{e.stdout}"
        stderr_log = e.stderr or ""
        return_code = int(e.returncode)

    attempt_log_path = _write_attempt_log(
        attempt=attempt,
        stdout_text=raw_log,
        stderr_text=stderr_log,
        return_code=return_code,
    )

    parsed = json.loads(parse_training_log(raw_log))
    slap_rate_pct = _extract_slap_rate_pct(raw_log)
    parsed["slap_rate_pct"] = slap_rate_pct

    best_ate_m = parsed.get("best_ate_m")
    ok = bool(parsed.get("ok")) and best_ate_m is not None
    crashed = bool(parsed.get("training_crashed"))

    current_best = _SESSION.get("best_ate_m")
    keep = False
    status = "discard"

    if ok and current_best is None:
        keep = True
        status = "baseline"
    elif ok:
        improved = bool(best_ate_m < current_best)
        slap_ok = bool(slap_rate_pct is None or slap_rate_pct <= 1.0)
        keep = improved and slap_ok
        status = "keep" if keep else "discard"
    else:
        status = "crash" if crashed else "invalid"

    if keep:
        _SESSION["best_ate_m"] = float(best_ate_m)
        _SESSION["last_good_controller"] = CONTROLLER_FILE.read_text(encoding='utf-8')
    else:
        last_good = _SESSION.get("last_good_controller")
        if last_good is not None:
            CONTROLLER_FILE.write_text(last_good, encoding='utf-8')

    _append_ledger_row(
        attempt=attempt,
        best_ate_m=best_ate_m,
        latest_eskf_ate_m=parsed.get("latest_eskf_ate_m"),
        slap_rate_pct=slap_rate_pct,
        best_round=parsed.get("best_round"),
        status=status,
        note=note,
    )

    return json.dumps(
        {
            "attempt": attempt,
            "status": status,
            "kept": keep,
            "best_ate_m": _SESSION.get("best_ate_m"),
            "run_best_ate_m": best_ate_m,
            "latest_eskf_ate_m": parsed.get("latest_eskf_ate_m"),
            "slap_rate_pct": slap_rate_pct,
            "training_crashed": crashed,
            "results_file": str(RESULTS_FILE),
            "attempt_log_file": str(attempt_log_path),
        },
        ensure_ascii=True,
    )


@tool
def get_ea_status() -> str:
    """Return AutoResearch session status and latest ledger entries as JSON."""
    _init_ledger()
    tail = []
    try:
        lines = RESULTS_FILE.read_text(encoding='utf-8').splitlines()
        tail = lines[-6:]
    except Exception:
        tail = []

    return json.dumps(
        {
            "attempt": _SESSION.get("attempt", 0),
            "best_ate_m": _SESSION.get("best_ate_m"),
            "results_file": str(RESULTS_FILE),
            "recent_rows": tail,
        },
        ensure_ascii=True,
    )

# 3. Read your custom system prompt
with open(BASE_DIR / "system.txt", "r", encoding="utf-8") as f:
    custom_instructions = f.read()

# 4. Initialize the model with the mandatory parameters
model = OpenAIModel(
    model_id="OmniClaw-V2-Q8_0",
    api_base=_get_llama_api_base(),
    api_key="sk-no-key-required"
)

# 5. Initialize the agent
agent = CodeAgent(
    tools=[
        read_safe,
        write_safe,
        run_training,
        parse_training_log,
        run_scored_experiment,
        get_ea_status,
    ],
    model=model,
    stream_outputs=True
)

# 6. Start the loop by injecting your rules
prompt = f"""
{custom_instructions}

YOUR MISSION:
1. Read `program.md` to understand your goals and constraints. 
2. Use `read_safe` to examine `talos_controller.py`.
3. For every attempt, after editing, call `run_scored_experiment(note=...)`.
4. Do NOT decide keep/discard yourself; trust `run_scored_experiment` status.
5. The controller is auto-rolled-back on discard/crash.
6. Use `get_ea_status` to track running best and recent results.
7. Iterate until you beat the 4.047 baseline.
"""

agent.run(prompt)