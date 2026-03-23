import os
from typing import Iterable

import requests


def _split_targets(raw: str) -> set[str]:
    return {x.strip().lower() for x in raw.split(',') if x.strip()}


def _format_status(best_ate: float, best_round: int, total_rounds: int, run_name: str) -> str:
    if best_round > 0 and best_ate == best_ate:
        return (
            f"TALOS done [{run_name}]. Best ATE: {best_ate:.3f}m "
            f"@ Round {best_round}/{total_rounds}"
        )
    return f"TALOS done [{run_name}]. No valid physical checkpoint."


def send_ntfy(message: str) -> bool:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False

    base_url = os.environ.get("NTFY_BASE_URL", "https://ntfy.sh").rstrip('/')
    url = f"{base_url}/{topic}"
    headers = {
        "Title": os.environ.get("NTFY_TITLE", "TALOS Training"),
        "Tags": os.environ.get("NTFY_TAGS", "robot,chart_with_upwards_trend"),
        "Priority": os.environ.get("NTFY_PRIORITY", "3"),
    }
    token = os.environ.get("NTFY_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
        ok = r.status_code < 400
        print(f"[reporting] ntfy {'ok' if ok else 'failed'} ({r.status_code})")
        return ok
    except Exception as e:
        print(f"[reporting] ntfy error: {e}")
        return False


def send_notion(best_ate: float, best_round: int, total_rounds: int) -> bool:
    try:
        from notion_logger import log_run
    except Exception as e:
        print(f"[reporting] notion import error: {e}")
        return False

    if not (best_round > 0 and best_ate == best_ate):
        print("[reporting] notion skipped (no valid physical checkpoint)")
        return False

    try:
        return bool(log_run(float(best_ate), int(best_round), int(total_rounds)))
    except Exception as e:
        print(f"[reporting] notion error: {e}")
        return False


def publish_training_summary(best_ate: float, best_round: int, total_rounds: int, run_name: str) -> None:
    """Publish end-of-run status to configured channels.

    Env controls:
    - REPORT_TARGETS=ntfy,notion (default: ntfy)
    - NTFY_TOPIC=talos-aman-lab (required for ntfy)
    - NTFY_BASE_URL=https://ntfy.sh
    - NTFY_TITLE, NTFY_TAGS, NTFY_PRIORITY, NTFY_TOKEN (optional)
    """
    targets = _split_targets(os.environ.get("REPORT_TARGETS", "ntfy"))
    if not targets:
        print("[reporting] no report targets configured")
        return

    status = _format_status(best_ate, best_round, total_rounds, run_name)

    if 'ntfy' in targets:
        send_ntfy(status)

    if 'notion' in targets:
        send_notion(best_ate, best_round, total_rounds)
