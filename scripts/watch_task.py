"""Tail ClearML task log + status. Usage:
    python scripts/watch_task.py <task_id> [--n 200] [--filter pip,torch,quilt,error] [--tail 60]
"""
import argparse
import os
import sys

os.environ.setdefault(
    "CLEARML_CONFIG_FILE", os.path.abspath("clearml.conf")
)

# Force UTF-8 stdout to avoid cp1251 UnicodeEncodeError on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from clearml import Task
from clearml.backend_api.session.client import APIClient


def fetch_log_tail(api: APIClient, task_id: str, n: int = 200):
    ev = api.events.get_task_log(task=task_id, order="desc", batch_size=n)
    lines = ev.events if hasattr(ev, "events") else (
        ev.get("events") if isinstance(ev, dict) else []
    )
    return list(reversed(lines))  # oldest-first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("--n", type=int, default=200, help="fetch this many recent events")
    ap.add_argument("--tail", type=int, default=0, help="after filter, keep last K lines (0=all)")
    ap.add_argument("--filter", default="")
    args = ap.parse_args()

    needles = [s.strip().lower() for s in args.filter.split(",") if s.strip()]
    t = Task.get_task(task_id=args.task_id)
    api = APIClient()

    print(f"status     : {t.get_status()}")
    print(f"status_msg : {t.data.status_message}")
    print(f"url        : {t.get_output_log_web_page()}")
    print("-" * 60)
    lines = fetch_log_tail(api, args.task_id, args.n)
    filtered = []
    for e in lines:
        msg = e.get("msg") if isinstance(e, dict) else getattr(e, "msg", "")
        if not msg:
            continue
        if needles and not any(n in msg.lower() for n in needles):
            continue
        if len(msg) > 800:
            msg = msg[:800] + "..."
        filtered.append(msg)

    if args.tail > 0:
        filtered = filtered[-args.tail:]

    for m in filtered:
        # safe print: replace unencodable chars
        try:
            print(m)
        except UnicodeEncodeError:
            print(m.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
