import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


CONFIG_PATH = Path("idp_uploader_config.json")
LIVE_CSV_ENDPOINT = "/api/live-csv/sync"
ATTENDANCE_ENDPOINT = "/api/attendance/events"

TARGET_FIELDS = ["timestamp", "student_name", "class_name", "status", "device_id", "note"]

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "created_at", "attendance_time"],
    "student_name": ["student_name", "name", "student", "student_id", "id", "label", "person"],
    "class_name": ["class_name", "class", "course", "course_code", "subject"],
    "status": ["status", "attendance_status", "result"],
    "device_id": ["device_id", "device", "camera_id"],
    "note": ["note", "notes", "remark", "remarks", "message"],
}


def now_timestamp():
    return datetime.now().isoformat(timespec="seconds")


def load_config(path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_state(path):
    if not path.exists():
        return {"uploaded_row_count": 0, "source_size": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"uploaded_row_count": 0, "source_size": 0}


def save_state(path, uploaded_row_count, source_size):
    save_json(path, {"uploaded_row_count": uploaded_row_count, "source_size": source_size})


def ensure_csv(path, fields):
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()


def append_unsent_rows(path, rows):
    ensure_csv(path, TARGET_FIELDS)
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TARGET_FIELDS)
        writer.writerows(rows)


def read_csv_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rewrite_unsent_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def get_first(row, aliases):
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for alias in aliases:
        value = lowered.get(alias.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_row(row, config):
    date_value = get_first(row, ["date", "attendance_date"])
    time_value = get_first(row, ["time", "clock_time", "time_in"])
    timestamp = ""
    if date_value and time_value:
        timestamp = f"{date_value} {time_value}".strip()
    if not timestamp:
        timestamp = get_first(row, COLUMN_ALIASES["timestamp"])
    if not timestamp:
        timestamp = now_timestamp()

    matrix_number = get_first(row, ["matrix_number", "matrix number", "student_id", "id"])
    course_code = get_first(row, ["course_code", "course code", "class_name", "class", "course", "subject"])
    course_name = get_first(row, ["course_name", "course name"])
    class_name = course_code or course_name or config.get("default_class_name", "")
    note = get_first(row, COLUMN_ALIASES["note"])
    note_parts = []
    if matrix_number:
        note_parts.append(f"matrix_number={matrix_number}")
    if course_name and course_code:
        note_parts.append(f"course_name={course_name}")
    if note:
        note_parts.append(note)

    return {
        "timestamp": timestamp,
        "student_name": get_first(row, COLUMN_ALIASES["student_name"]),
        "class_name": class_name,
        "status": get_first(row, COLUMN_ALIASES["status"]) or config.get("default_status", "detected"),
        "device_id": get_first(row, COLUMN_ALIASES["device_id"]) or config.get("device_id", "raspberry-pi-01"),
        "note": "; ".join(note_parts),
    }


def source_was_recreated(csv_path, state):
    if not csv_path.exists():
        return False
    previous_size = int(state.get("source_size", 0))
    return csv_path.stat().st_size < previous_size


def iter_source_csv_paths(config):
    csv_dir = config.get("local_csv_dir")
    if csv_dir:
        base_dir = Path(csv_dir)
        pattern = config.get("csv_glob", "*.csv")
        if not base_dir.exists():
            return []
        return sorted(path for path in base_dir.glob(pattern) if path.is_file())
    return [Path(config["local_csv_path"])]


def _state_for_source(state, csv_path):
    sources = state.setdefault("sources", {})
    return sources.setdefault(
        str(csv_path.resolve()),
        {"uploaded_row_count": 0, "source_size": 0},
    )


def collect_new_rows(config):
    state_path = Path(config.get("state_path", "idp_uploader_state.json"))
    state = load_state(state_path)
    source_paths = iter_source_csv_paths(config)
    all_normalized = []

    # Backward compatibility for the old single-file state format.
    if "sources" not in state and "uploaded_row_count" in state and len(source_paths) == 1:
        state = {
            "sources": {
                str(source_paths[0].resolve()): {
                    "uploaded_row_count": int(state.get("uploaded_row_count", 0)),
                    "source_size": int(state.get("source_size", 0)),
                }
            }
        }

    for csv_path in source_paths:
        source_state = _state_for_source(state, csv_path)
        all_rows = read_csv_rows(csv_path)
        if source_was_recreated(csv_path, source_state):
            source_state["uploaded_row_count"] = 0

        uploaded_count = int(source_state.get("uploaded_row_count", 0))
        new_raw_rows = all_rows[uploaded_count:]
        normalized = [normalize_row(row, config) for row in new_raw_rows]
        normalized = [row for row in normalized if row["student_name"]]
        all_normalized.extend(normalized)
        source_state["uploaded_row_count"] = len(all_rows)
        source_state["source_size"] = csv_path.stat().st_size if csv_path.exists() else 0

    if all_normalized:
        unsent_path = Path(config.get("unsent_csv_path", "raspberry_pi_unsent.csv"))
        append_unsent_rows(unsent_path, all_normalized)

    save_json(state_path, state)
    return len(all_normalized)


def post_json(url, payload, timeout=8):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")


def upload_live_csv(config, rows):
    url = config["server_url"].rstrip("/") + LIVE_CSV_ENDPOINT
    payload = {"api_key": config["api_key"], "rows": rows}
    return post_json(url, payload)


def upload_attendance_events(config, rows):
    url = config["server_url"].rstrip("/") + ATTENDANCE_ENDPOINT
    for row in rows:
        payload = {
            "student_name": row["student_name"],
            "class_name": row["class_name"],
            "device_id": row["device_id"],
            "source": "raspberry_pi_csv",
            "created_at": row["timestamp"],
            "api_key": config["api_key"],
        }
        post_json(url, payload)
    return 200, f"Uploaded {len(rows)} attendance events"


def sync_unsent_rows(config):
    unsent_path = Path(config.get("unsent_csv_path", "raspberry_pi_unsent.csv"))
    rows = read_csv_rows(unsent_path)
    if not rows:
        return 0

    batch_size = int(config.get("batch_size", 50))
    batch = rows[:batch_size]
    mode = config.get("mode", "live_csv")

    if mode == "attendance_events":
        status, body = upload_attendance_events(config, batch)
    else:
        status, body = upload_live_csv(config, batch)

    if status == 200:
        rewrite_unsent_rows(unsent_path, rows[batch_size:])
        print(f"Uploaded {len(batch)} rows. Server response: {body}")
        return len(batch)

    raise RuntimeError(f"Server returned HTTP {status}: {body}")


def run_once(config):
    added = collect_new_rows(config)
    if added:
        print(f"Queued {added} new CSV rows.")
    uploaded = sync_unsent_rows(config)
    if not added and not uploaded:
        print("No new rows to upload.")


def main():
    parser = argparse.ArgumentParser(description="Upload Raspberry Pi detection CSV rows to the IDP attendance server.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to uploader config JSON.")
    parser.add_argument("--once", action="store_true", help="Run one scan/upload cycle and exit.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    print("IDP Raspberry Pi CSV uploader")
    print(f"Server: {config['server_url']}")
    if config.get("local_csv_dir"):
        print(f"CSV folder: {config['local_csv_dir']} ({config.get('csv_glob', '*.csv')})")
    else:
        print(f"CSV: {config['local_csv_path']}")
    print(f"Mode: {config.get('mode', 'live_csv')}")

    while True:
        try:
            run_once(config)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"Computer server not reachable. Rows are kept for retry: {error}")
        except Exception as error:
            print(f"Upload cycle failed. Rows are kept for retry: {error}")

        if args.once:
            break
        time.sleep(int(config.get("poll_seconds", 5)))


if __name__ == "__main__":
    main()
