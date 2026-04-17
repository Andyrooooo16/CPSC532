#!/usr/bin/env python3
"""Export open-ended participant feedback to CSV.

The script scans session JSON files and extracts non-empty text responses from:
1) Per-paper questionnaire responses (e.g., p4comments)
2) Final questionnaire free-text responses (e.g., fq_like_most_least)

Output columns include session and participant identifiers plus paper/condition
context where available.

Run from repo root:
    python3 user-study/scripts/export_open_feedback.py

Optional custom output path:
    python3 user-study/scripts/export_open_feedback.py --output path/to/file.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any


ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = ROOT / "user-study" / "data" / "sessions"
QUESTIONS_PATH = ROOT / "user-study" / "questions.json"
OUT_DEFAULT = ROOT / "user-study" / "data" / "marked sessions" / "open_text_feedback.csv"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def get_question_prompt_map(questions: Dict[str, Any]) -> Dict[str, str]:
    prompt_by_id: Dict[str, str] = {}

    for paper_data in (questions.get("papers") or {}).values():
        for q in ((paper_data.get("questionnaire") or {}).get("questions") or []):
            qid = q.get("id")
            if qid:
                prompt_by_id[qid] = q.get("text", "")

    for q in ((questions.get("finalQuestionnaire") or {}).get("questions") or []):
        qid = q.get("id")
        if qid:
            prompt_by_id[qid] = q.get("text", "")

    return prompt_by_id


def get_paper_title_map(questions: Dict[str, Any]) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for paper_key, paper_data in (questions.get("papers") or {}).items():
        titles[paper_key] = paper_data.get("title", "")
    return titles


def build_condition_map(session: Dict[str, Any]) -> Dict[str, str]:
    condition_by_paper: Dict[str, str] = {}

    for task in (session.get("tasks") or []):
        paper_key = task.get("paperKey")
        condition = task.get("condition")
        if paper_key and condition:
            condition_by_paper[paper_key] = condition

    paper_order = session.get("paperOrder") or []
    conditions = session.get("conditions") or []
    for idx, paper_key in enumerate(paper_order):
        if paper_key and paper_key not in condition_by_paper and idx < len(conditions):
            condition_by_paper[paper_key] = conditions[idx]

    return condition_by_paper


def extract_rows(
    session: Dict[str, Any],
    session_path: Path,
    prompt_by_id: Dict[str, str],
    title_by_paper: Dict[str, str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    session_id = str(session.get("sessionId") or session.get("sessionid") or session_path.stem)
    participant_id = str(session.get("participantId") or session.get("participantid") or "")
    session_created_at = str(session.get("createdAt") or "")
    session_completed_at = str(session.get("completedAt") or "")
    condition_by_paper = build_condition_map(session)

    # Per-paper questionnaire feedback.
    for entry in (session.get("questionnaireResponses") or []):
        if not isinstance(entry, dict):
            continue
        paper_key = str(entry.get("paperKey") or "")
        responses = entry.get("responses") or {}
        submitted_at = str(entry.get("submittedAt") or "")
        if not isinstance(responses, dict):
            continue

        for question_id, value in responses.items():
            if not is_non_empty_text(value):
                continue

            rows.append(
                {
                    "sessionId": session_id,
                    "participantId": participant_id,
                    "feedbackScope": "paper",
                    "source": "paper_questionnaire",
                    "paperKey": paper_key,
                    "paperTitle": title_by_paper.get(paper_key, ""),
                    "highlightCondition": condition_by_paper.get(paper_key, ""),
                    "questionId": str(question_id),
                    "questionPrompt": prompt_by_id.get(str(question_id), ""),
                    "responseText": value.strip(),
                    "submittedAt": submitted_at,
                    "sessionCreatedAt": session_created_at,
                    "sessionCompletedAt": session_completed_at,
                    "sessionFile": session_path.name,
                }
            )

    # Final questionnaire feedback.
    final_responses = session.get("finalResponses") or {}
    if isinstance(final_responses, dict):
        for question_id, value in final_responses.items():
            if not is_non_empty_text(value):
                continue

            rows.append(
                {
                    "sessionId": session_id,
                    "participantId": participant_id,
                    "feedbackScope": "final",
                    "source": "final_questionnaire",
                    "paperKey": "",
                    "paperTitle": "",
                    "highlightCondition": "",
                    "questionId": str(question_id),
                    "questionPrompt": prompt_by_id.get(str(question_id), ""),
                    "responseText": value.strip(),
                    "submittedAt": "",
                    "sessionCreatedAt": session_created_at,
                    "sessionCompletedAt": session_completed_at,
                    "sessionFile": session_path.name,
                }
            )

    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sessionId",
        "participantId",
        "feedbackScope",
        "source",
        "paperKey",
        "paperTitle",
        "highlightCondition",
        "questionId",
        "questionPrompt",
        "responseText",
        "submittedAt",
        "sessionCreatedAt",
        "sessionCompletedAt",
        "sessionFile",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export open text feedback from user-study sessions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DEFAULT,
        help=f"Output CSV path (default: {OUT_DEFAULT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"Missing required file: {QUESTIONS_PATH}")
    if not SESSIONS_DIR.exists():
        raise SystemExit(f"Missing required directory: {SESSIONS_DIR}")

    questions = load_json(QUESTIONS_PATH)
    prompt_by_id = get_question_prompt_map(questions)
    title_by_paper = get_paper_title_map(questions)

    rows: List[Dict[str, str]] = []
    for session_path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            session = load_json(session_path)
        except Exception as exc:
            print(f"WARNING: skipping {session_path.name} (parse error: {exc})")
            continue
        rows.extend(extract_rows(session, session_path, prompt_by_id, title_by_paper))

    rows.sort(
        key=lambda r: (
            r["participantId"],
            r["sessionId"],
            r["feedbackScope"],
            r["paperKey"],
            r["questionId"],
        )
    )
    write_csv(rows, args.output)

    print(f"Wrote {len(rows)} open-text feedback rows to {args.output}")


if __name__ == "__main__":
    main()
