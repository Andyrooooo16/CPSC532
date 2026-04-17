"""
flatten_sessions.py

Reads raw session JSON files and the marked short-answer CSV and produces
four flattened CSVs for analysis:

 - user-study/data/marked sessions/paper_level.csv
 - user-study/data/marked sessions/question_level.csv
 - user-study/data/marked sessions/cross_level.csv
 - user-study/data/marked sessions/final_survey.csv

Expected inputs (relative to repo root):
 - user-study/data/sessions/*.json        (raw session files)
 - user-study/questions.json              (question definitions)
 - user-study/study-config.json           (paper metadata & participants)
 - user-study/data/marked sessions/CPSC Short Answer Marked - Sheet1.csv

Run from repo root as:
    python user-study/scripts/flatten_sessions.py

Notes:
 - Defensive about missing fields; prints warnings and continues.
 - Uses the marked short-answer CSV as the source of truth for free-text scoring.
 - Never marks empty free-text as correct by default.
"""

from pathlib import Path
import json
import datetime
from collections import defaultdict
import pandas as pd
import sys


ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = ROOT / 'user-study' / 'data' / 'sessions'
MARKED_DIR = ROOT / 'user-study' / 'data' / 'marked sessions'
MARKED_CSV = MARKED_DIR / 'CPSC Short Answer Marked - Sheet1.csv'
QUESTIONS_PATH = ROOT / 'user-study' / 'questions.json'
STUDY_CONFIG = ROOT / 'user-study' / 'study-config.json'

OUT_PAPER = MARKED_DIR / 'paper_level.csv'
OUT_QUESTION = MARKED_DIR / 'question_level.csv'
OUT_CROSS = MARKED_DIR / 'cross_level.csv'
OUT_FINAL = MARKED_DIR / 'final_survey.csv'


def warn(msg):
    print(f'WARNING: {msg}', file=sys.stderr)


def parse_iso(ts):
    if not ts:
        return None
    if isinstance(ts, datetime.datetime):
        return ts
    s = str(ts)
    # Accept several ISO formats, tolerant to trailing Z
    fmts = [
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
    ]
    for f in fmts:
        try:
            return datetime.datetime.strptime(s, f)
        except Exception:
            continue
    try:
        # last resort: strip Z and try fromisoformat
        return datetime.datetime.fromisoformat(s.replace('Z', ''))
    except Exception:
        warn(f'Failed to parse timestamp: {ts}')
        return None


def load_questions():
    try:
        return json.loads(QUESTIONS_PATH.read_text())
    except Exception as e:
        print('Failed to load questions.json:', e)
        raise


def load_config():
    try:
        return json.loads(STUDY_CONFIG.read_text())
    except Exception as e:
        print('Failed to load study-config.json:', e)
        raise


def load_marked():
    if not MARKED_CSV.exists():
        warn(f'Marked CSV not found at {MARKED_CSV}. Continuing without manual short answers.')
        return pd.DataFrame(columns=['sessionid', 'participantId', 'paper', 'questionId', 'answer', 'isCorrect', 'rationale'])
    df = pd.read_csv(MARKED_CSV, dtype=str)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if 'sessionid' not in [c.lower() for c in df.columns]:
        warn('Marked CSV is missing required column `sessionid` (lowercase).')
    # ensure lowercase sessionid column present
    cols = {c.lower(): c for c in df.columns}
    if 'sessionid' in cols and cols['sessionid'] != 'sessionid':
        df = df.rename(columns={cols['sessionid']: 'sessionid'})
    return df


def find_marked_row(marked_df, sessionId, participantId, paperKey, questionId):
    if marked_df is None or marked_df.empty:
        return None
    # Exact match first
    m = marked_df[
        (marked_df['sessionid'].astype(str).str.strip() == str(sessionId).strip()) &
        (marked_df['participantId'].astype(str).str.strip() == str(participantId).strip()) &
        (marked_df['paper'].astype(str).str.strip() == str(paperKey).strip()) &
        (marked_df['questionId'].astype(str).str.strip() == str(questionId).strip())
    ]
    if not m.empty:
        return m.iloc[0]
    # Try without a leading dash (some exports include a leading '-')
    alt = str(sessionId).lstrip('-')
    m2 = marked_df[
        (marked_df['sessionid'].astype(str).str.strip().str.lstrip('-') == alt) &
        (marked_df['participantId'].astype(str).str.strip() == str(participantId).strip()) &
        (marked_df['paper'].astype(str).str.strip() == str(paperKey).strip()) &
        (marked_df['questionId'].astype(str).str.strip() == str(questionId).strip())
    ]
    if not m2.empty:
        return m2.iloc[0]
    return None


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default


def main():
    questions = load_questions()
    config = load_config()
    marked_df = load_marked()

    session_files = sorted([p for p in SESSIONS_DIR.glob('*.json')])
    if not session_files:
        print('No session files found in', SESSIONS_DIR)
        return

    paper_rows = []
    question_rows = []
    cross_rows = []
    final_rows = []

    missing_manual = 0
    total_sessions = 0

    for p in session_files:
        total_sessions += 1
        try:
            sess = json.loads(p.read_text())
        except Exception as e:
            warn(f'Failed to parse session file {p.name}: {e}')
            continue

        sessionId = sess.get('sessionId') or sess.get('sessionid') or p.stem
        participantId = sess.get('participantId') or sess.get('participantid') or ''

        paperOrder = sess.get('paperOrder') or config.get('participants', {}).get(participantId, {}).get('paperOrder') or config.get('defaultParticipant', {}).get('paperOrder', [])
        conditions = sess.get('conditions') or config.get('participants', {}).get(participantId, {}).get('conditions') or config.get('defaultParticipant', {}).get('conditions', [])

        if len(paperOrder) != 3:
            warn(f'session {sessionId} expected 3 papers in paperOrder but found {len(paperOrder)}')

        # find contextual paper (first occurrence of contextual_highlights)
        contextual_paper = None
        for idx, cond in enumerate(conditions):
            if cond == 'contextual_highlights' and idx < len(paperOrder):
                contextual_paper = paperOrder[idx]
                break

        # Build a map of tasks by paperKey
        tasks_map = {t.get('paperKey'): t for t in sess.get('tasks', []) if t.get('paperKey')}

        # PAPER-LEVEL rows
        paper_level_scores = []
        for i, paperKey in enumerate(paperOrder):
            orderPosition = i + 1
            condition = conditions[i] if i < len(conditions) else None
            paper_meta = config.get('papers', {}).get(paperKey, {})
            paperTitle = paper_meta.get('title') if paper_meta else None

            task = tasks_map.get(paperKey, {})
            startedAt = parse_iso(task.get('startedAt'))
            completedAt = parse_iso(task.get('completedAt'))
            duration = None
            if startedAt and completedAt:
                try:
                    duration = int((completedAt - startedAt).total_seconds())
                except Exception:
                    duration = None

            # gather MC questions for this paper
            paper_questions = questions.get('papers', {}).get(paperKey, {}).get('questions', [])
            mc_qs = [q for q in paper_questions if q.get('type') == 'multiple_choice']
            ft_qs = [q for q in paper_questions if q.get('type') == 'free_text']
            mc_total_questions = len(mc_qs)

            attempts = task.get('attempts', []) if task else []
            mc_first_attempt_correct_count = 0
            mc_total_submissions = len(attempts)

            # compute first attempt correctness using attempt scores when available
            if attempts and isinstance(attempts, list):
                first = attempts[0]
                scores0 = first.get('scores', {})
                for q in mc_qs:
                    qid = q.get('id')
                    if qid and scores0.get(qid) is True:
                        mc_first_attempt_correct_count += 1
            else:
                # try to fallback to comparing first answers against key
                pass

            # compute wrong attempts before correct (sum across MC questions)
            mc_total_wrong_before_correct = 0
            for q in mc_qs:
                qid = q.get('id')
                wrongs = 0
                found_correct = False
                for att in attempts:
                    scores = att.get('scores', {})
                    if qid in scores:
                        if scores.get(qid):
                            found_correct = True
                            break
                        else:
                            wrongs += 1
                    else:
                        # if scores missing, infer by comparing answers to key
                        ans = att.get('answers', {}).get(qid)
                        if ans is not None and q.get('correct') is not None:
                            if str(ans) == str(q.get('correct')):
                                found_correct = True
                                break
                            else:
                                wrongs += 1
                mc_total_wrong_before_correct += wrongs

            mc_first_attempt_accuracy = (mc_first_attempt_correct_count / mc_total_questions) if mc_total_questions else None

            # FREE TEXT scoring using marked CSV
            free_text_possible = len(ft_qs)
            free_text_scored_count = 0
            free_text_score = 0
            for fq in ft_qs:
                qid = fq.get('id')
                finalAnswers = task.get('finalAnswers') or {}
                answer_text = finalAnswers.get(qid) if finalAnswers else None

                marked_row = find_marked_row(marked_df, sessionId, participantId, paperKey, qid)
                if marked_row is None:
                    if answer_text in (None, '', ''):
                        manual = 0
                        scoring_status = 'empty_answer'
                        # per requirement: empty answer must not be auto-marked correct
                    else:
                        manual = None
                        scoring_status = 'missing_manual_score'
                        missing_manual += 1
                else:
                    free_text_scored_count += 1
                    try:
                        manual = int(float(marked_row.get('isCorrect', 0)))
                    except Exception:
                        manual = 1 if str(marked_row.get('isCorrect')).strip() in ('1', 'True', 'true') else 0
                    scoring_status = 'scored'

                if manual:
                    free_text_score += int(manual)

            paper_total_score = (mc_first_attempt_correct_count or 0) + free_text_score

            # Difficulty and questionnaire
            questionnaire_list = sess.get('questionnaireResponses', [])
            # find the questionnaire submission for this paper
            q_resp = None
            for qr in questionnaire_list:
                if qr.get('paperKey') == paperKey:
                    q_resp = qr.get('responses', {})
                    break

            # compute difficulty averages
            diff_vals = []
            diff_mc_vals = []
            diff_ft_vals = []
            ease_rating = None
            paper_comments = ''
            if q_resp:
                for k, v in q_resp.items():
                    if k.startswith(paperKey.replace('paper', 'p') + 'diff') or k.startswith('p' + paperKey.replace('paper', '') + 'diff'):
                        try:
                            vi = float(v)
                            diff_vals.append(vi)
                            # map which question it refers to by number
                            # p4diff1 -> p4q1
                            # derive qid
                            num = ''.join([c for c in k if c.isdigit()])
                            if num:
                                mapped_qid = f'p{paperKey.replace("paper","") }q{num}'
                                # find type
                                qtype = None
                                for qq in paper_questions:
                                    if qq.get('id') == mapped_qid:
                                        qtype = qq.get('type')
                                        break
                                if qtype == 'multiple_choice':
                                    diff_mc_vals.append(vi)
                                elif qtype == 'free_text':
                                    diff_ft_vals.append(vi)
                        except Exception:
                            continue
                    if k.endswith('ease'):
                        try:
                            ease_rating = float(v)
                        except Exception:
                            pass
                    if k.endswith('comments'):
                        paper_comments = v if v is not None else ''

            avg_question_difficulty_all = (sum(diff_vals) / len(diff_vals)) if diff_vals else None
            avg_question_difficulty_mc = (sum(diff_mc_vals) / len(diff_mc_vals)) if diff_mc_vals else None
            avg_question_difficulty_free_text = (sum(diff_ft_vals) / len(diff_ft_vals)) if diff_ft_vals else None

            paper_rows.append({
                'sessionId': sessionId,
                'participantId': participantId,
                'paperKey': paperKey,
                'paperTitle': paperTitle,
                'condition': condition,
                'orderPosition': orderPosition,
                'startedAt': startedAt.isoformat() if startedAt else None,
                'completedAt': completedAt.isoformat() if completedAt else None,
                'paper_duration_seconds': duration,
                'mc_total_questions': mc_total_questions,
                'mc_first_attempt_correct_count': mc_first_attempt_correct_count,
                'mc_first_attempt_accuracy': mc_first_attempt_accuracy,
                'mc_total_submissions': mc_total_submissions,
                'mc_total_wrong_before_correct': mc_total_wrong_before_correct,
                'free_text_possible': free_text_possible,
                'free_text_scored_count': free_text_scored_count,
                'free_text_score': free_text_score,
                'paper_total_score': paper_total_score,
                'avg_question_difficulty_all': avg_question_difficulty_all,
                'avg_question_difficulty_mc': avg_question_difficulty_mc,
                'avg_question_difficulty_free_text': avg_question_difficulty_free_text,
                'ease_rating': ease_rating,
                'paper_comments': paper_comments,
            })

            # QUESTION-LEVEL rows for paper questions
            # determine presented order from session.questionOrders if available
            q_order = sess.get('questionOrders', {}).get(paperKey, [q.get('id') for q in paper_questions])

            for idx_q, q in enumerate(paper_questions):
                qid = q.get('id')
                qtype = q.get('type')
                qtext = q.get('text')

                # first attempt and final answers
                firstAttemptAnswer = None
                firstAttemptCorrect = None
                finalAnswer = None
                finalCorrect = None
                wrongAttemptsBeforeCorrect = None

                if attempts:
                    first = attempts[0]
                    firstAttemptAnswer = first.get('answers', {}).get(qid)
                    firstAttemptCorrect = None
                    if first.get('scores') and qid in first.get('scores'):
                        firstAttemptCorrect = bool(first.get('scores', {}).get(qid))

                finalAnswers = task.get('finalAnswers') or {}
                finalAnswer = finalAnswers.get(qid)
                # final correctness: use question key when possible
                if q.get('type') == 'multiple_choice' and q.get('correct') is not None:
                    try:
                        finalCorrect = (str(finalAnswer) == str(q.get('correct')))
                    except Exception:
                        finalCorrect = None

                # wrong attempts before correct per question
                if q.get('type') == 'multiple_choice':
                    wrongs = 0
                    for att in attempts:
                        scores = att.get('scores', {})
                        if qid in scores:
                            if scores.get(qid):
                                break
                            else:
                                wrongs += 1
                        else:
                            ans = att.get('answers', {}).get(qid)
                            if ans is not None and q.get('correct') is not None:
                                if str(ans) == str(q.get('correct')):
                                    break
                                else:
                                    wrongs += 1
                    wrongAttemptsBeforeCorrect = wrongs

                # manual short answer mapping
                manualShortAnswerCorrect = None
                manualShortAnswerRationale = ''
                scoring_status = ''
                if qtype == 'free_text':
                    marked_row = find_marked_row(marked_df, sessionId, participantId, paperKey, qid)
                    finalAns = finalAnswer
                    if marked_row is None:
                        if finalAns in (None, ''):
                            manualShortAnswerCorrect = 0
                            scoring_status = 'empty_answer'
                        else:
                            manualShortAnswerCorrect = None
                            scoring_status = 'missing_manual_score'
                            missing_manual += 1
                    else:
                        try:
                            manualShortAnswerCorrect = int(float(marked_row.get('isCorrect', 0)))
                        except Exception:
                            manualShortAnswerCorrect = 1 if str(marked_row.get('isCorrect')).strip() in ('1', 'True', 'true') else 0
                        manualShortAnswerRationale = str(marked_row.get('rationale', '') or '')
                        scoring_status = 'scored'

                question_rows.append({
                    'sessionId': sessionId,
                    'participantId': participantId,
                    'scope': 'paper',
                    'paperKey': paperKey,
                    'paperTitle': paperTitle,
                    'condition': condition,
                    'orderPosition': orderPosition,
                    'questionId': qid,
                    'questionText': qtext,
                    'questionType': qtype,
                    'finalAnswer': finalAnswer,
                    'firstAttemptAnswer': firstAttemptAnswer,
                    'firstAttemptCorrect': firstAttemptCorrect,
                    'finalCorrect': finalCorrect,
                    'wrongAttemptsBeforeCorrect': wrongAttemptsBeforeCorrect,
                    'manualShortAnswerCorrect': manualShortAnswerCorrect,
                    'manualShortAnswerRationale': manualShortAnswerRationale,
                    'scoring_status': scoring_status,
                    'difficulty_rating': None,
                    'questionPresentedOrder': (q_order.index(qid) + 1) if qid in q_order else None,
                })

        # CROSS-LEVEL aggregation (one row per participant)
        crossQ = questions.get('crossQuestionnaire', {}).get('questions', [])
        cross_mc_qs = [q for q in crossQ if q.get('type') == 'multiple_choice']
        cross_ft_qs = [q for q in crossQ if q.get('type') == 'free_text']

        # collect attempts summary
        cross_attempts = sess.get('crossAttempts', []) or []
        cross_mc_first_attempt_correct_count = 0
        cross_mc_total_wrong_before_correct = 0
        cross_mc_total_questions = len(cross_mc_qs)

        # first attempt correctness (cross attempts: look at first attempt if present)
        if cross_attempts:
            first = cross_attempts[0]
            scores0 = first.get('scores', {})
            for q in cross_mc_qs:
                if scores0.get(q.get('id')):
                    cross_mc_first_attempt_correct_count += 1

        # wrong before correct across cross attempts
        for q in cross_mc_qs:
            qid = q.get('id')
            wrongs = 0
            for att in cross_attempts:
                scr = att.get('scores', {})
                if qid in scr:
                    if scr.get(qid):
                        break
                    else:
                        wrongs += 1
            cross_mc_total_wrong_before_correct += wrongs

        cross_mc_first_attempt_accuracy = (cross_mc_first_attempt_correct_count / cross_mc_total_questions) if cross_mc_total_questions else None

        # cross free-text scoring via marked CSV
        cross_free_text_possible = len(cross_ft_qs)
        cross_free_text_scored_count = 0
        cross_free_text_score = 0
        for q in cross_ft_qs:
            qid = q.get('id')
            # find marked row: paper column in marked CSV uses 'cross_paper'
            marked_row = find_marked_row(marked_df, sessionId, participantId, 'cross_paper', qid)
            if marked_row is None:
                # see instructions: missing_manual_score
                pass
            else:
                cross_free_text_scored_count += 1
                try:
                    mc = int(float(marked_row.get('isCorrect', 0)))
                except Exception:
                    mc = 1 if str(marked_row.get('isCorrect')).strip() in ('1', 'True', 'true') else 0
                cross_free_text_score += mc

        cross_total_score = (cross_mc_first_attempt_correct_count or 0) + cross_free_text_score

        cross_rows.append({
            'sessionId': sessionId,
            'participantId': participantId,
            'first_paper': paperOrder[0] if len(paperOrder) > 0 else None,
            'second_paper': paperOrder[1] if len(paperOrder) > 1 else None,
            'third_paper': paperOrder[2] if len(paperOrder) > 2 else None,
            'condition_first': conditions[0] if len(conditions) > 0 else None,
            'condition_second': conditions[1] if len(conditions) > 1 else None,
            'condition_third': conditions[2] if len(conditions) > 2 else None,
            'contextual_paper': contextual_paper,
            'cross_mc_total_questions': cross_mc_total_questions,
            'cross_mc_first_attempt_correct_count': cross_mc_first_attempt_correct_count,
            'cross_mc_first_attempt_accuracy': cross_mc_first_attempt_accuracy,
            'cross_mc_total_wrong_before_correct': cross_mc_total_wrong_before_correct,
            'cross_free_text_possible': cross_free_text_possible,
            'cross_free_text_scored_count': cross_free_text_scored_count,
            'cross_free_text_score': cross_free_text_score,
            'cross_total_score': cross_total_score,
        })

        # FINAL survey row
        demographics = sess.get('demographicResponses') or {}
        final_responses = sess.get('finalResponses') or {}

        # flatten final questionnaire into columns (questions.json defines finalQuestionnaire)
        final_fields = {}
        fq = questions.get('finalQuestionnaire', {}).get('questions', [])
        for q in fq:
            qid = q.get('id')
            final_fields[qid] = final_responses.get(qid)

        # compute per participant paper summary means
        p_rows_for_participant = [r for r in paper_rows if r['sessionId'] == sessionId]
        mean_paper_total_score = None
        mean_paper_free_text_score = None
        if p_rows_for_participant:
            try:
                mean_paper_total_score = sum(r.get('paper_total_score', 0) for r in p_rows_for_participant) / len(p_rows_for_participant)
                mean_paper_free_text_score = sum(r.get('free_text_score', 0) for r in p_rows_for_participant) / len(p_rows_for_participant)
            except Exception:
                pass

        final_rows.append({
            'sessionId': sessionId,
            'participantId': participantId,
            **demographics,
            'paperOrder': json.dumps(paperOrder),
            'conditions': json.dumps(conditions),
            'contextual_paper': contextual_paper,
            **final_fields,
            'mean_paper_total_score': mean_paper_total_score,
            'mean_paper_free_text_score': mean_paper_free_text_score,
            'cross_total_score': cross_total_score,
        })

    # Write outputs
    pd.DataFrame(paper_rows).to_csv(OUT_PAPER, index=False)
    pd.DataFrame(question_rows).to_csv(OUT_QUESTION, index=False)
    pd.DataFrame(cross_rows).to_csv(OUT_CROSS, index=False)
    pd.DataFrame(final_rows).to_csv(OUT_FINAL, index=False)

    print('Processed sessions:', total_sessions)
    print('Paper rows:', len(paper_rows))
    print('Question rows:', len(question_rows))
    print('Missing manual short-answer scores (approx):', missing_manual)


if __name__ == '__main__':
    main()
