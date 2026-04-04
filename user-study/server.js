'use strict';

const express = require('express');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Boot: load config and questions, create data dirs
// ---------------------------------------------------------------------------

const CONFIG_PATH = path.join(__dirname, 'study-config.json');
const QUESTIONS_PATH = path.join(__dirname, 'questions.json');
const SESSIONS_DIR = path.join(__dirname, 'data', 'sessions');
const HIGHLIGHTS_DIR = path.join(__dirname, 'data', 'highlights');

if (!fs.existsSync(SESSIONS_DIR)) fs.mkdirSync(SESSIONS_DIR, { recursive: true });
if (!fs.existsSync(HIGHLIGHTS_DIR)) fs.mkdirSync(HIGHLIGHTS_DIR, { recursive: true });

const config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
const questions = JSON.parse(fs.readFileSync(QUESTIONS_PATH, 'utf8'));

// Cache highlight files in memory at startup (keyed by filename without .json)
const highlightCache = {};
if (fs.existsSync(HIGHLIGHTS_DIR)) {
  for (const f of fs.readdirSync(HIGHLIGHTS_DIR)) {
    if (f.endsWith('.json')) {
      highlightCache[f.slice(0, -5)] = JSON.parse(
        fs.readFileSync(path.join(HIGHLIGHTS_DIR, f), 'utf8')
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function newSessionId() {
  return crypto.randomBytes(9).toString('base64url');
}

function sessionPath(id) {
  return path.join(SESSIONS_DIR, id + '.json');
}

function readSession(id) {
  const p = sessionPath(id);
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function writeSession(session) {
  const p = sessionPath(session.sessionId);
  const tmp = p + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(session, null, 2), 'utf8');
  fs.renameSync(tmp, p);
}

// Returns the participant config (paperOrder, conditions) for a given participantId.
function participantConfig(participantId) {
  return config.participants[participantId] || config.defaultParticipant;
}

// Generate all session steps for a given paper count.
// Demographics is always first; final_questionnaire and done are always last.
function allSteps(paperCount) {
  const steps = ['demographics'];
  for (let i = 0; i < paperCount; i++) {
    steps.push(`pre_task_${i}`, `reading_${i}`, `questionnaire_${i}`);
  }
  steps.push('final_questionnaire', 'done');
  return steps;
}

// Advance to the next step, skipping per-paper questionnaire if not configured,
// and skipping final_questionnaire if not configured.
function nextStep(currentStep, session) {
  const paperCount = session.paperOrder.length;
  const steps = allSteps(paperCount);
  let idx = steps.indexOf(currentStep);
  if (idx === -1) return 'done';

  while (idx < steps.length - 1) {
    idx++;
    const candidate = steps[idx];

    // Skip per-paper questionnaire if not in questions.json
    const qMatch = candidate.match(/^questionnaire_(\d+)$/);
    if (qMatch) {
      const paperKey = session.paperOrder[parseInt(qMatch[1])];
      if (!questions.papers[paperKey]?.questionnaire) continue;
    }

    // Skip final_questionnaire if not in questions.json
    if (candidate === 'final_questionnaire' && !questions.finalQuestionnaire) continue;

    return candidate;
  }
  return 'done';
}

// Map a step to the page path segment (for redirect).
function stepToPath(step) {
  if (step === 'demographics') return 'demographics';
  if (step === 'final_questionnaire') return 'final';
  if (step === 'done') return 'done';
  if (step.startsWith('pre_task_')) return 'pre-task';
  if (step.startsWith('reading_')) return 'reader';
  if (step.startsWith('questionnaire_')) return 'questionnaire';
  return 'done';
}

// Current paper index from step string (reading_2 → 2).
function paperIndexFromStep(step) {
  const m = step.match(/_(\d+)$/);
  return m ? parseInt(m[1]) : null;
}

// Compute priorPapers for a paper at position i in the session:
// only papers with contextual_highlights condition that came before position i.
function computePriorPapers(session, paperIndex) {
  const prior = [];
  for (let i = 0; i < paperIndex; i++) {
    if (session.conditions[i] === 'contextual_highlights') {
      prior.push(session.paperOrder[i]);
    }
  }
  return prior;
}

// Build the highlight cache key for a paper/condition/priorPapers combo.
function highlightKey(paperKey, condition, priorPapers) {
  if (condition === 'no_highlights') return null;
  if (condition === 'all_highlights') return `${paperKey}_all`;
  // contextual_highlights
  const sorted = [...priorPapers].sort();
  return sorted.length === 0
    ? `${paperKey}_ctx_`
    : `${paperKey}_ctx_${sorted.join('-')}`;
}

// Fisher-Yates shuffle (returns a new array).
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Build randomized question/option orders for all papers at session creation.
function buildQuestionOrders(paperOrder) {
  const questionOrders = {};
  const optionOrders = {};

  for (const paperKey of paperOrder) {
    const paper = questions.papers[paperKey];
    if (!paper) continue;
    const ids = paper.questions.map(q => q.id);
    questionOrders[paperKey] = shuffle(ids);

    optionOrders[paperKey] = {};
    for (const q of paper.questions) {
      if (q.type === 'multiple_choice' && !q.orderedOptions) {
        optionOrders[paperKey][q.id] = shuffle(Object.keys(q.options));
      }
    }
  }
  return { questionOrders, optionOrders };
}

// Apply stored question/option orders to produce the client-facing question list.
function applyQuestionOrders(paperKey, session) {
  const paper = questions.papers[paperKey];
  if (!paper) return [];

  const order = session.questionOrders[paperKey] || paper.questions.map(q => q.id);
  const optOrders = session.optionOrders[paperKey] || {};

  // Build a map for quick lookup
  const qMap = Object.fromEntries(paper.questions.map(q => [q.id, q]));

  return order.map(id => {
    const q = { ...qMap[id] };
    // Never expose correct answer to client
    delete q.correct;

    if (q.type === 'multiple_choice' && optOrders[q.id]) {
      const orderedOptions = {};
      for (const key of optOrders[q.id]) {
        orderedOptions[key] = q.options[key];
      }
      q.options = orderedOptions;
    }
    return q;
  });
}

// Score a set of answers against correct answers for a paper's MC questions.
// Returns { q1: true/false, ... } only for MC questions.
function scoreAnswers(paperKey, answers) {
  const paper = questions.papers[paperKey];
  if (!paper) return {};
  const scores = {};
  for (const q of paper.questions) {
    if (q.type === 'multiple_choice' && q.correct !== undefined) {
      scores[q.id] = answers[q.id] === q.correct;
    }
  }
  return scores;
}

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ---------------------------------------------------------------------------
// Page routes
// ---------------------------------------------------------------------------

const APP_DIR = path.join(__dirname, 'public', 'app');

app.get('/', (_req, res) => res.sendFile(path.join(APP_DIR, 'landing.html')));

app.get('/session/:id', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);
  res.redirect(`/session/${req.params.id}/${stepToPath(session.step)}`);
});

app.get('/session/:id/demographics', (_req, res) =>
  res.sendFile(path.join(APP_DIR, 'demographics.html')));

app.get('/session/:id/pre-task', (_req, res) =>
  res.sendFile(path.join(APP_DIR, 'pre-task.html')));

app.get('/session/:id/reader', (_req, res) =>
  res.sendFile(path.join(APP_DIR, 'reader.html')));

app.get('/session/:id/questionnaire', (_req, res) =>
  res.sendFile(path.join(APP_DIR, 'questionnaire.html')));

app.get('/session/:id/final', (_req, res) =>
  res.sendFile(path.join(APP_DIR, 'final.html')));

app.get('/session/:id/done', (_req, res) =>
  res.sendFile(path.join(APP_DIR, 'done.html')));

// ---------------------------------------------------------------------------
// API: session creation
// ---------------------------------------------------------------------------

app.post('/api/sessions', (req, res) => {
  const { participantId } = req.body;
  if (!participantId || typeof participantId !== 'string') {
    return res.status(400).json({ error: 'participantId required' });
  }

  const pConfig = participantConfig(participantId.trim());
  const { questionOrders, optionOrders } = buildQuestionOrders(pConfig.paperOrder);
  const sessionId = newSessionId();

  const session = {
    sessionId,
    participantId: participantId.trim(),
    paperOrder: pConfig.paperOrder,
    conditions: pConfig.conditions,
    createdAt: new Date().toISOString(),
    step: 'demographics',
    demographicResponses: null,
    tasks: [],
    questionnaireResponses: [],
    finalResponses: null,
    completedAt: null,
    questionOrders,
    optionOrders,
  };

  writeSession(session);
  res.json({ sessionId });
});

// ---------------------------------------------------------------------------
// API: session state
// ---------------------------------------------------------------------------

app.get('/api/session/:id/state', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  const { step, paperOrder, conditions } = session;
  const paperIndex = paperIndexFromStep(step);
  const paperKey = paperIndex !== null ? paperOrder[paperIndex] : null;
  const paperMeta = paperKey ? config.papers[paperKey] : null;

  // Find startedAt for current reading step if applicable
  let startedAt = null;
  if (step.startsWith('reading_') && paperIndex !== null) {
    const task = session.tasks.find(t => t.paperKey === paperKey);
    startedAt = task?.startedAt ?? null;
  }

  res.json({
    step,
    expectedPath: stepToPath(step),
    paperIndex,
    paperKey,
    paperTitle: paperMeta?.title ?? null,
    totalPapers: paperOrder.length,
    condition: paperIndex !== null ? conditions[paperIndex] : null,
    startedAt,
  });
});

// ---------------------------------------------------------------------------
// API: task questions for current paper (no correct answers)
// ---------------------------------------------------------------------------

app.get('/api/session/:id/questions', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  const paperIndex = paperIndexFromStep(session.step);
  // Also allow fetching questions when on pre_task step
  const preIndex = session.step.startsWith('pre_task_')
    ? parseInt(session.step.split('_')[2])
    : null;
  const idx = paperIndex ?? preIndex;
  if (idx === null) return res.status(400).json({ error: 'No current paper' });

  const paperKey = session.paperOrder[idx];
  const paper = questions.papers[paperKey];
  if (!paper) return res.status(404).json({ error: 'Paper questions not found' });

  res.json({
    paperKey,
    title: paper.title,
    taskInstructions: paper.taskInstructions,
    questions: applyQuestionOrders(paperKey, session),
  });
});

// ---------------------------------------------------------------------------
// API: questionnaire items for current step
// ---------------------------------------------------------------------------

app.get('/api/session/:id/questionnaire-items', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  const { step } = session;

  if (step === 'final_questionnaire') {
    if (!questions.finalQuestionnaire) return res.status(404).json({ error: 'No final questionnaire' });
    return res.json(questions.finalQuestionnaire);
  }

  const qMatch = step.match(/^questionnaire_(\d+)$/);
  if (!qMatch) return res.status(400).json({ error: 'Not a questionnaire step' });

  const paperKey = session.paperOrder[parseInt(qMatch[1])];
  const q = questions.papers[paperKey]?.questionnaire;
  if (!q) return res.status(404).json({ error: 'No questionnaire for this paper' });

  res.json(q);
});

// ---------------------------------------------------------------------------
// API: demographics items
// ---------------------------------------------------------------------------

app.get('/api/demographics', (_req, res) => {
  if (!questions.demographics) return res.status(404).json({ error: 'No demographics configured' });
  res.json(questions.demographics);
});

// ---------------------------------------------------------------------------
// API: highlights for current paper
// ---------------------------------------------------------------------------

app.get('/api/session/:id/highlights', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  const paperIndex = paperIndexFromStep(session.step);
  if (paperIndex === null) return res.json([]);

  const paperKey = session.paperOrder[paperIndex];
  const condition = session.conditions[paperIndex];

  if (condition === 'no_highlights') return res.json([]);

  const priorPapers = computePriorPapers(session, paperIndex);
  const key = highlightKey(paperKey, condition, priorPapers);
  if (!key) return res.json([]);

  const highlights = highlightCache[key];
  if (!highlights) {
    console.warn(`Highlight file not found for key: ${key}`);
    return res.json([]);
  }

  res.json(highlights);
});

// ---------------------------------------------------------------------------
// API: PDF for current paper
// ---------------------------------------------------------------------------

app.get('/api/session/:id/pdf', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  // Determine paper index from reading or pre_task step
  let paperIndex = paperIndexFromStep(session.step);
  if (paperIndex === null) return res.sendStatus(400);

  const paperKey = session.paperOrder[paperIndex];
  const paperMeta = config.papers[paperKey];
  if (!paperMeta) return res.sendStatus(404);

  const filePath = path.join(__dirname, 'public', 'papers', paperMeta.filename);
  if (!fs.existsSync(filePath)) return res.sendStatus(404);

  res.sendFile(filePath);
});

// ---------------------------------------------------------------------------
// API: submit demographics
// ---------------------------------------------------------------------------

app.post('/api/session/:id/submit-demographics', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);
  if (session.step !== 'demographics') return res.status(400).json({ error: 'Not on demographics step' });

  const { responses } = req.body;
  if (!responses) return res.status(400).json({ error: 'responses required' });

  session.demographicResponses = responses;
  session.step = nextStep('demographics', session);
  writeSession(session);

  res.json({ nextPath: `/session/${session.sessionId}/${stepToPath(session.step)}` });
});

// ---------------------------------------------------------------------------
// API: start reading (participant clicked "Start" on pre-task)
// ---------------------------------------------------------------------------

app.post('/api/session/:id/start', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  if (!session.step.startsWith('pre_task_')) {
    return res.status(400).json({ error: 'Not on a pre-task step' });
  }

  const paperIndex = parseInt(session.step.split('_')[2]);
  const paperKey = session.paperOrder[paperIndex];

  // Advance step to reading_N
  session.step = `reading_${paperIndex}`;

  // Initialize task entry if not present
  const existingTask = session.tasks.find(t => t.paperKey === paperKey);
  if (!existingTask) {
    session.tasks.push({
      paperKey,
      condition: session.conditions[paperIndex],
      priorPapers: computePriorPapers(session, paperIndex),
      startedAt: new Date().toISOString(),
      completedAt: null,
      attempts: [],
      perQuestionTimeSeconds: null,
      firstAttemptCorrect: null,
      finalAnswers: null,
    });
  }

  writeSession(session);
  res.json({ nextPath: `/session/${session.sessionId}/reader` });
});

// ---------------------------------------------------------------------------
// API: submit reading attempt (may be called multiple times)
// ---------------------------------------------------------------------------

app.post('/api/session/:id/submit-reading', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  const paperIndex = paperIndexFromStep(session.step);
  if (!session.step.startsWith('reading_') || paperIndex === null) {
    return res.status(400).json({ error: 'Not on a reading step' });
  }

  const paperKey = session.paperOrder[paperIndex];
  const { answers } = req.body;
  if (!answers) return res.status(400).json({ error: 'answers required' });

  const scores = scoreAnswers(paperKey, answers);
  const wrongIds = Object.entries(scores)
    .filter(([, correct]) => !correct)
    .map(([id]) => id);
  const allCorrect = wrongIds.length === 0;

  const task = session.tasks.find(t => t.paperKey === paperKey);
  if (task) {
    task.attempts.push({
      submittedAt: new Date().toISOString(),
      answers: { ...answers },
      scores,
    });
  }

  writeSession(session);
  res.json({ allCorrect, wrongIds });
});

// ---------------------------------------------------------------------------
// API: complete reading (all MC answers correct; client sends per-question times)
// ---------------------------------------------------------------------------

app.post('/api/session/:id/complete-reading', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  const paperIndex = paperIndexFromStep(session.step);
  if (!session.step.startsWith('reading_') || paperIndex === null) {
    return res.status(400).json({ error: 'Not on a reading step' });
  }

  const paperKey = session.paperOrder[paperIndex];
  const { perQuestionTimeSeconds, finalAnswers } = req.body;

  const task = session.tasks.find(t => t.paperKey === paperKey);
  if (task) {
    task.completedAt = new Date().toISOString();
    task.perQuestionTimeSeconds = perQuestionTimeSeconds || {};
    task.finalAnswers = finalAnswers || {};

    // Derive firstAttemptCorrect from first attempt's scores
    if (task.attempts.length > 0) {
      task.firstAttemptCorrect = { ...task.attempts[0].scores };
    }
  }

  session.step = nextStep(`reading_${paperIndex}`, session);
  writeSession(session);

  res.json({ nextPath: `/session/${session.sessionId}/${stepToPath(session.step)}` });
});

// ---------------------------------------------------------------------------
// API: submit questionnaire (per-paper or final)
// ---------------------------------------------------------------------------

app.post('/api/session/:id/submit-questionnaire', (req, res) => {
  const session = readSession(req.params.id);
  if (!session) return res.sendStatus(404);

  const { step } = session;
  const { responses } = req.body;
  if (!responses) return res.status(400).json({ error: 'responses required' });

  if (step === 'final_questionnaire') {
    session.finalResponses = responses;
    session.step = 'done';
    session.completedAt = new Date().toISOString();
  } else {
    const qMatch = step.match(/^questionnaire_(\d+)$/);
    if (!qMatch) return res.status(400).json({ error: 'Not a questionnaire step' });

    const paperKey = session.paperOrder[parseInt(qMatch[1])];
    session.questionnaireResponses.push({
      paperKey,
      submittedAt: new Date().toISOString(),
      responses,
    });
    session.step = nextStep(step, session);

    // If all papers done and no final questionnaire, mark complete
    if (session.step === 'done') session.completedAt = new Date().toISOString();
  }

  writeSession(session);
  res.json({ nextPath: `/session/${session.sessionId}/${stepToPath(session.step)}` });
});

// ---------------------------------------------------------------------------
// Admin: export all sessions
// ---------------------------------------------------------------------------

app.get('/admin/sessions', (req, res) => {
  if (req.query.secret !== config.adminSecret) return res.sendStatus(403);

  const files = fs.readdirSync(SESSIONS_DIR).filter(f => f.endsWith('.json'));
  const sessions = files.map(f =>
    JSON.parse(fs.readFileSync(path.join(SESSIONS_DIR, f), 'utf8'))
  );
  res.json(sessions);
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`User study server running on http://localhost:${PORT}`);
  console.log(`Admin export: http://localhost:${PORT}/admin/sessions?secret=${config.adminSecret}`);
});
