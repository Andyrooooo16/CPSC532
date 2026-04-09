/**
 * Shared question rendering utilities.
 * Used by demographics.html, questionnaire.html, final.html, and the
 * read-only preview in pre-task.html.
 */

/**
 * Render a list of question objects into a container element.
 *
 * @param {HTMLElement} container - The element to render into.
 * @param {Array} questions       - Array of question objects from the API.
 * @param {Object} [opts]
 * @param {boolean} [opts.readonly]      - If true, inputs are disabled.
 * @param {Object}  [opts.prefill]       - Map of questionId → value to prefill.
 */
export function renderQuestions(container, questions, opts = {}) {
  container.innerHTML = '';
  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    const group = document.createElement('div');
    group.className = 'form-group';
    group.dataset.questionId = q.id;
    group.dataset.questionType = q.type;

    const label = document.createElement('label');
    label.innerHTML = `<span class="question-num">${i + 1}.</span> ${escapeHtml(q.text)}`;
    group.appendChild(label);

    if (q.type === 'multiple_choice') {
      group.appendChild(renderMC(q, opts));
    } else if (q.type === 'likert') {
      group.appendChild(renderLikert(q, opts));
    } else {
      group.appendChild(renderFreeText(q, opts));
    }

    container.appendChild(group);
  }
}

/**
 * Collect current answers from a rendered question container.
 * Returns a map of { questionId: value }.
 * MC → string key, likert → number, free_text → string.
 */
export function collectAnswers(container) {
  const answers = {};
  for (const group of container.querySelectorAll('.form-group[data-question-id]')) {
    const id = group.dataset.questionId;
    const type = group.dataset.questionType;

    if (type === 'multiple_choice') {
      const checked = group.querySelector('input[type="radio"]:checked');
      if (checked) answers[id] = checked.value;
    } else if (type === 'likert') {
      const checked = group.querySelector('input[type="radio"]:checked');
      if (checked) answers[id] = parseInt(checked.value, 10);
    } else {
      const ta = group.querySelector('textarea');
      if (ta && ta.value.trim()) answers[id] = ta.value.trim();
    }
  }
  return answers;
}

/**
 * Validate that all questions have been answered.
 * Returns an array of unanswered question IDs.
 */
export function validateAnswers(container, questions) {
  const answers = collectAnswers(container);
  return questions
    .filter(q => answers[q.id] === undefined || answers[q.id] === '')
    .map(q => q.id);
}

/**
 * Mark wrong/correct MC answers after a submission attempt.
 * wrongIds: array of question IDs that were wrong.
 * correctIds: array of question IDs that were correct (optional).
 */
export function markMCFeedback(container, wrongIds, correctIds = []) {
  for (const group of container.querySelectorAll('.form-group[data-question-type="multiple_choice"]')) {
    const id = group.dataset.questionId;
    for (const opt of group.querySelectorAll('.radio-option')) {
      opt.classList.remove('correct', 'incorrect');
    }
    if (wrongIds.includes(id)) {
      const checked = group.querySelector('input[type="radio"]:checked');
      if (checked) {
        checked.closest('.radio-option').classList.add('incorrect');
      }
    } else if (correctIds.includes(id)) {
      const checked = group.querySelector('input[type="radio"]:checked');
      if (checked) {
        checked.closest('.radio-option').classList.add('correct');
      }
    }
  }
}

/**
 * Lock (disable) MC options for questions that are already correct.
 */
export function lockCorrectQuestions(container, correctIds) {
  for (const id of correctIds) {
    const group = container.querySelector(`.form-group[data-question-id="${id}"]`);
    if (!group) continue;
    for (const input of group.querySelectorAll('input[type="radio"]')) {
      input.disabled = true;
    }
    const checked = group.querySelector('input[type="radio"]:checked');
    if (checked) checked.closest('.radio-option').classList.add('correct');
  }
}

// ---------------------------------------------------------------------------
// Private renderers
// ---------------------------------------------------------------------------

function renderMC(q, opts) {
  const wrap = document.createElement('div');
  wrap.className = 'radio-group';

  for (const [key, label] of Object.entries(q.options)) {
    const optDiv = document.createElement('label');
    optDiv.className = 'radio-option';

    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = q.id;
    radio.value = key;
    if (opts.readonly) radio.disabled = true;
    if (opts.prefill && opts.prefill[q.id] === key) radio.checked = true;

    const textSpan = document.createElement('span');
    textSpan.textContent = label;

    optDiv.appendChild(radio);
    optDiv.appendChild(textSpan);

    // Sync selected class
    radio.addEventListener('change', () => {
      for (const o of wrap.querySelectorAll('.radio-option')) o.classList.remove('selected');
      optDiv.classList.add('selected');
    });
    if (radio.checked) optDiv.classList.add('selected');

    wrap.appendChild(optDiv);
  }
  return wrap;
}

function renderLikert(q, opts) {
  const wrap = document.createElement('div');
  wrap.className = 'likert-group';

  const scale = q.scale || { min: 1, max: 5, minLabel: '', maxLabel: '' };
  const scaleDiv = document.createElement('div');
  scaleDiv.className = 'likert-scale';

  for (let v = scale.min; v <= scale.max; v++) {
    const lbl = document.createElement('label');
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = q.id;
    radio.value = v;
    if (opts.readonly) radio.disabled = true;
    if (opts.prefill && opts.prefill[q.id] === v) radio.checked = true;

    lbl.appendChild(radio);
    lbl.appendChild(document.createTextNode(v));
    scaleDiv.appendChild(lbl);
  }

  const labelsDiv = document.createElement('div');
  labelsDiv.className = 'likert-labels';
  labelsDiv.innerHTML = `<span>${escapeHtml(scale.minLabel || '')}</span><span>${escapeHtml(scale.maxLabel || '')}</span>`;

  wrap.appendChild(scaleDiv);
  wrap.appendChild(labelsDiv);
  return wrap;
}

function renderFreeText(q, opts) {
  const ta = document.createElement('textarea');
  ta.name = q.id;
  ta.placeholder = 'Your response…';
  ta.rows = 3;
  if (opts.readonly) ta.disabled = true;
  if (opts.prefill && opts.prefill[q.id]) ta.value = opts.prefill[q.id];
  return ta;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
