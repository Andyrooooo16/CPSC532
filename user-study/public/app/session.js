/**
 * Shared session utilities for all study pages.
 */

/**
 * Extract session ID from the current URL path.
 * Expects URLs like /session/<id>/...
 */
export function getSessionId() {
  const m = window.location.pathname.match(/^\/session\/([^/]+)/);
  return m ? m[1] : null;
}

/**
 * Fetch the current session state.
 * Returns the state JSON or throws on error.
 */
export async function fetchState(sessionId) {
  const res = await fetch(`/api/session/${sessionId}/state`);
  if (!res.ok) throw new Error(`Failed to load session (${res.status})`);
  return res.json();
}

/**
 * Guard: if the current URL page doesn't match the expected page for this step,
 * redirect. Call this at the top of each page's init.
 *
 * @param {string} expectedPath - The path segment this page handles
 *                                (e.g. 'demographics', 'pre-task', 'reader', ...)
 */
export async function guardPage(expectedPath) {
  const sessionId = getSessionId();
  if (!sessionId) {
    window.location.href = '/';
    return null;
  }

  let state;
  try {
    state = await fetchState(sessionId);
  } catch {
    window.location.href = '/';
    return null;
  }

  if (state.expectedPath !== expectedPath) {
    window.location.href = `/session/${sessionId}/${state.expectedPath}`;
    return null;
  }

  return state;
}
