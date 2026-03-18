var MyPlugin;

function install() {}
function uninstall() {}

async function startup({ id, version, rootURI }) {
  MyPlugin = new class {
    init({id, version, rootURI}) {
      this.id = id;
      this.version = version;
      this.rootURI = rootURI;
      this.cleanup = null;
      this.textToHighlightByItem = _getTextToHighlightsByItem();
      this.tabObserverID = Zotero.Notifier.registerObserver({
        notify: (event, _type, ids, _extraData) => {
          if (event === "select" || event === "close") {
            if (this.cleanup) {
              this.cleanup();
              this.cleanup = null;
            }
          }
          if (event === "select" || event === "load") {
            _addTabHighlights(ids[0]);
          }
        }
      }, ["tab"]);
      Services.console.logStringMessage(`MyPlugin initialized with id: ${id}, version: ${version}, rootURI: ${rootURI}`);
    }

    shutdown() {
      Zotero.Notifier.unregisterObserver(this.tabObserverID);
      Services.console.logStringMessage(`MyPlugin with id: ${this.id} is shutting down.`);
    }
  };

  MyPlugin.init({ id, version, rootURI });
}

function shutdown() {
  MyPlugin.shutdown();
  MyPlugin = undefined;
}

// --- Data ---

function _getTextToHighlightsByItem() {
  return {
    "49SELEVZ": [
      "Encouraged by the success of deep learning in a variety of domains, we investigate the effectiveness of a novel application of such methods for detecting user confusion with eye-tracking data.",
      "The image as a whole provides information about the user’s overall attention over the interface."
    ],
    "MJJCTP6P": ["another sentence"]
  };
}

// --- Tab / Reader setup ---

async function _addTabHighlights(tabId) {
  const reader = Zotero.Reader.getByTabID(tabId);
  if (!reader) return;

  await reader._initPromise;

  const textToHighlight = MyPlugin.textToHighlightByItem[reader._item.key];
  if (!textToHighlight) return;

  const readerContext = _getReaderContext(reader);
  if (!readerContext) return;

  const { innerFrame, innerReader } = readerContext;
  const pdfApp = await _waitForPdfApp(reader);
  if (!pdfApp) return;

  Services.console.logStringMessage(`[MyPlugin] Applying highlights for tab: ${tabId}`);
  _applyHighlights(pdfApp, innerFrame, innerReader, textToHighlight);
}

function _getReaderContext(reader) {
  const pdfIframe = reader._iframeWindow.document.querySelector('iframe');
  if (!pdfIframe) return null;
  const innerFrame = reader._iframeWindow.wrappedJSObject;
  const innerReader = innerFrame._reader;
  return { pdfIframe, innerFrame, innerReader };
}

async function _waitForPdfApp(reader) {
  const pdfIframe = reader._iframeWindow.document.querySelector('iframe');
  const pdfWindow = pdfIframe ? pdfIframe.contentWindow : null;
  if (!pdfWindow) return null;

  while (!pdfWindow.wrappedJSObject.PDFViewerApplication) {
    await new Promise(r => setTimeout(r, 50));
  }
  const pdfApp = pdfWindow.wrappedJSObject.PDFViewerApplication;
  await pdfApp.initializedPromise;
  while (pdfApp.pdfViewer.pagesCount === 0) {
    await new Promise(r => setTimeout(r, 50));
  }
  return pdfApp;
}

// --- Highlighting ---

async function _applyHighlights(pdfApp, innerFrame, innerReader, textToHighlight) {
  const pagesCount = pdfApp.pdfViewer.pagesCount;
  const highlights = [];

  for (let i = 0; i < pagesCount; i++) {
    const page = await pdfApp.pdfDocument.getPage(i + 1); // 1-indexed
    const textContent = await page.wrappedJSObject.getTextContent();
    const { fullText, charToItem } = _buildPageIndex(textContent.items);

    for (const target of textToHighlight) {
      const matchRange = _fuzzyFind(fullText, target);
      if (!matchRange) continue;

      const [matchStart, matchEnd] = matchRange;
      const rects = _buildRects(matchStart, matchEnd, charToItem, textContent.items);
      highlights.push(_createHighlight(target, i, rects));
    }
  }

  Services.console.logStringMessage(`[MyPlugin] Found ${highlights.length}/${textToHighlight.length} highlights`);

  const highlightsInner = Cu.cloneInto(highlights, innerFrame);
  innerReader.setAnnotations(highlightsInner);
  pdfApp.pdfViewer.refresh();

  const highlightIDs = highlights.map(a => a.id);
  MyPlugin.cleanup = () => {
    try {
      innerReader.unsetAnnotations(Cu.cloneInto(highlightIDs, innerFrame));
    } catch (e) {
      // Tab compartment may already be dead if closed
    }
  };
}

// Builds a flat string and a char-to-item index from a page's text items.
// Strips hyphenated line breaks so split words match correctly.
function _buildPageIndex(items) {
  let fullText = '';
  const charToItem = [];

  for (let j = 0; j < items.length; j++) {
    const str = items[j].str;
    // Strip trailing hyphens that break words across lines
    const cleanStr = items[j].hasEOL ? str.replace(/-$/, '') : str;
    for (let k = 0; k < cleanStr.length; k++) {
      charToItem.push(j);
    }
    fullText += cleanStr;
    if (items[j].hasEOL) {
      fullText += ' ';
      charToItem.push(-1);
    }
  }

  return { fullText, charToItem };
}

// Builds merged line-level rects for the character range [matchStart, matchEnd).
function _buildRects(matchStart, matchEnd, charToItem, items) {
  const coveredItems = new Set();
  for (let k = matchStart; k < matchEnd; k++) {
    if (charToItem[k] !== -1) coveredItems.add(charToItem[k]);
  }

  // Merge items on the same line into one rect
  const lineMap = new Map();
  for (const itemIdx of coveredItems) {
    const item = items[itemIdx];
    const [, , , , x, y] = item.transform;
    const key = y.toFixed(2);
    if (!lineMap.has(key)) {
      lineMap.set(key, [x, y, x + item.width, y + item.height]);
    } else {
      const r = lineMap.get(key);
      r[2] = Math.max(r[2], x + item.width); // extend right edge
    }
  }

  return Array.from(lineMap.values()).filter(r => r[3] - r[1] > 0);
}

// Returns [startIndex, endIndex] of the fuzzy match in fullText, or null if not found.
// Words in target must appear in order within maxGap chars of each other.
function _fuzzyFind(fullText, target, maxGap = 200) {
  const targetWords = target.split(/\s+/);
  const firstWord = targetWords[0];

  let searchFrom = 0;
  while (searchFrom < fullText.length) {
    const firstIdx = fullText.indexOf(firstWord, searchFrom);
    if (firstIdx === -1) return null;

    let pos = firstIdx + firstWord.length;
    let matched = true;
    for (let i = 1; i < targetWords.length; i++) {
      const nextIdx = fullText.indexOf(targetWords[i], pos);
      if (nextIdx === -1 || nextIdx - pos > maxGap) {
        matched = false;
        break;
      }
      pos = nextIdx + targetWords[i].length;
    }

    if (matched) return [firstIdx, pos];
    searchFrom = firstIdx + 1;
  }
  return null;
}

function _createHighlight(text, pageIndex, rects) {
  return {
    id: `tmp-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    type: "highlight",
    text: text,
    color: "#004cff",
    position: {
      pageIndex: pageIndex,
      rects: rects
    },
    tags: [],
  };
}
