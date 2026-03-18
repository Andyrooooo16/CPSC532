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
        notify: (event, type, ids, extraData) => {
          Services.console.logStringMessage(`[MyPlugin] ${event} on ${type}: ${JSON.stringify(ids)}`);
          if (event === "select" || event === "close") {
            // unset annotations if present.
            if (this.cleanup) {
              this.cleanup();
              this.cleanup = null;
            }
          }
          if (event === "select" || event === "load") {
            // add annotations if possible
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

function _getTextToHighlightsByItem() {
  return {
    "49SELEVZ": ["Encouraged by the success of deep learning in a variety of domains, we investigate the effectiveness of a novel application of such methods for detecting user confusion with eye-tracking data.", "The image as a whole provides information about the user’s overall attention over the interface."],
    "MJJCTP6P": ["another sentence"]
  }
}

async function _addTabHighlights(tabId) {
  Services.console.logStringMessage(`[MyPlugin] Adding highlights for tab: ${tabId}`);
  let reader = Zotero.Reader.getByTabID(tabId);
  if (reader) {
    Services.console.logStringMessage(`[MyPlugin] Found reader for tab: ${tabId}`);
    // Wait for the reader to be fully initialized
    await reader._initPromise;
    const textToHighlight = MyPlugin.textToHighlightByItem[reader._item.key];
    if (!textToHighlight) return;
    Services.console.logStringMessage(`[MyPlugin] Found text to highlight for tab ${tabId}: ${textToHighlight}`)

    let pdfIframe = reader._iframeWindow.document.querySelector('iframe');
    let innerFrame = reader._iframeWindow.wrappedJSObject;
    let innerReader = innerFrame._reader;
    let pdfWindow = pdfIframe ? pdfIframe.contentWindow : null;
    if (pdfWindow) {
      Services.console.logStringMessage(`[MyPlugin] Found PDF window for tab: ${tabId}`);
      while (!pdfWindow.wrappedJSObject.PDFViewerApplication) {
        await new Promise(r => setTimeout(r, 50));
      }
      let pdfApp = pdfWindow.wrappedJSObject.PDFViewerApplication;
      await pdfApp.initializedPromise;
      while (pdfApp.pdfViewer.pagesCount === 0) {
        await new Promise(r => setTimeout(r, 50));
      }
      Services.console.logStringMessage(`[MyPlugin] Found PDFViewerApplication for tab: ${tabId}`);
      _applyHighlights(pdfApp, innerFrame, innerReader, textToHighlight);
    }
  }
}

async function _applyHighlights(pdfApp, innerFrame, innerReader, textToHighlight) {
  // To highlight I need a set of words I want to highlight and then I need to go find them in the pdf
  // and get their rects so I can add them to the set of highlights in the document. On tab closing, I should clean up.
  const pagesCount = pdfApp.pdfViewer.pagesCount;
  Services.console.logStringMessage(`[MyPlugin] Starting highlighting: (${pagesCount} pages found)`);

  const highlights = [];

  for (let i = 0; i < pagesCount; i++) {
    // Per page:
    // 1. Get the text content of the page (which has positions)
    // 2. For each text content item, get the transform matrix (to convert into pdf space)
    // 3. Find the items that match the text we want to highlight and build the rect.
    Services.console.logStringMessage(`[MyPlugin] Starting highlighting: (page ${i + 1})`);
    const page = await pdfApp.pdfDocument.getPage(i + 1); // 1-indexed
    const textContent = await page.wrappedJSObject.getTextContent();
    const items = textContent.items;

    // Build a flat string from all items, tracking which char maps to which item.
    // Strip hyphenated line breaks so split words match correctly.
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

    for (const target of textToHighlight) {
      // Find the start index of the first target word, then match remaining
      // words in order with a gap allowance to handle noise between words.
      const matchRange = _fuzzyFind(fullText, target);
      if (!matchRange) continue;
      Services.console.logStringMessage(`[MyPlugin] Found fuzzy match for "${target}": ${matchRange}`);

      const [matchStart, matchEnd] = matchRange;

      // Find which items are covered by this match
      const coveredItems = new Set();
      for (let k = matchStart; k < matchEnd; k++) {
        if (charToItem[k] !== -1) coveredItems.add(charToItem[k]);
      }
      Services.console.logStringMessage(`[MyPlugin] Found items covered for "${target}": ${coveredItems}`);

      // Build rects from covered items, merging items on the same line into one rect
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
      const rects = Array.from(lineMap.values());
      Services.console.logStringMessage(`[MyPlugin] Found rects for items for "${target}": ${rects}`);


      const highlight = _createHighlight(target, i, rects);
      Services.console.logStringMessage(`[MyPlugin] Generating highlight for "${target}": ${JSON.stringify(highlight)}`);
      highlights.push(highlight);
    }
  }

  Services.console.logStringMessage(`[MyPlugin] Finished reading pages; found ${highlights.length} highlights out of ${textToHighlight.length}: ${JSON.stringify(highlights)}`);

  const highlightsInner = Cu.cloneInto(highlights, innerFrame);
  innerReader.setAnnotations(highlightsInner);
  Services.console.logStringMessage(`[MyPlugin] Annotations set`);

  pdfApp.pdfViewer.refresh();
  Services.console.logStringMessage(`[MyPlugin] PDF Viewer refreshed`);

  const highlightIDs = highlights.map(a => a.id);
  MyPlugin.cleanup = () => {
    try {
      const idsInner = Cu.cloneInto(highlightIDs, innerFrame);
      innerReader.unsetAnnotations(idsInner);
    } catch (e) {
      // Tab compartment may already be dead if closed
    }
  };
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
    "id": `tmp-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    "type": "highlight",
    "text": text,
    "color": "#004cff",
    "position": {
        "pageIndex": pageIndex,
        "rects": rects
    },
    "tags": [],
  }
}