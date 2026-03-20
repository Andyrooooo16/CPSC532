var MyPlugin;

function install() {}
function uninstall() {}

// --- Constants ---

const PYTHON_PATH = "/Users/peytonrapo/Desktop/Professional/School/Courses/Human-Centered AI/Project Code/.venv/bin/python";
const SERVICE_DIR = "/Users/peytonrapo/Desktop/Professional/School/Courses/Human-Centered AI/Project Code/service";
const HIGHLIGHTS_PATH = SERVICE_DIR + "/highlights.json";
const READ_TAG = "Read";

const LABEL_COLORS = {
  BACKGROUND:  "#a0c4ff", // light blue
  METHODS:     "#caffbf", // light green
  RESULTS:     "#ffadad", // light red
  OBJECTIVE:   "#ffd6a5", // light orange
  CONCLUSIONS: "#bdb2ff", // light purple
};

// --- Plugin lifecycle ---

async function startup({ id, version, rootURI }) {
  MyPlugin = new class {
    async init({ id, version, rootURI }) {
      this.id = id;
      this.version = version;
      this.rootURI = rootURI;
      this.cleanup = null;
      this.highlights = {};

      await this._reloadHighlights();

      // Observe tab events (open/close PDF reader)
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

      // Observe item tag changes — rerun phase2 when "Read" tag is added
      this.itemObserverID = Zotero.Notifier.registerObserver({
        notify: async (_event, _type, ids, _extraData) => {
          if (_event !== "modify") return;
          for (const id of ids) {
            const item = await Zotero.Items.getAsync(id);
            if (!item) continue;
            const tags = item.getTags().map(t => t.tag);
            if (!tags.includes(READ_TAG)) continue;

            // Find all collections this item is in that are already in highlights.json
            const collectionIDs = item.getCollections();
            for (const colID of collectionIDs) {
              const col = Zotero.Collections.get(colID);
              if (!col) continue;
              if (!(col.name in this.highlights)) continue;
              Services.console.logStringMessage(`[MyPlugin] "${READ_TAG}" tag detected on ${item.key} — rerunning phase2 for "${col.name}"`);
              await _runPhase2(col.name);
              this._reloadHighlights();
            }
          }
        }
      }, ["item"]);

      // Add right-click menu item to collection tree
      this._setupCollectionMenu();

      Services.console.logStringMessage(`[MyPlugin] initialized v${version}`);
    }

    shutdown() {
      Zotero.Notifier.unregisterObserver(this.tabObserverID);
      Zotero.Notifier.unregisterObserver(this.itemObserverID);
      this._teardownCollectionMenu();
      Services.console.logStringMessage(`[MyPlugin] shutdown`);
    }

    async _reloadHighlights() {
      try {
        const text = await IOUtils.readUTF8(HIGHLIGHTS_PATH);
        this.highlights = JSON.parse(text);
        Services.console.logStringMessage(`[MyPlugin] Loaded highlights for: ${Object.keys(this.highlights).join(", ")}`);
      } catch (e) {
        Services.console.logStringMessage(`[MyPlugin] No highlights.json found or parse error: ${e}`);
        this.highlights = {};
      }
    }

    _setupCollectionMenu() {
      const win = Services.wm.getMostRecentWindow("navigator:browser");
      if (!win) return;
      const menu = win.document.getElementById("zotero-collectionmenu");
      if (!menu) return;

      this._menuListener = (event) => {
        // Remove any previously inserted item
        const existing = menu.querySelector("#myPlugin-processCollection");
        if (existing) existing.remove();

        const treeRow = win.ZoteroPane.getCollectionTreeRow();
        if (!treeRow || !treeRow.isCollection()) return;

        const menuitem = win.document.createXULElement("menuitem");
        menuitem.id = "myPlugin-processCollection";
        menuitem.setAttribute("label", "Process Collection (Contextual Highlighter)");
        menuitem.addEventListener("command", async () => {
          const colName = treeRow.ref.name;
          Services.console.logStringMessage(`[MyPlugin] Processing collection: ${colName}`);
          await _runPhase1(colName);
          await _runPhase2(colName);
          this._reloadHighlights();
          Services.console.logStringMessage(`[MyPlugin] Done processing: ${colName}`);
        });

        menu.appendChild(menuitem);
      };

      menu.addEventListener("popupshowing", this._menuListener);
    }

    _teardownCollectionMenu() {
      const win = Services.wm.getMostRecentWindow("navigator:browser");
      if (!win) return;
      const menu = win.document.getElementById("zotero-collectionmenu");
      if (!menu || !this._menuListener) return;
      menu.removeEventListener("popupshowing", this._menuListener);
      const existing = menu.querySelector("#myPlugin-processCollection");
      if (existing) existing.remove();
    }
  };

  await MyPlugin.init({ id, version, rootURI });
}

function shutdown() {
  MyPlugin.shutdown();
  MyPlugin = undefined;
}

// --- Python runner ---

function _runPython(scriptName, args) {
  return new Promise((resolve, reject) => {
    try {
      const pyFile = Components.classes["@mozilla.org/file/local;1"]
        .createInstance(Components.interfaces.nsIFile);
      pyFile.initWithPath(PYTHON_PATH);

      const process = Components.classes["@mozilla.org/process/util;1"]
        .createInstance(Components.interfaces.nsIProcess);
      process.init(pyFile);

      const fullArgs = [SERVICE_DIR + "/" + scriptName, ...args];
      const observer = {
        observe(_subject, topic) {
          if (topic === "process-finished") resolve();
          else reject(new Error(`${scriptName} process failed`));
        }
      };
      process.runAsync(fullArgs, fullArgs.length, observer, false);
    } catch (e) {
      reject(e);
    }
  });
}

async function _runPhase1(collectionName) {
  Services.console.logStringMessage(`[MyPlugin] Running phase1 for "${collectionName}"...`);
  await _runPython("phase1.py", [collectionName]);
  Services.console.logStringMessage(`[MyPlugin] phase1 done.`);
}

async function _runPhase2(collectionName) {
  Services.console.logStringMessage(`[MyPlugin] Running phase2 for "${collectionName}"...`);
  await _runPython("phase2.py", [collectionName]);
  Services.console.logStringMessage(`[MyPlugin] phase2 done.`);
}

// --- Tab / Reader setup ---

async function _addTabHighlights(tabId) {
  Services.console.logStringMessage(`[MyPlugin] _addTabHighlights called for tab: ${tabId}`);
  const reader = Zotero.Reader._readers.find(r => r.tabID === tabId);
  if (!reader) {
    Services.console.logStringMessage(`[MyPlugin] No reader found for tab: ${tabId}`);
    return;
  }

  await reader._initPromise;

  const attachmentKey = reader._item.key;
  const parentItem = Zotero.Items.get(reader._item.parentItemID);
  const itemKey = parentItem ? parentItem.key : attachmentKey;
  Services.console.logStringMessage(`[MyPlugin] Reader found for attachment: ${attachmentKey}, parent: ${itemKey}`);

  // Find highlights for this item across all collections
  let highlights = null;
  for (const colHighlights of Object.values(MyPlugin.highlights)) {
    if (itemKey in colHighlights) {
      highlights = colHighlights[itemKey];
      break;
    }
  }
  if (!highlights || highlights.length === 0) {
    Services.console.logStringMessage(`[MyPlugin] No highlights found for item: ${itemKey}`);
    return;
  }
  Services.console.logStringMessage(`[MyPlugin] Found ${highlights.length} highlights for ${itemKey}`);

  const readerContext = _getReaderContext(reader);
  if (!readerContext) {
    Services.console.logStringMessage(`[MyPlugin] Could not get reader context for ${itemKey}`);
    return;
  }

  const { innerFrame, innerReader } = readerContext;
  const pdfApp = await _waitForPdfApp(reader);
  if (!pdfApp) {
    Services.console.logStringMessage(`[MyPlugin] Could not get pdfApp for ${itemKey}`);
    return;
  }

  Services.console.logStringMessage(`[MyPlugin] Applying ${highlights.length} highlights for ${itemKey}`);
  _applyHighlights(pdfApp, innerFrame, innerReader, highlights);
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

async function _applyHighlights(pdfApp, innerFrame, innerReader, highlightData) {
  const pagesCount = pdfApp.pdfViewer.pagesCount;
  const highlights = [];

  for (let i = 0; i < pagesCount; i++) {
    const page = await pdfApp.pdfDocument.getPage(i + 1);
    const textContent = await page.wrappedJSObject.getTextContent();
    const { fullText, charToItem, itemStartPos } = _buildPageIndex(textContent.items);

    for (const h of highlightData) {
      const matchRange = _fuzzyFind(fullText, h.sentence);
      if (!matchRange) continue;

      const [matchStart, matchEnd] = matchRange;
      Services.console.logStringMessage(`[MyPlugin] MATCH page=${i} start=${matchStart} end=${matchEnd} sentence="${h.sentence.slice(0, 50)}"`);
      const rects = _buildRects(matchStart, matchEnd, charToItem, textContent.items, itemStartPos);
      Services.console.logStringMessage(`[MyPlugin] RECTS ${JSON.stringify(rects)}`);
      highlights.push(_createHighlight(h.sentence, h.label, i, rects));
    }
  }

  Services.console.logStringMessage(`[MyPlugin] Found ${highlights.length}/${highlightData.length} highlights`);

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

function _buildPageIndex(items) {
  let fullText = '';
  const charToItem = [];
  const itemStartPos = []; // itemStartPos[j] = position in fullText where item j begins

  for (let j = 0; j < items.length; j++) {
    const str = items[j].str;
    const cleanStr = items[j].hasEOL ? str.replace(/-$/, '') : str;
    itemStartPos[j] = fullText.length;
    for (let k = 0; k < cleanStr.length; k++) {
      charToItem.push(j);
    }
    fullText += cleanStr;
    // Always add a space between items to prevent "word1word2" concatenation
    fullText += ' ';
    charToItem.push(-1);
  }

  return { fullText, charToItem, itemStartPos };
}

function _buildRects(matchStart, matchEnd, charToItem, items, itemStartPos) {
  const coveredItems = new Set();
  for (let k = matchStart; k < matchEnd; k++) {
    if (charToItem[k] !== -1) coveredItems.add(charToItem[k]);
  }

  const lineMap = new Map();
  for (const itemIdx of coveredItems) {
    const item = items[itemIdx];
    const [, , , , x, y] = item.transform;
    const key = y.toFixed(2);
    // y (transform[5]) is the text baseline. item.height is the full em-size.
    // Descent is below the baseline (~25%), ascent is above (~75%).
    // Using [baseline, baseline+height] shifts the rect up by the descent, causing
    // the top to bleed into the line above. Use [baseline-descent, baseline+ascent] instead.
    const descent = item.height * 0.25;
    const ascent  = item.height * 0.75;
    if (!lineMap.has(key)) {
      // The match may start partway through this item (e.g. when one PDF text item
      // contains both the previous sentence's tail AND our sentence's start).
      // Interpolate to find the approximate x of matchStart within the item.
      let x0 = x;
      const itemStart = itemStartPos[itemIdx];
      const cleanLen  = item.str.length; // cleanStr ≈ item.str in practice
      if (matchStart > itemStart && cleanLen > 0) {
        // Back off by one average character width to account for proportional
        // font rendering — the true start may be slightly left of the interpolated pos.
        const offset = Math.max(0, matchStart - itemStart - 1);
        x0 = x + (offset / cleanLen) * item.width;
      }
      lineMap.set(key, [x0, y - descent, x + item.width, y + ascent]);
      Services.console.logStringMessage(`[MyPlugin]   INIT  y=${y.toFixed(1)} x=${x.toFixed(1)} x0=${x0.toFixed(1)} offset=${matchStart > itemStart ? matchStart - itemStart : 0} cleanLen=${cleanLen} str="${item.str.slice(0,30)}"`);
    } else {
      const r = lineMap.get(key);
      if (x >= r[0]) {
        r[2] = Math.max(r[2], x + item.width);
        Services.console.logStringMessage(`[MyPlugin]   INCL  y=${y.toFixed(1)} x=${x.toFixed(1)} str="${item.str.slice(0,30)}"`);
      } else {
        Services.console.logStringMessage(`[MyPlugin]   SKIP  y=${y.toFixed(1)} x=${x.toFixed(1)} r[0]=${r[0].toFixed(1)} str="${item.str.slice(0,30)}"`);
      }
    }
  }

  return Array.from(lineMap.values()).filter(r => r[3] - r[1] > 0);
}

function _isWordStart(text, idx) {
  if (idx === 0) return true;
  return /[\s.,;:!?()\[\]{}'"\/]/.test(text[idx - 1]);
}

function _fuzzyFind(fullText, target, maxGap = 200) {
  const targetWords = target.split(/\s+/);
  const firstWord = targetWords[0];

  let searchFrom = 0;
  while (searchFrom < fullText.length) {
    const firstIdx = fullText.indexOf(firstWord, searchFrom);
    if (firstIdx === -1) return null;

    // Skip if not at a word boundary
    if (!_isWordStart(fullText, firstIdx)) {
      searchFrom = firstIdx + 1;
      continue;
    }

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

function _createHighlight(text, label, pageIndex, rects) {
  const color = LABEL_COLORS[label] || "#ffff00";
  return {
    id: `tmp-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    type: "highlight",
    text: text,
    color: color,
    position: {
      pageIndex: pageIndex,
      rects: rects
    },
    tags: [],
  };
}
