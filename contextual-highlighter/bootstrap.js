var MyPlugin;

function install() {}
function uninstall() {}

async function startup({ id, version, rootURI }) {
  MyPlugin = new class {
    init({id, version, rootURI}) {
      this.id = id;
      this.version = version;
      this.rootURI = rootURI;
      this.activeHighlights = [];
      this.unsetAnnotations = null;
      this.textToHighlightByItem = _getTextToHighlightsByItem();
      this.tabObserverID = Zotero.Notifier.registerObserver({
        notify: (event, type, ids, extraData) => {
          Services.console.logStringMessage(`[MyPlugin] ${event} on ${type}: ${JSON.stringify(ids)}`);
          if (event === "select" || event === "close") {
            // unset annotations if present.
            if (this.unsetAnnotations && this.activeHighlights) {
              this.unsetAnnotations(this.activeHighlights);
              this.activeHighlights = [];
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
    "49SELEVZ": ["Encouraged by the", "to contain the path made by"],
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
    const textToHighlight = MyPlugin.textToHighlightByItem(reader._item.key);
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
      Services.console.logStringMessage(`[MyPlugin] Found PDFViewerApplication for tab: ${tabId}`);
      _applyHighlights(pdfApp, innerReader, textToHighlight);  
    }
  }
}

async function _applyHighlights(pdfApp, innerReader, textToHighlight) {
  // To highlight I need a set of words I want to highlight and then I need to go find them in the pdf
  // and get their rects so I can add them to the set of highlights in the document. On tab closing, I should clean up.
  const pagesCount = pdfApp.pdfViewer.pagesCount;
  Services.console.logStringMessage(`[MyPlugin] Starting highlighting: (${pagesCount} pages found)`);

  for (let i = 0; i < pagesCount; i++) {
    // Per page:
    // 1. Get the text content of the page (which has positions)
    // 2. For each text content item, get the transform matrix (to convert into pdf space)
    // 3. Find the items that match the text we want to highlight and build the rect.
    Services.console.logStringMessage(`[MyPlugin] Starting highlighting: (page ${i + 1})`);
    const page = await pdfApp.pdfDocument.getPage(i + 1); // 1-indexed
    const textContent = await page.wrappedJSObject.getTextContent();
  }

  const currentAnnotations = innerReader._state.annotations;
  const highlightsInner = Cu.cloneInto(highlights, innerReader);
  const annotationsInner = Cu.cloneInto([...currentAnnotations, ...highlightsInner], innerReader);
  innerReader.setAnnotations(annotationsInner);

  MyPlugin.activeHighlights = highlightsInner;
  MyPlugin.unsetAnnotations = innerReader.unsetAnnotations.bind(innerReader);
}

function _createHighlight(text, rects) {
  return {
    "type": "highlight",
    "text": text,
    "color": "#004cff",
    "position": {
        "pageIndex": 1,
        "rects": rects
    },
    // "tags": [],
  }
}