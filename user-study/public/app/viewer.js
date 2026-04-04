/**
 * viewer.js — PDF.js rendering for the reader page.
 *
 * Adapted from AltCite WebReader/public/app/viewer-host.js.
 * Changes from original:
 *  - No sources panel / annotation popup code
 *  - Page wrappers use .pdf-page-wrapper class
 *  - Text layer uses .pdf-text-layer class
 *  - Highlight overlay drawn on canvas after page render
 *  - Scale fixed at 1.25 (no fitWidth mode needed)
 */

export async function renderPdf({ container, url, highlights = [] }) {
  const pdfjsLib = globalThis.pdfjsLib;
  if (!pdfjsLib?.getDocument) {
    throw new Error('PDF.js not loaded — ensure /pdfjs/pdf.mjs is imported before viewer.js');
  }

  // Increment generation counter so stale renders self-abort
  const generation = (container._renderGeneration = (container._renderGeneration || 0) + 1);

  const pdf = await pdfjsLib.getDocument(url).promise;
  if (container._renderGeneration !== generation) return { numPages: 0 };

  container.innerHTML = '';

  const SCALE = 1.25;

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    if (container._renderGeneration !== generation) return { numPages: 0 };

    const page = await pdf.getPage(pageNum);
    const dpr = window.devicePixelRatio || 1;
    const layoutViewport = page.getViewport({ scale: SCALE });
    const renderViewport = page.getViewport({ scale: SCALE * dpr });

    const wrapper = document.createElement('div');
    wrapper.className = 'pdf-page-wrapper';
    wrapper.dataset.page = pageNum;
    wrapper.style.width = `${layoutViewport.width}px`;
    wrapper.style.height = `${layoutViewport.height}px`;

    // Canvas
    const canvas = document.createElement('canvas');
    canvas.width = renderViewport.width;
    canvas.height = renderViewport.height;
    canvas.style.width = `${layoutViewport.width}px`;
    canvas.style.height = `${layoutViewport.height}px`;

    const ctx = canvas.getContext('2d');
    await page.render({ canvasContext: ctx, viewport: renderViewport }).promise;

    // Draw highlight overlay on the canvas
    const pageHighlights = highlights.filter(h => h.page === pageNum);
    if (pageHighlights.length > 0) {
      ctx.save();
      ctx.fillStyle = 'rgba(255, 220, 0, 0.38)';
      for (const h of pageHighlights) {
        for (const rect of h.rects) {
          // rect is [x0, y0, x1, y1] in PDF coordinate space (origin bottom-left)
          const [x0, y0, x1, y1] = rect;
          const vRect = renderViewport.convertToViewportRectangle([x0, y0, x1, y1]);
          // vRect is [left, top, right, bottom] in canvas space (origin top-left, at render resolution)
          const left = Math.min(vRect[0], vRect[2]);
          const top = Math.min(vRect[1], vRect[3]);
          const width = Math.abs(vRect[2] - vRect[0]);
          const height = Math.abs(vRect[3] - vRect[1]);
          ctx.fillRect(left, top, width, height);
        }
      }
      ctx.restore();
    }

    wrapper.appendChild(canvas);

    // Text layer (for text selection)
    const textLayerDiv = document.createElement('div');
    textLayerDiv.className = 'pdf-text-layer';
    wrapper.appendChild(textLayerDiv);

    const textLayer = new pdfjsLib.TextLayer({
      textContentSource: page.streamTextContent(),
      container: textLayerDiv,
      viewport: layoutViewport,
    });
    await textLayer.render();

    container.appendChild(wrapper);
  }

  return { numPages: pdf.numPages };
}

export function setupNav({ container, numPages, prevBtn, nextBtn, pageLabel }) {
  let currentPage = 1;

  const updateLabel = () => {
    if (pageLabel) pageLabel.textContent = `${currentPage} / ${numPages}`;
  };
  updateLabel();

  const scrollToPage = (n) => {
    const target = container.querySelector(`[data-page="${n}"]`);
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const onPrev = () => { if (currentPage > 1) scrollToPage(currentPage - 1); };
  const onNext = () => { if (currentPage < numPages) scrollToPage(currentPage + 1); };
  prevBtn.addEventListener('click', onPrev);
  nextBtn.addEventListener('click', onNext);

  const observer = new IntersectionObserver((entries) => {
    let best = null;
    for (const entry of entries) {
      if (entry.isIntersecting) {
        const n = parseInt(entry.target.dataset.page, 10);
        if (!best || entry.intersectionRatio > best.ratio) {
          best = { n, ratio: entry.intersectionRatio };
        }
      }
    }
    if (best) { currentPage = best.n; updateLabel(); }
  }, { root: container, threshold: [0, 0.25, 0.5, 0.75, 1] });

  container.querySelectorAll('.pdf-page-wrapper').forEach(p => observer.observe(p));

  return () => {
    observer.disconnect();
    prevBtn.removeEventListener('click', onPrev);
    nextBtn.removeEventListener('click', onNext);
    if (pageLabel) pageLabel.textContent = '— / —';
  };
}
