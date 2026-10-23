const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

export function formatTokenText(text, escapeNewlines) {
  return escapeNewlines ? text.replace(/\r\n|\r|\n/g, "\\n") : text;
}

export function cropBounds(rect, width, height) {
  const w = clamp(Math.round(rect.w / 16) * 16, 16, Math.floor(width / 16) * 16);
  const h = clamp(Math.round(rect.h / 16) * 16, 16, Math.floor(height / 16) * 16);
  return {x: clamp(Math.round(rect.x), 0, width - w), y: clamp(Math.round(rect.y), 0, height - h), w, h};
}

export function fullCrop(width, height) {
  const w = Math.floor(width / 16) * 16, h = Math.floor(height / 16) * 16;
  return {x: Math.floor((width - w) / 2), y: Math.floor((height - h) / 2), w, h};
}

export function dragCrop(start, point, rect, mode, width, height) {
  if (mode === "move") {
    return cropBounds({...rect, x: clamp(rect.x + point.x - start.x, 0, width - rect.w), y: clamp(rect.y + point.y - start.y, 0, height - rect.h)}, width, height);
  }
  const anchor = mode === "new" ? start : {
    x: mode.includes("w") ? rect.x + rect.w : rect.x,
    y: mode.includes("n") ? rect.y + rect.h : rect.y,
  };
  const x = clamp(point.x, 0, width), y = clamp(point.y, 0, height);
  const w = Math.max(16, Math.round(Math.abs(anchor.x - x) / 16) * 16);
  const h = Math.max(16, Math.round(Math.abs(anchor.y - y) / 16) * 16);
  return cropBounds({x: x < anchor.x ? anchor.x - w : anchor.x, y: y < anchor.y ? anchor.y - h : anchor.y, w, h}, width, height);
}

export function resizeLimits(minShortEdge, maxLongEdge) {
  if (![minShortEdge, maxLongEdge].every(value => Number.isSafeInteger(value) && value >= 16 && value <= 0xfffffff0 && value % 16 === 0)) throw new Error("Resize limits must be positive multiples of 16.");
  return {minShortEdge, maxLongEdge};
}

export function initialSize(width, height, minShortEdge = 384, maxLongEdge = 1152) {
  resizeLimits(minShortEdge, maxLongEdge);
  const edge = Math.min(maxLongEdge, Math.max(width, height) * Math.max(1, minShortEdge / Math.min(width, height)));
  return resizeSize(width, height, edge, maxLongEdge);
}

export function resizeSize(width, height, longEdge = Math.max(width, height), maxLongEdge = 1152) {
  if (![width, height, longEdge].every(value => Number.isFinite(value) && value > 0)) throw new Error("Enter a positive output size.");
  resizeLimits(16, maxLongEdge);
  const maximum = maxLongEdge / Math.max(width, height);
  const minimum = 16 / Math.min(width, height);
  if (minimum > maximum) throw new Error("This aspect ratio needs a higher Max long edge in Resize limits.");
  const scale = clamp(longEdge / Math.max(width, height), minimum, maximum);
  return [width, height].map(value => clamp(Math.round(value * scale), 16, maxLongEdge));
}

export function resizeTo(width, height, axis, value, locked, maxLongEdge = 1152) {
  if (!Number.isFinite(value) || value <= 0) throw new Error("Enter a positive output size.");
  resizeLimits(16, maxLongEdge);
  if (locked) return resizeSize(width, height, value * Math.max(width, height) / (axis === "width" ? width : height), maxLongEdge);
  return [axis === "width" ? value : width, axis === "height" ? value : height].map(size => clamp(Math.round(size), 16, maxLongEdge));
}

if (typeof document !== "undefined") startApp();

function startApp() {
  const byId = id => document.getElementById(id);
  const ids = ["file", "dropzone", "choose", "paste", "clear", "editor", "crop-canvas", "reset-crop", "crop-mode", "cancel-crop", "show-grid", "zoom", "viewport", "edit-hint", "crop-x", "crop-y", "crop-width", "crop-height", "crop-size", "width", "height", "lock-ratio", "scale", "scale-value", "fit", "model-size", "preview-size", "media-info", "file-name", "source-size", "audio-preview", "audio", "encode", "cancel", "result", "copy", "auto-copy", "escape-newlines", "api-key", "token-count", "inference-time", "request-time", "status"];
  const el = Object.fromEntries(ids.map(id => [id, byId(id)]));
  for (const id of ["min-short-edge", "max-long-edge", "apply-limits"]) el[id] = byId(id);
  const settings = document.querySelector("main").dataset;
  const uploadLimit = Number(settings.uploadLimit), maxImagePixels = Number(settings.maxImagePixels);
  let sizeLimits = resizeLimits(Number(settings.minShortEdge), Number(settings.maxLongEdge));
  el["min-short-edge"].value = sizeLimits.minShortEdge;
  el["max-long-edge"].value = sizeLimits.maxLongEdge;
  const state = {file: null, image: null, audioURL: null, crop: null, cropRatio: null, customCrop: false, width: 0, height: 0, base: [0, 0], version: 0, controller: null, timer: null, busy: false, drag: null, render: null, cropping: false, selection: null, zoom: 1, rawText: ""};
  const resizedCanvas = document.createElement("canvas");
  const resizedContext = resizedCanvas.getContext("2d");
  const exportCanvas = document.createElement("canvas");
  const cropContext = el["crop-canvas"].getContext("2d"), outputContext = exportCanvas.getContext("2d");

  function status(message, kind = "") {
    el.status.textContent = message;
    el.status.className = `status ${kind}`;
  }

  function resetResult() {
    state.rawText = "";
    el.result.value = "";
    el.copy.disabled = true;
    el["token-count"].textContent = "No result yet";
    el["inference-time"].textContent = "--";
    el["request-time"].textContent = "--";
  }

  function busy(value) {
    state.busy = value;
    el.editor.disabled = value;
    el.encode.disabled = value || !state.file || state.cropping;
    el.encode.textContent = value ? "Encoding..." : "Encode";
    el.cancel.hidden = !value;
    for (const id of ["min-short-edge", "max-long-edge", "apply-limits"]) el[id].disabled = value || state.cropping;
    if (!value) { clearInterval(state.timer); state.timer = null; }
  }

  function abort() {
    state.version++;
    state.controller?.abort();
    state.controller = null;
    busy(false);
  }

  function releaseMedia() {
    state.image?.close?.();
    state.image = null;
    state.file = null;
    state.crop = null;
    state.cropRatio = null;
    state.customCrop = false;
    state.drag = null;
    state.cropping = false; state.selection = null; state.zoom = 1;
    if (state.render !== null) cancelAnimationFrame(state.render);
    state.render = null;
    el.audio.pause();
    el.audio.removeAttribute("src");
    el.audio.load();
    if (state.audioURL) URL.revokeObjectURL(state.audioURL);
    state.audioURL = null;
    el["crop-canvas"].width = el["crop-canvas"].height = 1;
    exportCanvas.width = exportCanvas.height = 1;
    resizedCanvas.width = resizedCanvas.height = 1;
    el.file.value = "";
    el.editor.hidden = el["audio-preview"].hidden = el["media-info"].hidden = true;
    el.dropzone.hidden = false;
    el["file-name"].textContent = el["source-size"].textContent = "";
  }

  function clear() {
    abort(); releaseMedia(); resetResult(); busy(false);
    status("Select a file to encode.");
  }

  function typeOf(file) {
    const type = file.type.toLowerCase();
    if (["image/png", "image/jpeg", "image/webp"].includes(type)) return type;
    if (["audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"].includes(type)) return "audio/wav";
    if (["audio/mpeg", "audio/mp3"].includes(type)) return "audio/mpeg";
    const extension = file.name.split(".").pop().toLowerCase();
    return {png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp", wav: "audio/wav", mp3: "audio/mpeg"}[extension];
  }

  async function loadFile(file) {
    clear();
    const version = state.version;
    try {
      if (!file || !file.size) throw new Error("Choose a nonempty media file.");
      const type = typeOf(file);
      if (!type) throw new Error("Choose a PNG, JPEG, WebP, WAV, or MP3 file.");
      if (file.size > uploadLimit) throw new Error(`File exceeds the ${(uploadLimit / 1048576).toFixed(0)} MiB upload limit.`);
      status("Preparing preview...");
      if (type.startsWith("image/")) {
        const image = await createImageBitmap(file);
        if (state.version !== version) { image.close(); return; }
        if (image.width * image.height > maxImagePixels) { image.close(); throw new Error(`Image exceeds the ${maxImagePixels.toLocaleString()} pixel budget.`); }
        state.image = image;
        state.width = image.width; state.height = image.height;
        el.editor.hidden = false;
        fitModel();
        el["source-size"].textContent = `${image.width} \u00d7 ${image.height}`;
      } else {
        state.audioURL = URL.createObjectURL(file);
        el.audio.src = state.audioURL;
        el["audio-preview"].hidden = false;
        el["source-size"].textContent = `${(file.size / 1048576).toFixed(2)} MiB`;
      }
      state.file = file;
      el["file-name"].textContent = file.name || "Clipboard image";
      el["media-info"].hidden = false;
      el.dropzone.hidden = true;
      busy(false);
      status(type.startsWith("image/") ? "Adjust the crop and size, then encode." : "Audio ready to encode.");
    } catch (error) {
      if (state.version !== version) return;
      releaseMedia(); busy(false); status(error.message, "error");
    }
  }

  function fitModel() {
    const size = initialSize(state.image.width, state.image.height, sizeLimits.minShortEdge, sizeLimits.maxLongEdge);
    checkPixels(size);
    [state.width, state.height] = size;
    state.crop = fullCrop(state.width, state.height);
    state.customCrop = false;
    resetScale(); render();
  }

  function resetScale() {
    state.base = el["lock-ratio"].checked ? [state.image.width, state.image.height] : [state.width, state.height];
    el.scale.min = Math.ceil(16 * Math.max(...state.base) / Math.min(...state.base));
    el.scale.max = el.width.max = el.height.max = sizeLimits.maxLongEdge;
  }

  function checkPixels([width, height]) {
    if (width * height > maxImagePixels) throw new Error(`Resized image exceeds the ${maxImagePixels.toLocaleString()} pixel budget.`);
  }

  function resizeImage(size) {
    checkPixels(size);
    const [width, height] = size;
    const ratio = state.cropRatio;
    state.crop = state.customCrop ? cropBounds({x: ratio.x * width, y: ratio.y * height, w: ratio.w * width, h: ratio.h * height}, width, height) : fullCrop(width, height);
    state.width = width; state.height = height;
  }

  function edited() {
    state.version++;
    resetResult();
    status("Edits ready. Encode to update the token string.");
  }

  function cropChanged() {
    state.cropping = false; state.selection = null;
    state.customCrop = true;
    const rect = state.crop;
    state.cropRatio = {x: rect.x / state.width, y: rect.y / state.height, w: rect.w / state.width, h: rect.h / state.height};
    render(); edited();
  }

  function render() {
    if (!state.image) return;
    const rect = state.crop;
    if (resizedCanvas.width !== state.width || resizedCanvas.height !== state.height) {
      resizedCanvas.width = state.width; resizedCanvas.height = state.height;
      resizedContext.fillStyle = "#fff";
      resizedContext.fillRect(0, 0, state.width, state.height);
      resizedContext.imageSmoothingEnabled = true;
      resizedContext.imageSmoothingQuality = "high";
      resizedContext.drawImage(state.image, 0, 0, state.width, state.height);
    }
    exportCanvas.width = rect.w; exportCanvas.height = rect.h;
    outputContext.drawImage(resizedCanvas, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
    const canvas = el["crop-canvas"];
    const preview = state.cropping ? resizedCanvas : exportCanvas;
    canvas.width = preview.width; canvas.height = preview.height;
    cropContext.drawImage(preview, 0, 0);
    const fit = layoutPreview();
    el.viewport.classList.toggle("zoomed", state.zoom > 1);
    el.viewport.classList.toggle("cropping", state.cropping);
    el.zoom.textContent = state.zoom === 1 ? "Zoom 2x" : "Fit view";
    if (el["show-grid"].checked && !state.cropping) {
      cropContext.strokeStyle = "rgba(60, 90, 150, .4)";
      cropContext.lineWidth = 0.8;
      cropContext.beginPath();
      for (let x = 16; x < canvas.width; x += 16) { cropContext.moveTo(x, 0); cropContext.lineTo(x, canvas.height); }
      for (let y = 16; y < canvas.height; y += 16) { cropContext.moveTo(0, y); cropContext.lineTo(canvas.width, y); }
      cropContext.stroke();
    }
    if (state.cropping && state.selection) {
      const {x, y, w, h} = state.selection;
      cropContext.fillStyle = "rgba(17, 24, 39, .5)";
      cropContext.beginPath(); cropContext.rect(0, 0, state.width, state.height);
      cropContext.rect(x, y, w, h); cropContext.fill("evenodd");
      if (el["show-grid"].checked) {
        cropContext.strokeStyle = "rgba(60, 90, 150, .4)";
        cropContext.lineWidth = 0.8;
        cropContext.beginPath();
        for (let gx = x + 16; gx < x + w; gx += 16) { cropContext.moveTo(gx, y); cropContext.lineTo(gx, y + h); }
        for (let gy = y + 16; gy < y + h; gy += 16) { cropContext.moveTo(x, gy); cropContext.lineTo(x + w, gy); }
        cropContext.stroke();
      }
      cropContext.strokeStyle = "#81a5ff";
      cropContext.lineWidth = 2 / (fit * state.zoom);
      cropContext.strokeRect(x, y, w, h);
      cropContext.fillStyle = "#fff";
      const handle = 7 / (fit * state.zoom);
      for (const cx of [x, x + w]) for (const cy of [y, y + h]) cropContext.fillRect(cx - handle / 2, cy - handle / 2, handle, handle);
    }
    el["crop-mode"].textContent = state.cropping ? "Apply crop" : "Crop";
    el["cancel-crop"].hidden = !state.cropping;
    el["edit-hint"].textContent = state.cropping ? "Crop size snaps to 16 px. Drag to select, move, or resize. Enter applies; Escape cancels." : "Output preview. Edges are cropped to whole patches; cropping never resizes pixels.";
    if (document.activeElement !== el.width) el.width.value = state.width;
    if (document.activeElement !== el.height) el.height.value = state.height;
    const selected = state.selection || rect;
    for (const [key, value] of Object.entries({x: selected.x, y: selected.y, width: selected.w, height: selected.h})) {
      if (document.activeElement !== el[`crop-${key}`]) el[`crop-${key}`].value = value;
    }
    el["crop-size"].textContent = state.selection ? `${state.selection.w} x ${state.selection.h} selected` : `${rect.w} x ${rect.h}`;
    el["preview-size"].textContent = `${selected.w} x ${selected.h} px`;
    el["model-size"].textContent = `${selected.w / 16} x ${selected.h / 16} patches / ${selected.w * selected.h / 256} tokens`;
    el.scale.value = Math.max(state.width, state.height);
    el["scale-value"].textContent = `${Math.max(state.width, state.height)} px`;
    for (const name of ["width", "height", "lock-ratio", "fit"]) el[name].disabled = state.cropping;
    el.scale.disabled = state.cropping || Number(el.scale.min) === sizeLimits.maxLongEdge;
    for (const id of ["min-short-edge", "max-long-edge", "apply-limits"]) el[id].disabled = state.busy || state.cropping;
    el.encode.disabled = state.busy || !state.file || state.cropping;
  }

  function layoutPreview() {
    if (!state.image) return 1;
    const canvas = el["crop-canvas"];
    const fit = Math.max(0.01, Math.min((el.viewport.clientWidth - 18) / canvas.width, (el.viewport.clientHeight - 18) / canvas.height, 1));
    canvas.style.width = `${canvas.width * fit * state.zoom}px`;
    canvas.style.height = `${canvas.height * fit * state.zoom}px`;
    return fit;
  }

  function point(event) {
    const bounds = el["crop-canvas"].getBoundingClientRect();
    return {x: clamp((event.clientX - bounds.left) / bounds.width * state.width, 0, state.width), y: clamp((event.clientY - bounds.top) / bounds.height * state.height, 0, state.height)};
  }

  function applyCrop() {
    if (!state.selection) return;
    const selection = state.selection;
    state.crop = {...selection};
    state.zoom = 1;
    cropChanged();
  }

  el["crop-mode"].addEventListener("click", () => {
    if (state.cropping) { applyCrop(); return; }
    state.cropping = true;
    state.selection = {...state.crop};
    render();
  });
  function cancelCrop() { state.cropping = false; state.selection = null; state.drag = null; render(); }
  el["cancel-crop"].addEventListener("click", cancelCrop);
  el.zoom.addEventListener("click", () => { state.zoom = state.zoom === 1 ? 2 : 1; render(); });
  el["show-grid"].addEventListener("change", render);
  new ResizeObserver(layoutPreview).observe(el.viewport);

  el["crop-canvas"].addEventListener("pointerdown", event => {
    if (!state.image || !state.cropping || state.busy || (event.pointerType === "mouse" && event.button !== 0)) return;
    const p = point(event), rect = {...state.selection};
    const bounds = el["crop-canvas"].getBoundingClientRect();
    const hitX = 10 * state.width / bounds.width, hitY = 10 * state.height / bounds.height;
    let mode = "new";
    for (const [name, x, y] of [["nw", rect.x, rect.y], ["ne", rect.x + rect.w, rect.y], ["sw", rect.x, rect.y + rect.h], ["se", rect.x + rect.w, rect.y + rect.h]]) {
      if (Math.abs(p.x - x) <= hitX && Math.abs(p.y - y) <= hitY) mode = name;
    }
    if (mode === "new" && (state.width - rect.w >= 16 || state.height - rect.h >= 16) && p.x > rect.x && p.x < rect.x + rect.w && p.y > rect.y && p.y < rect.y + rect.h) mode = "move";
    state.drag = {start: p, rect, mode, id: event.pointerId};
    el["crop-canvas"].setPointerCapture(event.pointerId);
    el["crop-canvas"].focus();
    event.preventDefault();
  });
  el["crop-canvas"].addEventListener("pointermove", event => {
    if (!state.drag || event.pointerId !== state.drag.id) return;
    const {start, rect, mode} = state.drag;
    state.selection = dragCrop(start, point(event), rect, mode, state.width, state.height);
    if (state.render === null) state.render = requestAnimationFrame(() => { state.render = null; render(); });
  });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) el["crop-canvas"].addEventListener(event, () => { state.drag = null; });
  el["crop-canvas"].addEventListener("keydown", event => {
    if (!state.cropping || state.busy) return;
    if (event.key === "Enter") { event.preventDefault(); applyCrop(); return; }
    if (event.key === "Escape") { event.preventDefault(); cancelCrop(); return; }
    const delta = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.key];
    if (!delta) return;
    event.preventDefault();
    const step = event.shiftKey ? 10 : 1;
    state.selection = dragCrop({x: 0, y: 0}, {x: delta[0] * step, y: delta[1] * step}, state.selection, "move", state.width, state.height);
    render();
  });
  for (const name of ["x", "y", "width", "height"]) {
    el[`crop-${name}`].addEventListener("input", () => {
      const numbers = ["x", "y", "width", "height"].map(key => Number(el[`crop-${key}`].value));
      if (!numbers.every(Number.isFinite)) return;
      const rect = cropBounds({x: numbers[0], y: numbers[1], w: numbers[2], h: numbers[3]}, state.width, state.height);
      if (state.cropping) { state.selection = rect; render(); }
      else { state.crop = rect; cropChanged(); }
    });
    el[`crop-${name}`].addEventListener("blur", () => {
      const rect = state.selection || state.crop;
      if (rect) el[`crop-${name}`].value = rect[{width: "w", height: "h"}[name] || name];
    });
  }
  el["reset-crop"].addEventListener("click", () => { state.crop = fullCrop(state.width, state.height); state.zoom = 1; cropChanged(); state.customCrop = false; });
  el.fit.addEventListener("click", () => { fitModel(); edited(); });
  el["lock-ratio"].addEventListener("change", () => { resetScale(); render(); });
  for (const axis of ["width", "height"]) {
    let base;
    el[axis].addEventListener("focus", () => { base = [state.width, state.height]; });
    el[axis].addEventListener("input", () => {
      try {
        const ratio = el["lock-ratio"].checked ? [state.image.width, state.image.height] : base || [state.width, state.height];
        resizeImage(resizeTo(...ratio, axis, Number(el[axis].value), el["lock-ratio"].checked, sizeLimits.maxLongEdge));
        resetScale(); render(); edited();
      } catch (error) { el.encode.disabled = true; status(error.message, "error"); }
    });
    el[axis].addEventListener("blur", () => {
      el[axis].value = axis === "width" ? state.width : state.height;
      el.encode.disabled = state.busy || !state.file || state.cropping;
    });
  }
  el.scale.addEventListener("input", () => {
    try {
      resizeImage(resizeSize(...state.base, Number(el.scale.value), sizeLimits.maxLongEdge));
      render(); edited();
    } catch (error) { el.scale.value = Math.max(state.width, state.height); status(error.message, "error"); }
  });
  el["apply-limits"].addEventListener("click", () => {
    try {
      const next = resizeLimits(Number(el["min-short-edge"].value), Number(el["max-long-edge"].value));
      const size = state.image ? initialSize(state.image.width, state.image.height, next.minShortEdge, next.maxLongEdge) : null;
      if (size) checkPixels(size);
      sizeLimits = next;
      if (size) { resizeImage(size); resetScale(); render(); edited(); }
      else status("Resize limits updated.");
    } catch (error) { status(error.message, "error"); }
  });

  function chooseFiles(files) {
    if (files.length !== 1) { status("Choose one media file at a time.", "error"); return; }
    loadFile(files[0]);
  }
  el.choose.addEventListener("click", () => el.file.click());
  el.dropzone.addEventListener("click", () => el.file.click());
  el.dropzone.addEventListener("keydown", event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); el.file.click(); } });
  el.file.addEventListener("change", () => { if (el.file.files.length) chooseFiles(el.file.files); });
  const panel = document.querySelector(".input-panel");
  document.addEventListener("dragover", event => { if (event.dataTransfer.types.includes("Files")) { event.preventDefault(); panel.classList.add("dragging"); } });
  document.addEventListener("dragleave", event => { if (!event.relatedTarget) panel.classList.remove("dragging"); });
  document.addEventListener("drop", event => {
    event.preventDefault(); panel.classList.remove("dragging");
    if (event.dataTransfer.files.length) chooseFiles(event.dataTransfer.files);
  });
  document.addEventListener("paste", event => {
    const files = [...(event.clipboardData?.files || [])];
    if (files.length) { event.preventDefault(); chooseFiles(files); }
  });
  el.paste.addEventListener("click", async () => {
    const version = state.version;
    try {
      if (!navigator.clipboard?.read) throw new Error("Paste with Ctrl / Cmd + V, or choose a file.");
      const items = await navigator.clipboard.read();
      const item = items.find(value => value.types.some(type => type.startsWith("image/")));
      if (!item) throw new Error("The clipboard does not contain an image.");
      const type = item.types.find(type => ["image/png", "image/jpeg", "image/webp"].includes(type));
      if (!type) throw new Error("Copy a PNG, JPEG, or WebP image first.");
      const blob = await item.getType(type);
      if (state.version === version) loadFile(new File([blob], "Clipboard image", {type}));
    } catch (error) { if (state.version === version) status(`Clipboard unavailable. ${error.message} You can also press Ctrl / Cmd + V.`, "error"); }
  });

  const duration = ms => ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
  async function copyResult(automatic = false) {
    const text = el.result.value, version = state.version;
    if (!text) return;
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
      else {
        el.result.focus(); el.result.select();
        if (!document.execCommand("copy")) throw new Error("Copy unavailable");
      }
      if (state.version === version) status("Encoded. Copied to clipboard.", "success");
    } catch {
      if (state.version === version) status(automatic ? "Encoded. Auto-copy was blocked; use Copy or select the text." : "Copy was blocked. Select the token string and press Ctrl / Cmd + C.");
    }
  }

  el.encode.addEventListener("click", async () => {
    if (!state.file || state.busy || state.cropping) return;
    const version = ++state.version;
    const controller = new AbortController(); state.controller = controller;
    resetResult(); busy(true); status("Encoding media...");
    let started;
    try {
      let body = state.file, type = typeOf(state.file);
      if (state.image) {
        body = await new Promise(resolve => exportCanvas.toBlob(resolve, "image/png"));
        type = "image/png";
      }
      if (version !== state.version) return;
      if (!body) throw new Error("Could not export the image.");
      if (body.size > uploadLimit) throw new Error("The edited image exceeds the upload limit. Reduce its dimensions.");
      const headers = {"Content-Type": type};
      if (el["api-key"].value.trim()) headers.Authorization = `Bearer ${el["api-key"].value.trim()}`;
      started = performance.now();
      state.timer = setInterval(() => { el["request-time"].textContent = duration(performance.now() - started); }, 100);
      const query = state.image ? `?min_short_edge=${sizeLimits.minShortEdge}&max_long_edge=${sizeLimits.maxLongEdge}` : "";
      const response = await fetch(`/ui/encode${query}`, {method: "POST", body, headers, signal: controller.signal, cache: "no-store", credentials: "omit", redirect: "error"});
      const data = await response.json();
      if (version !== state.version) return;
      el["request-time"].textContent = duration(performance.now() - started);
      if (!response.ok) throw new Error(response.status === 401 ? "Enter your server API key, then try again." : data.error?.message || `Request failed (${response.status}).`);
      if (typeof data.text !== "string" || !data.text || !Number.isFinite(data.inference_ms) || !Number.isInteger(data.code_count)) throw new Error("The server returned an invalid result.");
      state.rawText = data.text;
      el.result.value = formatTokenText(state.rawText, el["escape-newlines"].checked);
      el.result.scrollIntoView({block: "nearest"});
      el["token-count"].textContent = `${data.code_count.toLocaleString()} tokens`;
      el["inference-time"].textContent = duration(data.inference_ms);
      el.copy.disabled = false;
      state.controller = null;
      busy(false); status("Encoded. Token string ready.", "success");
      if (el["auto-copy"].checked) await copyResult(true);
    } catch (error) {
      if (version === state.version) {
        if (started) el["request-time"].textContent = duration(performance.now() - started);
        status(error.name === "AbortError" ? "Cancelled." : error.message, error.name === "AbortError" ? "" : "error");
      }
    } finally {
      if (version === state.version) { busy(false); state.controller = null; }
    }
  });
  el.copy.addEventListener("click", () => copyResult());
  el["escape-newlines"].addEventListener("change", () => {
    if (!state.rawText) return;
    el.result.value = formatTokenText(state.rawText, el["escape-newlines"].checked);
    if (el["auto-copy"].checked) copyResult(true);
  });
  el.cancel.addEventListener("click", () => { abort(); resetResult(); status("Cancelled. Your media is ready to try again."); });
  el.clear.addEventListener("click", clear);
  window.addEventListener("pagehide", () => { clear(); el["api-key"].value = ""; });
  clear();
}
