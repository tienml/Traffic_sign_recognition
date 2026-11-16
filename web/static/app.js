"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_FILE_SIZE = 10 * 1024 * 1024;
  const MAX_IMAGE_PIXELS = 16_000_000;
  const IMAGE_EXTENSIONS = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };
  const VIEWS = {
    "nhan-dien": ["Nhận diện biển báo", "Từ một ảnh biển báo đến một dự đoán có thể kiểm tra."],
    "danh-muc": ["Danh mục 43 biển báo", "Tra cứu các lớp biển báo mà mô hình GTSRB có thể nhận diện."]
  };
  const state = {
    models: [], modelsLoading: false, input: null, imageVersion: 0, copyController: null,
    predictionVersion: 0, predictionController: null, busy: false, result: null,
    classes: [], classesLoaded: false, classesLoading: false, samplesLoading: false,
    view: "nhan-dien"
  };
  const percent = new Intl.NumberFormat("vi-VN", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const number = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 });
  const integer = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 });
  const featureNames = { pixel: "Pixel", hog: "HOG", hog_pca: "HOG + PCA" };
  const algorithmNames = { knn: "KNN", dt: "Decision Tree" };
  const roiInputs = [$("roi-x1"), $("roi-y1"), $("roi-x2"), $("roi-y2")];

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function text(value, fallback = "") {
    return typeof value === "string" ? value : fallback;
  }

  function validClassId(value) {
    return Number.isInteger(value) && value >= 0 && value < 43;
  }

  function validScore(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  }

  function modelName(model) {
    return text(model.name) || `${algorithmNames[model.algorithm] || text(model.algorithm)} · ${featureNames[model.method] || text(model.method)}`;
  }

  function selectedModel() {
    return state.models.find((model) => String(model.id) === $("model-select").value);
  }

  function announce(message) {
    $("result-live").textContent = message;
  }

  function sameOriginImage(value) {
    if (typeof value !== "string" || !value) return null;
    try {
      const url = new URL(value, window.location.href);
      return url.origin === window.location.origin && /^https?:$/.test(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch (_) { return null; }
  }

  async function apiJSON(path, options = {}, timeout = 25000) {
    const controller = new AbortController();
    const upstream = options.signal;
    let timedOut = false;
    const cancel = () => controller.abort();
    if (upstream) {
      if (upstream.aborted) controller.abort();
      else upstream.addEventListener("abort", cancel, { once: true });
    }
    const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    try {
      const response = await fetch(path, { ...options, signal: controller.signal, credentials: "same-origin", headers: { Accept: "application/json", ...options.headers } });
      let payload;
      try { payload = await response.json(); }
      catch (_) { throw new Error("Máy chủ trả về dữ liệu không hợp lệ. Kiểm tra terminal Flask rồi thử lại."); }
      if (!response.ok) throw new Error(text(payload?.error) || `Yêu cầu thất bại (HTTP ${response.status}). Hãy thử lại.`);
      return payload;
    } catch (error) {
      if (timedOut) throw new Error("Máy chủ phản hồi quá lâu. Kiểm tra terminal Flask rồi thử lại.");
      if (error.name === "AbortError") throw error;
      if (error instanceof TypeError) throw new Error("Không kết nối được với máy chủ. Kiểm tra Flask còn đang chạy rồi thử lại.");
      throw error;
    } finally {
      window.clearTimeout(timer);
      if (upstream) upstream.removeEventListener("abort", cancel);
    }
  }

  function setTheme(value, remember = false) {
    const theme = ["light", "dark"].includes(value) ? value : "system";
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.dataset.theme = theme;
    $("theme-select").value = theme;
    if (remember) {
      try { localStorage.setItem("signal-lab-theme", theme); }
      catch (_) { /* Theme selection remains usable when browser storage is unavailable. */ }
    }
  }

  function setView() {
    const hash = window.location.hash.slice(1);
    const view = Object.hasOwn(VIEWS, hash) ? hash : "nhan-dien";
    state.view = view;
    for (const name of Object.keys(VIEWS)) $("view-" + name).hidden = name !== view;
    for (const link of document.querySelectorAll(".nav-link[data-view]")) {
      const active = link.dataset.view === view;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    $("page-title").textContent = VIEWS[view][0];
    $("page-subtitle").textContent = VIEWS[view][1];
    if (view === "danh-muc") {
      if (!state.classesLoaded && !state.classesLoading) loadClasses();
      else renderClasses();
    }
  }

  function fieldError(id, message) {
    $(id).textContent = message;
    $(id).hidden = !message;
  }

  function getROI(showError = false) {
    if (!$("roi-enabled").checked || !state.input || state.input.kind === "sample") return null;
    const values = roiInputs.map((input) => /^\d+$/.test(input.value) ? Number(input.value) : NaN);
    let message = "";
    if (values.some((value) => !Number.isSafeInteger(value) || value < 0)) message = "Nhập đủ bốn tọa độ là số nguyên không âm.";
    else if (values[0] > values[2] || values[1] > values[3]) message = "Góc dưới phải phải nằm sau hoặc trùng góc trên trái.";
    else if (values[2] >= state.input.width || values[3] >= state.input.height) message = `Tọa độ tối đa là x = ${state.input.width - 1}, y = ${state.input.height - 1}.`;
    if (showError) {
      fieldError("roi-error", message);
      roiInputs.forEach((input) => input.setAttribute("aria-invalid", String(Boolean(message))));
    }
    return message ? false : values;
  }

  function syncROI(reset = false) {
    const input = state.input;
    const ready = Boolean(input?.ready);
    const isSample = input?.kind === "sample";
    $("roi-enabled").disabled = !ready || isSample;
    if (reset || isSample || !ready) $("roi-enabled").checked = false;
    if (reset && ready) {
      [0, 0, input.width - 1, input.height - 1].forEach((value, index) => { roiInputs[index].value = String(value); });
    }
    roiInputs.forEach((field, index) => {
      field.disabled = !ready || isSample || !$("roi-enabled").checked;
      if (ready) field.max = String(index % 2 === 0 ? input.width - 1 : input.height - 1);
      else field.removeAttribute("max");
      field.removeAttribute("aria-invalid");
    });
    $("roi-help").textContent = !ready ? "Chọn ảnh để biết giới hạn tọa độ." : isSample ? "Ảnh minh họa dùng vùng biển báo được chú thích trong Train.csv. Không cần nhập ROI thủ công." : `Ảnh gốc ${input.width} × ${input.height} px. Giới hạn x: 0–${input.width - 1}, y: 0–${input.height - 1}.`;
    fieldError("roi-error", "");
  }

  function syncControls() {
    const model = selectedModel();
    const imageReady = Boolean(state.input?.ready);
    const ready = imageReady && model?.ready === true && !state.modelsLoading && getROI() !== false;
    $("recognize-button").disabled = !ready || state.busy;
    $("retry-predict").disabled = !ready || state.busy;
    $("predict-again").disabled = !ready || state.busy;
    $("copy-image-button").disabled = !imageReady || Boolean(state.copyController);
    $("predict-button-text").textContent = state.busy ? "Đang nhận diện…" : "Nhận diện biển báo";
    $("predict-hint").textContent = state.busy ? "Đang chạy mô hình. Đổi ảnh hoặc mô hình sẽ hủy yêu cầu này." : !model?.ready ? "Cần một mô hình sẵn sàng để nhận diện." : !imageReady ? "Chọn ảnh tải lên hoặc một ảnh minh họa để bắt đầu." : getROI() === false ? "Kiểm tra lại tọa độ vùng biển báo." : "Sẵn sàng phân loại. Điểm hỗ trợ không thay thế việc kiểm chứng.";
  }

  function setResultState(mode) {
    $("result-panel").dataset.state = mode;
    $("result-panel").setAttribute("aria-busy", String(mode === "loading"));
    $("result-empty").hidden = mode === "success" || mode === "error";
    $("request-error").hidden = mode !== "error";
    $("prediction-result").hidden = mode !== "success";
    $("empty-icon").hidden = mode === "loading";
    $("result-spinner").hidden = mode !== "loading";
    $("result-status").textContent = { idle: "Chưa nhận diện", loading: "Đang xử lý", success: "Đã phân loại", error: "Cần thử lại" }[mode];
    $("empty-title").textContent = mode === "loading" ? "Đang phân tích biển báo…" : state.input?.ready ? "Ảnh đã sẵn sàng" : "Kết quả sẽ xuất hiện ở đây";
    $("empty-description").textContent = mode === "loading" ? "Trích xuất đặc trưng và chạy mô hình đã chọn. Chỉ hiển thị kết quả thật từ máy chủ." : state.input?.ready ? "Chọn mô hình sẵn sàng rồi bấm Nhận diện để xem dự đoán và các lớp có điểm hỗ trợ cao nhất." : "Chọn ảnh biển báo và mô hình, sau đó bấm Nhận diện để xem dự đoán cùng điểm hỗ trợ.";
  }

  function invalidatePrediction() {
    state.predictionVersion += 1;
    if (state.predictionController) state.predictionController.abort();
    state.predictionController = null;
    state.busy = false;
    state.result = null;
    $("predicted-icon").onload = null;
    $("predicted-icon").onerror = null;
    $("predicted-icon").hidden = true;
    $("predicted-icon").removeAttribute("src");
    $("processed-image").removeAttribute("src");
    $("processed-details").hidden = true;
    setResultState("idle");
    syncControls();
  }

  function markSelectedSample() {
    for (const button of document.querySelectorAll(".sample-button")) {
      const selected = state.input?.kind === "sample" && button.dataset.sampleId === String(state.input.sample.id);
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
  }

  function releaseInput() {
    state.copyController?.abort();
    state.copyController = null;
    state.input?.decodeController?.abort();
    if (state.input?.image) state.input.image.src = "";
    if (state.input?.objectURL) URL.revokeObjectURL(state.input.objectURL);
    state.input = null;
    state.imageVersion += 1;
    clipboardStatus("");
    $("image-file").value = "";
    $("upload-preview").removeAttribute("src");
    $("image-preview-panel").hidden = true;
    $("dropzone").hidden = false;
    $("image-preview-panel").setAttribute("aria-busy", "false");
    syncROI(true);
    markSelectedSample();
    invalidatePrediction();
  }

  function decodeImage(url, signal) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      const cleanup = () => {
        window.clearTimeout(timer);
        image.onload = null;
        image.onerror = null;
        signal.removeEventListener("abort", cancel);
      };
      const fail = (error) => { cleanup(); image.src = ""; reject(error); };
      const cancel = () => fail(new DOMException("Image changed", "AbortError"));
      const timer = window.setTimeout(() => fail(new Error("Không đọc được ảnh. Hãy chọn lại hoặc dùng một ảnh khác.")), 20000);
      image.onload = () => {
        if (!image.naturalWidth || !image.naturalHeight) { fail(new Error("Ảnh không có kích thước hợp lệ.")); return; }
        if (image.naturalWidth * image.naturalHeight > MAX_IMAGE_PIXELS) { fail(new Error("Ảnh không được vượt quá 16 triệu pixel.")); return; }
        cleanup();
        resolve({ width: image.naturalWidth, height: image.naturalHeight, image });
      };
      image.onerror = () => fail(new Error("Không đọc được ảnh. Hãy dùng một tệp PNG, JPG hoặc WEBP hợp lệ."));
      if (signal.aborted) cancel();
      else { signal.addEventListener("abort", cancel, { once: true }); image.src = url; }
    });
  }

  async function chooseImage(input) {
    releaseInput();
    fieldError("file-error", "");
    const version = state.imageVersion;
    state.input = input;
    input.decodeController = new AbortController();
    $("dropzone").hidden = true;
    $("image-preview-panel").hidden = false;
    $("image-preview-panel").setAttribute("aria-busy", "true");
    $("file-name").textContent = input.kind === "file" ? input.file.name : text(input.sample.label, "Ảnh minh họa từ tập train");
    $("image-dimensions").textContent = "Đang đọc ảnh…";
    $("upload-preview").src = input.url;
    markSelectedSample();
    syncControls();
    try {
      const dimensions = await decodeImage(input.url, input.decodeController.signal);
      if (version !== state.imageVersion) { dimensions.image.src = ""; return; }
      Object.assign(input, dimensions, { ready: true });
      $("image-dimensions").textContent = `${dimensions.width} × ${dimensions.height} px${input.kind === "file" ? " · " + number.format(input.file.size / 1024) + " KB" : " · Ảnh tập train"}`;
      $("image-preview-panel").setAttribute("aria-busy", "false");
      syncROI(true);
      setResultState("idle");
      syncControls();
      announce("Đã chọn ảnh. Kết quả trước đã được xóa; bấm Nhận diện để chạy mô hình.");
    } catch (error) {
      if (version !== state.imageVersion) return;
      releaseInput();
      fieldError("file-error", error.message);
      announce(error.message);
    }
  }

  function chooseFile(file) {
    if (!file) return;
    const supportedType = ["image/png", "image/jpeg", "image/webp"].includes(file.type);
    const supportedName = /\.(png|jpe?g|webp)$/i.test(file.name);
    let error = "";
    if (!supportedType && !(file.type === "" && supportedName)) error = "Định dạng chưa được hỗ trợ. Chọn ảnh PNG, JPG hoặc WEBP.";
    else if (file.size === 0) error = "Tệp ảnh rỗng. Hãy chọn một ảnh khác.";
    else if (file.size > MAX_FILE_SIZE) error = "Ảnh vượt quá 10 MB. Hãy giảm kích thước hoặc chọn một ảnh khác.";
    if (error) {
      releaseInput();
      fieldError("file-error", error);
      announce(error);
      return;
    }
    const objectURL = URL.createObjectURL(file);
    chooseImage({ kind: "file", file, url: objectURL, objectURL, ready: false });
  }

  function clipboardStatus(message, isError = false) {
    $("clipboard-status").textContent = message;
    $("clipboard-status").dataset.state = isError ? "error" : "info";
  }

  function editableTarget(target) {
    if (!(target instanceof Element)) return false;
    if (target.isContentEditable) return true;
    const field = target.closest("input, textarea, select");
    if (!field) return false;
    // File/checkbox/button focus is not text editing: Ctrl+V should still accept an image.
    return field.tagName !== "INPUT" || !["file", "checkbox", "radio", "button", "submit", "reset", "range", "color", "image", "hidden"].includes(field.type);
  }

  function clipboardAllowed(event) {
    return !event.defaultPrevented && state.view === "nhan-dien" && !$("view-nhan-dien").hidden &&
      !editableTarget(event.target) && !editableTarget(document.activeElement) && !window.getSelection()?.toString();
  }

  function pasteImage(event) {
    if (!clipboardAllowed(event) || !event.clipboardData) return;
    const items = Array.from(event.clipboardData.items || []).filter((item) => item.kind === "file");
    const files = items.length ? items.map((item) => ({ file: item.getAsFile(), type: item.type })) :
      Array.from(event.clipboardData.files || []).map((file) => ({ file, type: file.type }));
    const images = files.filter(({ file, type }) => /^image\//i.test(type || file?.type || "") ||
      (!(type || file?.type) && /\.(png|jpe?g|webp)$/i.test(file?.name || "")));
    if (!images.length) return; // Never read clipboard HTML/URLs or fetch remote images.
    event.preventDefault();
    if (files.length !== 1 || images.length !== 1) {
      clipboardStatus("Mỗi lần chỉ dán một ảnh biển báo.", true);
      return;
    }
    const { file, type } = images[0];
    if (!file) { clipboardStatus("Không đọc được ảnh trong bộ nhớ tạm. Hãy chọn tệp từ thiết bị.", true); return; }
    const mime = file.type || type;
    const extension = IMAGE_EXTENSIONS[mime];
    if (!extension && mime) { chooseFile(file.type ? file : new File([file], "clipboard-image", { type: mime })); return; }
    // Native clipboard images may be unnamed. Match the MIME instead of trusting a suffix.
    const safeName = extension ? `anh-dan.${extension}` : `anh-dan.${/\.webp$/i.test(file.name) ? "webp" : /\.jpe?g$/i.test(file.name) ? "jpg" : "png"}`;
    chooseFile(new File([file], safeName, { type: mime, lastModified: Date.now() }));
  }

  function inputImagePNG(input, version, signal) {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement("canvas");
      let settled = false;
      const finish = (error, blob) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        signal.removeEventListener("abort", cancel);
        canvas.width = canvas.height = 1;
        if (error) reject(error);
        else resolve(blob);
      };
      const cancel = () => finish(new DOMException("Image changed", "AbortError"));
      const timer = window.setTimeout(() => finish(new Error("Chuyển ảnh mất quá lâu. Hãy thử lại.")), 20000);
      if (signal.aborted) { cancel(); return; }
      signal.addEventListener("abort", cancel, { once: true });
      try {
        if (version !== state.imageVersion || input !== state.input || !input.ready) { cancel(); return; }
        canvas.width = input.width;
        canvas.height = input.height;
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Trình duyệt không chuyển được ảnh sang PNG.");
        context.drawImage(input.image, 0, 0, input.width, input.height);
        canvas.toBlob((blob) => {
          if (signal.aborted || version !== state.imageVersion || input !== state.input) { cancel(); return; }
          finish(blob?.type === "image/png" ? null : new Error("Không tạo được ảnh PNG để sao chép."), blob);
        }, "image/png");
      } catch (error) { finish(error); }
    });
  }

  async function copyInputImage() {
    const input = state.input;
    if (state.view !== "nhan-dien" || !input?.ready || state.copyController) return;
    const fallback = "Nhấp chuột phải vào ảnh xem trước và chọn Sao chép hình ảnh.";
    if (!navigator.clipboard?.write || typeof ClipboardItem !== "function") {
      clipboardStatus(`Trình duyệt chưa hỗ trợ sao chép ảnh tại đây. Dùng HTTPS hoặc localhost. ${fallback}`, true);
      return;
    }
    const version = state.imageVersion;
    const controller = new AbortController();
    state.copyController = controller;
    syncControls();
    clipboardStatus("Đang sao chép ảnh đầu vào…");
    try {
      const png = inputImagePNG(input, version, controller.signal);
      png.catch(() => {}); // A constructor/write rejection must not leave an unhandled conversion rejection.
      // Pass a Promise so write starts within the click/key's user activation (including Safari).
      await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
      if (version === state.imageVersion && input === state.input && !controller.signal.aborted) {
        clipboardStatus(`Đã sao chép ảnh đầu vào (${input.width} × ${input.height} px).`);
      }
    } catch (error) {
      if (version !== state.imageVersion || input !== state.input || controller.signal.aborted) return;
      clipboardStatus(`Không thể sao chép ảnh. Cho phép trình duyệt ghi vào bộ nhớ tạm và thử lại trên HTTPS hoặc localhost. ${fallback}`, true);
    } finally {
      controller.abort();
      if (state.copyController === controller) { state.copyController = null; syncControls(); }
    }
  }

  async function loadModels() {
    if (state.modelsLoading) return;
    const previous = $("model-select").value;
    state.modelsLoading = true;
    invalidatePrediction();
    $("reload-models").disabled = true;
    $("model-select").disabled = true;
    $("model-status").textContent = "Đang kiểm tra";
    $("model-status").dataset.status = "loading";
    try {
      const payload = await apiJSON("/api/models");
      if (!Array.isArray(payload.models)) throw new Error("Danh sách mô hình từ máy chủ không hợp lệ.");
      state.models = payload.models.filter((model) => model && typeof model.id === "string" && model.id.length > 0);
      const select = $("model-select");
      select.replaceChildren();
      const groups = new Map();
      for (const model of state.models) {
        const groupName = algorithmNames[model.algorithm] || "Mô hình khác";
        if (!groups.has(groupName)) {
          const group = element("optgroup");
          group.label = groupName;
          groups.set(groupName, group);
          select.append(group);
        }
        const option = element("option", "", `${modelName(model)}${model.ready === true ? "" : " · Chưa sẵn sàng"}`);
        option.value = model.id;
        option.disabled = model.ready !== true;
        groups.get(groupName).append(option);
      }
      const readyModels = state.models.filter((model) => model.ready === true);
      const chosen = readyModels.find((model) => model.id === previous) || readyModels.find((model) => model.id === payload.default_model_id) || readyModels[0] || state.models[0];
      if (chosen) select.value = chosen.id;
      else {
        const empty = element("option", "", "Chưa có mô hình đã lưu");
        empty.value = "";
        select.append(empty);
      }
      select.disabled = readyModels.length === 0;
      $("model-setup").hidden = readyModels.length > 0;
      $("setup-title").textContent = "Chưa có mô hình sẵn sàng";
      $("setup-description").textContent = "Mở terminal tại thư mục dự án và chạy lệnh bên dưới để huấn luyện. Sau đó kiểm tra lại.";
      updateModelDescription();
    } catch (error) {
      $("model-status").textContent = "Không kết nối";
      $("model-status").dataset.status = "missing";
      $("model-description").textContent = error.message;
      $("model-setup").hidden = false;
      $("setup-title").textContent = "Chưa tải được danh sách mô hình";
      $("setup-description").textContent = "Kiểm tra máy chủ Flask rồi bấm Kiểm tra lại. Chỉ chạy lệnh huấn luyện nếu dự án chưa có mô hình.";
    } finally {
      state.modelsLoading = false;
      $("reload-models").disabled = false;
      syncControls();
    }
  }

  function updateModelDescription() {
    const model = selectedModel();
    const ready = model?.ready === true;
    $("model-status").textContent = ready ? "Sẵn sàng" : "Chưa sẵn sàng";
    $("model-status").dataset.status = ready ? "ready" : "missing";
    const parts = [];
    if (model) {
      parts.push(`${algorithmNames[model.algorithm] || text(model.algorithm)} · ${featureNames[model.method] || text(model.method)}`);
      if (Number.isInteger(model.n_classes) && model.n_classes >= 0) parts.push(`${integer.format(model.n_classes)} lớp`);
      if (Number.isInteger(model.training_samples) && model.training_samples >= 0) parts.push(`${integer.format(model.training_samples)} ảnh train`);
      if (!ready && text(model.error)) parts.push(model.error);
      if (!ready && text(model.status)) parts.push(`Trạng thái: ${model.status}`);
    }
    $("model-description").textContent = parts.length ? parts.join(" · ") : "Chưa tìm thấy mô hình hợp lệ. Huấn luyện rồi kiểm tra lại danh sách.";
  }

  async function loadSamples() {
    if (state.samplesLoading) return;
    state.samplesLoading = true;
    $("sample-list").setAttribute("aria-busy", "true");
    $("sample-status").textContent = "Đang tìm ảnh minh họa…";
    $("reload-samples").hidden = true;
    try {
      const payload = await apiJSON("/api/samples");
      if (!Array.isArray(payload.samples)) throw new Error("Danh sách ảnh minh họa không hợp lệ.");
      const samples = payload.samples.filter((sample) => sample && typeof sample.id === "string" && sameOriginImage(sample.image_url)).slice(0, 12);
      $("sample-list").replaceChildren();
      samples.forEach((sample, index) => {
        const button = element("button", "sample-button");
        button.id = `sample-${index}`;
        button.type = "button";
        button.dataset.sampleId = sample.id;
        button.title = text(sample.label, `Ảnh tập train, lớp ${sample.class_id}`);
        button.setAttribute("aria-label", `Thử ảnh tập train: ${button.title}`);
        button.setAttribute("aria-pressed", "false");
        const image = element("img");
        image.alt = "";
        image.loading = "lazy";
        image.decoding = "async";
        image.src = sameOriginImage(sample.image_url);
        image.addEventListener("error", () => image.replaceWith(icon("image")), { once: true });
        button.append(image);
        button.addEventListener("click", () => chooseImage({ kind: "sample", sample, url: sameOriginImage(sample.image_url), ready: false }));
        $("sample-list").append(button);
      });
      $("sample-status").textContent = samples.length ? "" : "Chưa có ảnh minh họa. Bạn có thể tải ảnh của mình lên.";
      if (!samples.length) $("reload-samples").hidden = false;
      markSelectedSample();
    } catch (error) {
      $("sample-status").textContent = error.message;
      $("reload-samples").hidden = false;
    } finally {
      state.samplesLoading = false;
      $("sample-list").setAttribute("aria-busy", "false");
    }
  }

  function attachClassImage(container, classId) {
    if (!container.isConnected) return;
    const image = element("img");
    image.alt = "";
    image.loading = "lazy";
    image.decoding = "async";
    const fallback = [...container.childNodes];
    image.addEventListener("error", () => {
      if (container.isConnected && image.parentNode === container) container.replaceChildren(...fallback);
    }, { once: true });
    container.replaceChildren(image);
    image.src = `/api/classes/${classId}/image`;
  }

  function renderPrediction(payload) {
    const prediction = payload.prediction;
    const top = payload.top_k.slice(0, 5);
    $("prediction-label").textContent = text(prediction.label, `Lớp ${prediction.class_id}`);
    $("prediction-label-en").textContent = text(prediction.label_en);
    $("prediction-label-en").lang = "en";
    $("prediction-class-id").textContent = `Mã lớp ${prediction.class_id} · GTSRB`;
    $("prediction-score").textContent = percent.format(prediction.score);
    $("prediction-model").textContent = `${modelName(payload.model)} · ${featureNames[payload.model.method] || text(payload.model.method)}`;
    $("prediction-latency").textContent = typeof payload.elapsed_ms === "number" && Number.isFinite(payload.elapsed_ms) ? `${number.format(payload.elapsed_ms)} ms` : "Không có dữ liệu";
    $("predicted-icon-fallback").textContent = String(prediction.class_id).padStart(2, "0");
    $("predicted-icon-fallback").hidden = false;
    const image = $("predicted-icon");
    image.hidden = true;
    image.onload = () => {
      if (state.result !== payload) return;
      image.hidden = false;
      $("predicted-icon-fallback").hidden = true;
    };
    image.onerror = () => {
      if (state.result !== payload) return;
      image.hidden = true;
      $("predicted-icon-fallback").hidden = false;
    };
    image.src = `/api/classes/${prediction.class_id}/image`;
    $("top-heading").textContent = `${top.length} lớp có điểm cao nhất`;
    $("top-list").replaceChildren();
    $("top-table-body").replaceChildren();
    top.forEach((item, index) => {
      const label = text(item.label, `Lớp ${item.class_id}`);
      const row = element("li", "score-row");
      row.tabIndex = 0;
      const heading = element("div", "score-row-heading");
      heading.append(element("span", "score-name", `${index + 1}. ${label}`), element("span", "score-value", percent.format(item.score)));
      const bar = element("progress", "score-progress");
      bar.max = 1;
      bar.value = item.score;
      bar.setAttribute("aria-label", `${label}, điểm hỗ trợ chưa hiệu chỉnh`);
      bar.setAttribute("aria-valuetext", percent.format(item.score));
      const tooltip = element("span", "score-tooltip");
      tooltip.id = `score-tooltip-${index}`;
      tooltip.setAttribute("role", "tooltip");
      tooltip.append(element("strong", "", `${percent.format(item.score)} · ${item.score.toFixed(4)}`), document.createTextNode(`${label} · Mã lớp ${item.class_id} · Điểm hỗ trợ, không phải độ chính xác`));
      row.setAttribute("aria-describedby", tooltip.id);
      row.append(heading, bar, tooltip);
      $("top-list").append(row);
      const tableRow = element("tr");
      tableRow.append(element("td", "", item.class_id), element("td", "", label), element("td", "numeric", percent.format(item.score)));
      $("top-table-body").append(tableRow);
    });
    const warnings = Array.isArray(payload.warnings) ? payload.warnings.filter((warning) => typeof warning === "string") : [];
    $("prediction-warnings").replaceChildren(...warnings.map((warning) => element("li", "", warning)));
    $("prediction-warnings").hidden = !warnings.length;
    const processed = text(payload.processed_image);
    const validPNG = processed.length < 1500000 && /^data:image\/png;base64,[A-Za-z0-9+/=\r\n]+$/.test(processed);
    $("processed-details").hidden = !validPNG;
    $("processed-details").open = false;
    if (validPNG) $("processed-image").src = processed;
    const showingTable = $("score-table-toggle").getAttribute("aria-expanded") === "true";
    $("top-table-wrapper").hidden = !showingTable;
    $("score-chart").hidden = showingTable;
    setResultState("success");
    announce(`Kết quả: ${labelForPrediction(prediction)}, mã lớp ${prediction.class_id}. Điểm hỗ trợ ${percent.format(prediction.score)}, không phải độ chính xác.`);
  }

  function labelForPrediction(prediction) {
    return text(prediction.label, `Lớp ${prediction.class_id}`);
  }

  async function predict() {
    if (state.busy || !state.input?.ready || selectedModel()?.ready !== true || state.modelsLoading) return;
    const roi = getROI(true);
    if (roi === false) { $("roi-details").open = true; roiInputs.find((input) => !input.disabled)?.focus(); return; }
    const form = new FormData();
    if (state.input.kind === "file") form.append("image", state.input.file);
    else form.append("sample_id", state.input.sample.id);
    form.append("model_id", $("model-select").value);
    form.append("top_k", "5");
    if (roi) form.append("roi", JSON.stringify(roi));
    const version = ++state.predictionVersion;
    const controller = new AbortController();
    state.predictionController = controller;
    state.busy = true;
    state.result = null;
    setResultState("loading");
    syncControls();
    announce("Đang nhận diện ảnh bằng mô hình đã chọn.");
    try {
      const payload = await apiJSON("/api/predict", { method: "POST", body: form, signal: controller.signal }, 120000);
      if (version !== state.predictionVersion) return;
      if (!payload?.model || !validClassId(payload.prediction?.class_id) || !validScore(payload.prediction?.score) || !Array.isArray(payload.top_k) || !payload.top_k.length || !payload.top_k.every((item) => validClassId(item?.class_id) && validScore(item?.score))) throw new Error("Máy chủ trả kết quả dự đoán không hợp lệ. Không hiển thị điểm thay thế; hãy kiểm tra mô hình rồi thử lại.");
      state.result = payload;
      renderPrediction(payload);
    } catch (error) {
      if (version !== state.predictionVersion || error.name === "AbortError") return;
      $("request-error-text").textContent = error.message;
      setResultState("error");
      announce(`Chưa thể nhận diện. ${error.message}`);
    } finally {
      if (version === state.predictionVersion) {
        state.busy = false;
        state.predictionController = null;
        syncControls();
      }
    }
  }

  function normalize(value) {
    return String(value).normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D").toLowerCase();
  }

  async function loadClasses() {
    if (state.classesLoading) return;
    state.classesLoading = true;
    $("reload-classes").disabled = true;
    $("class-grid").setAttribute("aria-busy", "true");
    $("class-count").textContent = "Đang tải danh mục…";
    try {
      const payload = await apiJSON("/api/classes");
      if (!Array.isArray(payload.classes)) throw new Error("Danh mục lớp từ máy chủ không hợp lệ.");
      const unique = new Map();
      payload.classes.forEach((item) => { if (validClassId(item?.class_id)) unique.set(item.class_id, item); });
      state.classes = [...unique.values()].sort((a, b) => a.class_id - b.class_id);
      state.classesLoaded = true;
      const filter = $("category-filter");
      const previous = filter.value;
      const all = element("option", "", "Tất cả nhóm biển");
      all.value = "";
      filter.replaceChildren(all);
      const categories = [...new Set(state.classes.map((item) => text(item.category)).filter(Boolean))].sort((a, b) => a.localeCompare(b, "vi"));
      categories.forEach((category) => { const option = element("option", "", category); option.value = category; filter.append(option); });
      if (categories.includes(previous)) filter.value = previous;
      renderClasses();
    } catch (error) {
      $("class-count").textContent = error.message;
      if (!state.classes.length) {
        $("catalog-empty").hidden = false;
        $("catalog-empty-title").textContent = "Chưa tải được danh mục";
        $("catalog-empty-description").textContent = "Kiểm tra máy chủ rồi dùng nút Tải lại danh mục ở phía trên.";
        $("clear-class-filters").hidden = true;
      }
    } finally {
      state.classesLoading = false;
      $("reload-classes").disabled = false;
      $("class-grid").setAttribute("aria-busy", "false");
    }
  }

  function renderClasses() {
    if (!state.classesLoaded) return;
    const query = normalize($("class-search").value.trim());
    const category = $("category-filter").value;
    const filtered = state.classes.filter((item) => (!category || item.category === category) && (!query || normalize(`${item.class_id} ${text(item.label)} ${text(item.label_en)} ${text(item.category)}`).includes(query)));
    $("class-count").textContent = `${filtered.length} / ${state.classes.length} biển báo${state.classes.length !== 43 ? " · Danh mục máy chủ chưa đủ 43 lớp" : ""}`;
    $("class-grid").replaceChildren();
    $("catalog-empty").hidden = filtered.length > 0;
    $("catalog-empty-title").textContent = state.classes.length ? "Không tìm thấy biển báo" : "Danh mục hiện chưa có dữ liệu";
    $("catalog-empty-description").textContent = state.classes.length ? "Thử tên khác, mã lớp từ 0 đến 42 hoặc bỏ bộ lọc nhóm." : "Tải lại danh mục sau khi máy chủ có dữ liệu lớp.";
    $("clear-class-filters").hidden = !state.classes.length;
    filtered.forEach((item) => {
      const card = element("article", "class-card");
      const top = element("div", "class-card-top");
      const image = element("div", "class-card-image");
      image.setAttribute("aria-hidden", "true");
      image.append(element("span", "class-image-fallback", String(item.class_id).padStart(2, "0")));
      top.append(image, element("span", "", `ID ${item.class_id}`));
      const heading = element("h3", "", text(item.label, `Lớp ${item.class_id}`));
      const english = element("p", "", text(item.label_en));
      english.lang = "en";
      card.append(top, heading, english, element("span", "class-card-category", text(item.category)));
      $("class-grid").append(card);
      if (state.view === "danh-muc") attachClassImage(image, item.class_id);
    });
  }

  async function copyCommand() {
    const command = $("train-command");
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(command.value);
      $("copy-status").textContent = "Đã sao chép lệnh.";
    } catch (_) {
      command.focus();
      command.select();
      command.setSelectionRange(0, command.value.length);
      $("copy-status").textContent = "Đã chọn lệnh. Nhấn Ctrl+C hoặc dùng Sao chép trên thiết bị.";
    }
  }

  let theme = "system";
  try { theme = localStorage.getItem("signal-lab-theme") || "system"; }
  catch (_) { /* System preference remains the default without storage. */ }
  setTheme(theme);
  $("theme-select").addEventListener("change", (event) => setTheme(event.target.value, true));
  window.addEventListener("hashchange", setView);
  for (const link of document.querySelectorAll('.skip-link[href="#main-content"]')) {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      $("main-content").focus({ preventScroll: true });
      $("main-content").scrollIntoView({ block: "start" });
    });
  }
  $("prediction-form").addEventListener("submit", (event) => { event.preventDefault(); predict(); });
  $("image-file").addEventListener("change", (event) => chooseFile(event.target.files[0]));
  $("change-image").addEventListener("click", () => $("image-file").click());
  $("error-change-image").addEventListener("click", () => $("image-file").click());
  $("remove-image").addEventListener("click", () => { releaseInput(); fieldError("file-error", ""); announce("Đã bỏ ảnh và kết quả trước đó."); $("image-file").focus(); });
  $("model-select").addEventListener("change", () => { invalidatePrediction(); updateModelDescription(); announce("Đã đổi mô hình. Bấm Nhận diện để tạo kết quả mới."); });
  $("roi-enabled").addEventListener("change", () => { invalidatePrediction(); syncROI(); syncControls(); });
  roiInputs.forEach((input) => input.addEventListener("input", () => { invalidatePrediction(); getROI(true); syncControls(); }));
  $("reload-models").addEventListener("click", loadModels);
  $("reload-samples").addEventListener("click", loadSamples);
  $("reload-classes").addEventListener("click", loadClasses);
  $("retry-predict").addEventListener("click", predict);
  $("predict-again").addEventListener("click", predict);
  $("copy-command").addEventListener("click", copyCommand);
  $("copy-image-button").addEventListener("click", copyInputImage);
  $("input-image-area").addEventListener("click", () => $("input-image-area").focus({ preventScroll: true }));
  document.addEventListener("paste", pasteImage);
  document.addEventListener("keydown", (event) => {
    if (event.repeat || event.altKey || event.shiftKey || !(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "c") return;
    if (!clipboardAllowed(event) || document.activeElement !== $("input-image-area") || !state.input?.ready) return;
    event.preventDefault();
    copyInputImage();
  });
  $("score-table-toggle").addEventListener("click", () => {
    const show = $("score-table-toggle").getAttribute("aria-expanded") !== "true";
    $("score-table-toggle").setAttribute("aria-expanded", String(show));
    $("score-toggle-text").textContent = show ? "Xem biểu đồ" : "Xem bảng";
    $("top-table-wrapper").hidden = !show;
    $("score-chart").hidden = show;
  });
  $("class-search").addEventListener("input", renderClasses);
  $("category-filter").addEventListener("change", renderClasses);
  $("clear-class-filters").addEventListener("click", () => { $("class-search").value = ""; $("category-filter").value = ""; renderClasses(); $("class-search").focus(); });

  let dragDepth = 0;
  $("dropzone").addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth += 1; $("dropzone").classList.add("is-dragging"); });
  $("dropzone").addEventListener("dragover", (event) => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"; });
  $("dropzone").addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("dropzone").classList.remove("is-dragging"); });
  $("dropzone").addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    $("dropzone").classList.remove("is-dragging");
    if (event.dataTransfer?.files.length === 1) chooseFile(event.dataTransfer.files[0]);
    else if (event.dataTransfer?.files.length > 1) { releaseInput(); fieldError("file-error", "Mỗi lần chỉ chọn một ảnh biển báo."); }
  });
  document.addEventListener("dragover", (event) => { if ([...(event.dataTransfer?.types || [])].includes("Files")) event.preventDefault(); });
  document.addEventListener("drop", (event) => { if ([...(event.dataTransfer?.types || [])].includes("Files")) event.preventDefault(); });
  window.addEventListener("beforeunload", () => {
    state.copyController?.abort();
    state.input?.decodeController?.abort();
    if (state.input?.objectURL) URL.revokeObjectURL(state.input.objectURL);
    state.predictionController?.abort();
  });

  syncROI();
  syncControls();
  setView();
  loadModels();
  loadSamples();
})();
