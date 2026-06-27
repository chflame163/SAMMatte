const DEFAULT_TRIMAP_ERODE_PX = 12;
const DEFAULT_TRIMAP_DILATE_PX = 16;
const DEFAULT_VIDEOMAMA_MAX_RESOLUTION = 1024;

const state = {
  sessionId: null,
  videoUrl: null,
  videoName: "",
  frameCount: 0,
  fps: 25,
  width: 0,
  height: 0,
  samInference: null,
  currentFrame: 0,
  currentMode: "points",
  currentPointLabel: 1,
  points: [],
  keyframeMode: false,
  pointKeyframes: {},
  keyframeFrames: [],
  box: null,
  draftBox: null,
  isDrawingBox: false,
  promptFrameIndex: null,
  promptMaskUrl: null,
  promptMaskImage: null,
  objectCount: 0,
  overlayPreviewUrl: null,
  maskPreviewUrl: null,
  maskPostprocessMode: "videomama",
  videomamaMaxResolution: DEFAULT_VIDEOMAMA_MAX_RESOLUTION,
  vitmatteDevice: "gpu",
  trimapErodePx: DEFAULT_TRIMAP_ERODE_PX,
  trimapDilatePx: DEFAULT_TRIMAP_DILATE_PX,
  pollTimer: null,
  pollInFlight: false,
  lastSnapshotUpdatedAt: 0,
  zoom: 1,
  fitMode: "contain",
  loadingProgress: 0,
  loadingTimer: null,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  panStartScrollLeft: 0,
  panStartScrollTop: 0,
  pointContextIndex: null,
};

const els = {
  dropZone: document.getElementById("dropZone"),
  videoInput: document.getElementById("videoInput"),
  dropZoneInner: document.getElementById("dropZoneInner"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  loadingTitle: document.getElementById("loadingTitle"),
  loadingText: document.getElementById("loadingText"),
  loadingFill: document.getElementById("loadingFill"),
  loadingPercent: document.getElementById("loadingPercent"),
  pickVideoButton: document.getElementById("pickVideoButton"),
  replaceVideoButton: document.getElementById("replaceVideoButton"),
  editorCard: document.getElementById("editorCard"),
  sourceVideo: document.getElementById("sourceVideo"),
  videoWrapper: document.getElementById("videoWrapper"),
  videoViewport: document.getElementById("videoViewport"),
  videoStage: document.getElementById("videoStage"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  zoomOutButton: document.getElementById("zoomOutButton"),
  zoomInButton: document.getElementById("zoomInButton"),
  zoomFitButton: document.getElementById("zoomFitButton"),
  zoomFitWidthButton: document.getElementById("zoomFitWidthButton"),
  zoomFitHeightButton: document.getElementById("zoomFitHeightButton"),
  zoomValue: document.getElementById("zoomValue"),
  frameSlider: document.getElementById("frameSlider"),
  frameNumberInput: document.getElementById("frameNumberInput"),
  prevFrameButton: document.getElementById("prevFrameButton"),
  nextFrameButton: document.getElementById("nextFrameButton"),
  videoName: document.getElementById("videoName"),
  videoMetaText: document.getElementById("videoMetaText"),
  frameSummary: document.getElementById("frameSummary"),
  modeHint: document.getElementById("modeHint"),
  modeTabs: Array.from(document.querySelectorAll(".mode-tab")),
  pointButtons: Array.from(document.querySelectorAll(".pill-button")),
  pointsPanel: document.getElementById("pointsPanel"),
  bboxPanel: document.getElementById("bboxPanel"),
  textPanel: document.getElementById("textPanel"),
  textPromptInput: document.getElementById("textPromptInput"),
  keyframeModeInput: document.getElementById("keyframeModeInput"),
  keyframeLocalText: document.getElementById("keyframeLocalText"),
  prevKeyframeButton: document.getElementById("prevKeyframeButton"),
  nextKeyframeButton: document.getElementById("nextKeyframeButton"),
  deleteKeyframeButton: document.getElementById("deleteKeyframeButton"),
  pointsCountText: document.getElementById("pointsCountText"),
  bboxSummaryText: document.getElementById("bboxSummaryText"),
  applyPromptButton: document.getElementById("applyPromptButton"),
  resetSessionButton: document.getElementById("resetSessionButton"),
  propagateButton: document.getElementById("propagateButton"),
  previewBitrateInput: document.getElementById("previewBitrateInput"),
  samMaxPixelsInput: document.getElementById("samMaxPixelsInput"),
  maskPostprocessSelect: document.getElementById("maskPostprocessSelect"),
  videomamaMaxResolutionField: document.getElementById("videomamaMaxResolutionField"),
  videomamaMaxResolutionInput: document.getElementById("videomamaMaxResolutionInput"),
  vitmatteOptions: document.getElementById("vitmatteOptions"),
  vitmatteDeviceField: document.getElementById("vitmatteDeviceField"),
  vitmatteDeviceSelect: document.getElementById("vitmatteDeviceSelect"),
  trimapErodeInput: document.getElementById("trimapErodeInput"),
  trimapDilateInput: document.getElementById("trimapDilateInput"),
  trimapErodeValue: document.getElementById("trimapErodeValue"),
  trimapDilateValue: document.getElementById("trimapDilateValue"),
  statusBadge: document.getElementById("statusBadge"),
  statusText: document.getElementById("statusText"),
  progressShell: document.getElementById("progressShell"),
  progressFill: document.getElementById("progressFill"),
  progressText: document.getElementById("progressText"),
  objectCountText: document.getElementById("objectCountText"),
  fpsText: document.getElementById("fpsText"),
  resolutionText: document.getElementById("resolutionText"),
  samResolutionText: document.getElementById("samResolutionText"),
  anchorFrameText: document.getElementById("anchorFrameText"),
  keyframeCountText: document.getElementById("keyframeCountText"),
  previewSection: document.getElementById("previewSection"),
  overlayPreviewVideo: document.getElementById("overlayPreviewVideo"),
  overlayPreviewToggleButton: document.getElementById("overlayPreviewToggleButton"),
  overlayPreviewSlider: document.getElementById("overlayPreviewSlider"),
  overlayPreviewFrameText: document.getElementById("overlayPreviewFrameText"),
  maskPreviewVideo: document.getElementById("maskPreviewVideo"),
  maskPreviewToggleButton: document.getElementById("maskPreviewToggleButton"),
  maskPreviewSlider: document.getElementById("maskPreviewSlider"),
  maskPreviewFrameText: document.getElementById("maskPreviewFrameText"),
  maskPreviewCard: document.getElementById("maskPreviewCard"),
  maskContextMenu: document.getElementById("maskContextMenu"),
  pointContextMenu: document.getElementById("pointContextMenu"),
  togglePointButton: document.getElementById("togglePointButton"),
  deletePointButton: document.getElementById("deletePointButton"),
  exportBitrateInput: document.getElementById("exportBitrateInput"),
  exportMaskButton: document.getElementById("exportMaskButton"),
};

const canvasCtx = els.overlayCanvas.getContext("2d");
const previewTimelines = [
  {
    video: els.overlayPreviewVideo,
    toggleButton: els.overlayPreviewToggleButton,
    slider: els.overlayPreviewSlider,
    frameText: els.overlayPreviewFrameText,
  },
  {
    video: els.maskPreviewVideo,
    toggleButton: els.maskPreviewToggleButton,
    slider: els.maskPreviewSlider,
    frameText: els.maskPreviewFrameText,
  },
];

const STATUS_LABELS = {
  idle: "空闲",
  prompting: "生成中",
  prompted: "已确认",
  propagating: "传播中",
  rendering: "渲染中",
  completed: "已完成",
  closed: "已关闭",
  error: "错误",
};

function statusKey(status) {
  return String(status || "idle").toLowerCase();
}

function statusLabel(status) {
  return STATUS_LABELS[statusKey(status)] || String(status || "");
}

function translateMessage(message) {
  return String(message || "");
}

function pointCountText(count) {
  return `已选择 ${count} 个点`;
}

function frameKey(frameIndex) {
  return String(clampFrame(frameIndex));
}

function clonePoints(points) {
  return (points || []).map((point) => ({...point}));
}

function localKeyframeFrames() {
  return Object.keys(state.pointKeyframes)
    .map((frame) => Number(frame))
    .filter((frame) => Number.isFinite(frame))
    .sort((a, b) => a - b);
}

function knownKeyframeFrames() {
  const frames = new Set([
    ...localKeyframeFrames(),
    ...state.keyframeFrames
      .map((frame) => Number(frame))
      .filter((frame) => Number.isFinite(frame)),
  ]);
  return Array.from(frames).sort((a, b) => a - b);
}

function updatePointsUI() {
  els.pointsCountText.textContent = pointCountText(state.points.length);
}

function updateKeyframeUI() {
  const frames = localKeyframeFrames();
  const knownFrames = knownKeyframeFrames();
  const controlsDisabled =
    !state.videoUrl || !state.keyframeMode || state.currentMode !== "points";
  const hasPreviousKeyframe = knownFrames.some((frame) => frame < state.currentFrame);
  const hasNextKeyframe = knownFrames.some((frame) => frame > state.currentFrame);
  const hasCurrentKeyframe =
    knownFrames.includes(state.currentFrame) ||
    (state.currentMode === "points" && state.points.length > 0);
  els.keyframeModeInput.checked = state.keyframeMode;
  els.keyframeLocalText.textContent = `点组帧：${frames.length}`;
  els.keyframeCountText.textContent =
    state.keyframeMode && state.keyframeFrames.length > 0
      ? String(state.keyframeFrames.length)
      : "-";
  els.prevKeyframeButton.disabled = controlsDisabled || !hasPreviousKeyframe;
  els.nextKeyframeButton.disabled = controlsDisabled || !hasNextKeyframe;
  els.deleteKeyframeButton.disabled = controlsDisabled || !hasCurrentKeyframe;
}

function saveCurrentPointKeyframe() {
  if (!state.keyframeMode || state.currentMode !== "points") {
    return;
  }
  const key = frameKey(state.currentFrame);
  if (state.points.length === 0) {
    delete state.pointKeyframes[key];
  } else {
    state.pointKeyframes[key] = clonePoints(state.points);
  }
  updateKeyframeUI();
}

function loadCurrentPointKeyframe() {
  if (!state.keyframeMode || state.currentMode !== "points") {
    return;
  }
  state.points = clonePoints(state.pointKeyframes[frameKey(state.currentFrame)] || []);
  updatePointsUI();
  updateKeyframeUI();
}

function setKeyframeMode(enabled) {
  if (state.keyframeMode && state.currentMode === "points") {
    saveCurrentPointKeyframe();
  }
  state.keyframeMode = Boolean(enabled);
  if (state.keyframeMode) {
    if (state.currentMode === "points" && state.points.length > 0) {
      state.pointKeyframes[frameKey(state.currentFrame)] = clonePoints(state.points);
    }
    loadCurrentPointKeyframe();
  } else {
    state.pointKeyframes = {};
    state.keyframeFrames = [];
    updatePointsUI();
  }
  updateKeyframeUI();
  updateModeHint();
  redrawCanvas();
}

function setStatus(status, message) {
  els.statusBadge.textContent = statusLabel(status);
  els.statusBadge.className = `status-badge ${statusKey(status)}`;
  els.statusText.textContent = translateMessage(message);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollStatusOnce() {
  if (state.pollInFlight) {
    return;
  }
  state.pollInFlight = true;
  try {
    await fetchStatus();
  } catch (error) {
    stopPolling();
    setStatus("error", error.message);
  } finally {
    state.pollInFlight = false;
  }
}

function startPolling() {
  stopPolling();
  pollStatusOnce();
  state.pollTimer = setInterval(pollStatusOnce, 1000);
}

function toggleHidden(element, hidden) {
  element.classList.toggle("hidden", hidden);
}

function stopLoadingAnimation() {
  if (state.loadingTimer) {
    clearInterval(state.loadingTimer);
    state.loadingTimer = null;
  }
}

function setLoadingState(title, text, progress) {
  state.loadingProgress = Math.max(0, Math.min(100, progress));
  els.loadingTitle.textContent = title;
  els.loadingText.textContent = text;
  els.loadingFill.classList.remove("indeterminate");
  els.loadingFill.style.width = `${state.loadingProgress}%`;
  els.loadingPercent.textContent = `${Math.round(state.loadingProgress)}%`;
}

function setLoadingIndeterminate(title, text) {
  els.loadingTitle.textContent = title;
  els.loadingText.textContent = text;
  els.loadingFill.classList.add("indeterminate");
  els.loadingFill.style.width = "42%";
  els.loadingPercent.textContent = "导入中";
}

function showLoadingOverlay() {
  toggleHidden(els.loadingOverlay, false);
}

function hideLoadingOverlay() {
  stopLoadingAnimation();
  toggleHidden(els.loadingOverlay, true);
}

function setEditorLoaded(loaded) {
  els.dropZone.classList.toggle("has-video", loaded);
  toggleHidden(els.dropZoneInner, loaded);
  toggleHidden(els.editorCard, !loaded);
}

function clampFrame(frame) {
  return Math.max(0, Math.min(state.frameCount - 1, frame));
}

function updateVideoMeta() {
  els.videoName.textContent = state.videoName || "未命名视频";
  els.videoMetaText.textContent = `${state.width} x ${state.height} | ${state.fps.toFixed(2)} 帧/秒 | 共 ${state.frameCount} 帧`;
  els.fpsText.textContent = state.fps.toFixed(2);
  els.resolutionText.textContent = `${state.width} x ${state.height}`;
  if (els.samResolutionText) {
    const sam = state.samInference;
    els.samResolutionText.textContent = sam
      ? `${sam.width} x ${sam.height}${sam.resized ? " (resized)" : ""}`
      : "-";
  }
}

function updateFrameUI() {
  els.frameSlider.max = Math.max(0, state.frameCount - 1);
  els.frameSlider.value = state.currentFrame;
  els.frameNumberInput.max = Math.max(0, state.frameCount - 1);
  els.frameNumberInput.value = state.currentFrame;
  els.frameSummary.textContent = `${state.currentFrame} / ${Math.max(0, state.frameCount - 1)}`;
}

function previewMaxFrame() {
  return Math.max(0, state.frameCount - 1);
}

function previewFrameText(frameIndex) {
  return `${frameIndex} / ${previewMaxFrame()}`;
}

function getPreviewFrame(video) {
  if (!Number.isFinite(state.fps) || state.fps <= 0) {
    return 0;
  }
  const currentTime = Number.isFinite(video.currentTime) ? video.currentTime : 0;
  return Math.max(0, Math.min(previewMaxFrame(), Math.round(currentTime * state.fps)));
}

function updatePreviewToggleButton(video, toggleButton) {
  const hasSource = Boolean(video.getAttribute("src"));
  toggleButton.disabled = !hasSource;
  toggleButton.textContent = hasSource && !video.paused && !video.ended ? "暂停" : "播放";
}

function updatePreviewTimeline(video, toggleButton, slider, frameText) {
  const maxFrame = previewMaxFrame();
  const hasSource = Boolean(video.getAttribute("src"));
  const frameIndex = hasSource ? getPreviewFrame(video) : 0;
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    video.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
  updatePreviewToggleButton(video, toggleButton);
  slider.max = String(maxFrame);
  slider.value = String(frameIndex);
  slider.disabled = !hasSource || maxFrame <= 0;
  frameText.textContent = previewFrameText(frameIndex);
}

function resetPreviewTimeline(video, toggleButton, slider, frameText) {
  video.pause();
  video.style.removeProperty("aspect-ratio");
  toggleButton.disabled = true;
  toggleButton.textContent = "播放";
  slider.min = "0";
  slider.max = "0";
  slider.value = "0";
  slider.disabled = true;
  frameText.textContent = "0 / 0";
}

function seekPreviewTimeline(video, toggleButton, slider, frameText, frameIndex) {
  if (!video.getAttribute("src") || !Number.isFinite(state.fps) || state.fps <= 0) {
    updatePreviewTimeline(video, toggleButton, slider, frameText);
    return;
  }
  const targetFrame = Math.max(0, Math.min(previewMaxFrame(), Number(frameIndex) || 0));
  const maxTime =
    Number.isFinite(video.duration) && video.duration > 0
      ? Math.max(0, video.duration - 1 / state.fps)
      : Infinity;
  video.currentTime = Math.min(targetFrame / state.fps, maxTime);
  slider.value = String(targetFrame);
  frameText.textContent = previewFrameText(targetFrame);
}

function togglePreviewPlayback(video) {
  if (!video.getAttribute("src")) {
    return;
  }
  if (video.paused || video.ended) {
    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
    return;
  }
  video.pause();
}

function bindPreviewTimeline(video, toggleButton, slider, frameText) {
  const sync = () => updatePreviewTimeline(video, toggleButton, slider, frameText);
  ["loadedmetadata", "durationchange", "timeupdate", "seeked", "ended", "play", "pause"].forEach((eventName) => {
    video.addEventListener(eventName, sync);
  });
  toggleButton.addEventListener("click", () => {
    togglePreviewPlayback(video);
    sync();
  });
  slider.addEventListener("input", () => {
    seekPreviewTimeline(video, toggleButton, slider, frameText, slider.value);
  });
  sync();
}

function updatePromptSummary(summary = {}) {
  state.objectCount = summary.num_objects || 0;
  els.objectCountText.textContent = String(state.objectCount);
  if (state.keyframeMode && state.keyframeFrames.length > 0) {
    els.anchorFrameText.textContent = state.keyframeFrames.join(", ");
  } else {
    els.anchorFrameText.textContent =
      state.promptFrameIndex === null ? "-" : String(state.promptFrameIndex);
  }
  updateKeyframeUI();
}

function updateModeHint() {
  if (state.currentMode === "points") {
    els.modeHint.textContent =
      state.keyframeMode
        ? "左键添加当前点类型，右键空白添加负向点，右键点位编辑；当前帧拥有独立点组。"
        : "左键添加当前点类型，右键空白添加负向点，右键点位编辑。";
  } else if (state.currentMode === "bbox") {
    els.modeHint.textContent = "在画面上拖拽创建一个框选区域。";
  } else {
    els.modeHint.textContent = "输入文字提示后，点击生成遮罩。";
  }
}

function setMode(mode) {
  hidePointContextMenu();
  if (state.keyframeMode && state.currentMode === "points" && mode !== "points") {
    saveCurrentPointKeyframe();
  }
  state.currentMode = mode;
  els.modeTabs.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  toggleHidden(els.pointsPanel, mode !== "points");
  toggleHidden(els.bboxPanel, mode !== "bbox");
  toggleHidden(els.textPanel, mode !== "text");
  if (mode === "points") {
    loadCurrentPointKeyframe();
  }
  updateModeHint();
  redrawCanvas();
}

function setPointLabel(label) {
  state.currentPointLabel = label;
  els.pointButtons.forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.pointLabel) === label);
  });
}

function resetLocalAnnotations() {
  hidePointContextMenu();
  state.points = [];
  state.pointKeyframes = {};
  state.keyframeFrames = [];
  state.box = null;
  state.draftBox = null;
  updatePointsUI();
  updateKeyframeUI();
  els.bboxSummaryText.textContent = "尚未选择框。";
  redrawCanvas();
}

function resetPromptPreview() {
  state.promptFrameIndex = null;
  state.promptMaskUrl = null;
  state.promptMaskImage = null;
  state.overlayPreviewUrl = null;
  state.maskPreviewUrl = null;
  state.keyframeFrames = [];
  previewTimelines.forEach(({video, toggleButton, slider, frameText}) => {
    video.removeAttribute("src");
    video.load();
    resetPreviewTimeline(video, toggleButton, slider, frameText);
  });
  updatePromptSummary({});
  redrawCanvas();
}

function readBoundedInteger(input, fallback) {
  const parsed = Number.parseInt(input.value, 10);
  if (!Number.isFinite(parsed)) {
    input.value = String(fallback);
    return fallback;
  }
  const min = Number(input.min || 0);
  const max = Number(input.max || 256);
  const bounded = Math.max(min, Math.min(max, parsed));
  input.value = String(bounded);
  return bounded;
}

function updateMaskPostprocessUI() {
  state.maskPostprocessMode = els.maskPostprocessSelect.value;
  toggleHidden(
    els.videomamaMaxResolutionField,
    state.maskPostprocessMode !== "videomama",
  );
  toggleHidden(els.vitmatteDeviceField, state.maskPostprocessMode !== "vitmatte");
  toggleHidden(els.vitmatteOptions, state.maskPostprocessMode !== "vitmatte");
}

function syncVideomamaMaxResolution() {
  state.videomamaMaxResolution = readBoundedInteger(
    els.videomamaMaxResolutionInput,
    DEFAULT_VIDEOMAMA_MAX_RESOLUTION,
  );
  return state.videomamaMaxResolution;
}

function syncTrimapControl(input, output, fallback) {
  const value = readBoundedInteger(input, fallback);
  if (output) {
    output.textContent = `${value} px`;
  }
  return value;
}

function syncTrimapControls() {
  state.trimapErodePx = syncTrimapControl(
    els.trimapErodeInput,
    els.trimapErodeValue,
    DEFAULT_TRIMAP_ERODE_PX,
  );
  state.trimapDilatePx = syncTrimapControl(
    els.trimapDilateInput,
    els.trimapDilateValue,
    DEFAULT_TRIMAP_DILATE_PX,
  );
}

function getMaskPostprocessPayload() {
  state.vitmatteDevice = els.vitmatteDeviceSelect?.value || "gpu";
  syncVideomamaMaxResolution();
  syncTrimapControls();
  return {
    mode: state.maskPostprocessMode,
    videomamaMaxResolution: state.videomamaMaxResolution,
    vitmatteDevice: state.vitmatteDevice,
    trimapErodePx: state.trimapErodePx,
    trimapDilatePx: state.trimapDilatePx,
  };
}

function loadMaskImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

function updateZoomUI() {
  const stageTooWide = els.videoStage.offsetWidth > els.videoViewport.clientWidth + 1;
  const stageTooTall = els.videoStage.offsetHeight > els.videoViewport.clientHeight + 1;
  els.zoomValue.textContent = `${Math.round(state.zoom * 100)}%`;
  els.zoomOutButton.disabled = state.zoom <= 1;
  els.videoViewport.classList.toggle("pan-ready", stageTooWide || stageTooTall);
  els.zoomFitButton.classList.toggle("active", state.fitMode === "contain" && state.zoom === 1);
  els.zoomFitWidthButton.classList.toggle(
    "active",
    state.fitMode === "width" && state.zoom === 1,
  );
  els.zoomFitHeightButton.classList.toggle(
    "active",
    state.fitMode === "height" && state.zoom === 1,
  );
}

function getBaseStageSize() {
  const viewportWidth = Math.max(1, els.videoViewport.clientWidth);
  const viewportHeight = Math.max(1, els.videoViewport.clientHeight);
  const videoWidth = Math.max(1, state.width || viewportWidth);
  const videoHeight = Math.max(1, state.height || viewportHeight);
  const aspect = videoWidth / videoHeight;

  if (state.fitMode === "width") {
    return {
      width: viewportWidth,
      height: viewportWidth / aspect,
    };
  }
  if (state.fitMode === "height") {
    return {
      width: viewportHeight * aspect,
      height: viewportHeight,
    };
  }

  const containScale = Math.min(viewportWidth / videoWidth, viewportHeight / videoHeight);
  return {
    width: videoWidth * containScale,
    height: videoHeight * containScale,
  };
}

function applyZoom(options = {}) {
  if (!state.videoUrl) {
    return;
  }
  const {keepCenter = false, anchor = null} = options;
  const previousScale = Number(els.videoStage.dataset.scale || "1");
  const previousWidth = Math.max(1, els.videoStage.offsetWidth || 1);
  const previousHeight = Math.max(1, els.videoStage.offsetHeight || 1);
  const previousCenterX = els.videoViewport.scrollLeft + els.videoViewport.clientWidth / 2;
  const previousCenterY = els.videoViewport.scrollTop + els.videoViewport.clientHeight / 2;
  const baseSize = getBaseStageSize();
  const nextWidth = Math.max(1, Math.round(baseSize.width * state.zoom));
  const nextHeight = Math.max(1, Math.round(baseSize.height * state.zoom));

  els.videoStage.style.width = `${nextWidth}px`;
  els.videoStage.style.height = `${nextHeight}px`;
  els.videoStage.dataset.scale = String(state.zoom);

  if (anchor) {
    const relX = (els.videoViewport.scrollLeft + anchor.x) / previousWidth;
    const relY = (els.videoViewport.scrollTop + anchor.y) / previousHeight;
    els.videoViewport.scrollLeft = Math.max(
      0,
      relX * nextWidth - anchor.x,
    );
    els.videoViewport.scrollTop = Math.max(
      0,
      relY * nextHeight - anchor.y,
    );
  } else if (keepCenter) {
    const ratio = state.zoom / previousScale;
    els.videoViewport.scrollLeft = Math.max(
      0,
      previousCenterX * ratio - els.videoViewport.clientWidth / 2,
    );
    els.videoViewport.scrollTop = Math.max(
      0,
      previousCenterY * ratio - els.videoViewport.clientHeight / 2,
    );
  } else if (state.zoom === 1) {
    els.videoViewport.scrollLeft = 0;
    els.videoViewport.scrollTop = 0;
  }

  updateZoomUI();
  redrawCanvas();
}

function resizeCanvas() {
  const rect = els.videoStage.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (els.overlayCanvas.width !== width || els.overlayCanvas.height !== height) {
    els.overlayCanvas.width = width;
    els.overlayCanvas.height = height;
  }
}

function drawPoint(point) {
  const x = point.x * els.overlayCanvas.width;
  const y = point.y * els.overlayCanvas.height;
  const color = point.label === 1 ? "#4dd0c2" : "#ff7a87";
  canvasCtx.beginPath();
  canvasCtx.arc(x, y, 7, 0, Math.PI * 2);
  canvasCtx.fillStyle = color;
  canvasCtx.fill();
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = "#081019";
  canvasCtx.stroke();
}

function drawBox(box, dashed = false) {
  if (!box) {
    return;
  }
  const x = box.x * els.overlayCanvas.width;
  const y = box.y * els.overlayCanvas.height;
  const width = box.w * els.overlayCanvas.width;
  const height = box.h * els.overlayCanvas.height;
  canvasCtx.save();
  if (dashed) {
    canvasCtx.setLineDash([8, 6]);
  }
  canvasCtx.strokeStyle = "#f7b955";
  canvasCtx.lineWidth = 3;
  canvasCtx.strokeRect(x, y, width, height);
  canvasCtx.fillStyle = "rgba(247, 185, 85, 0.12)";
  canvasCtx.fillRect(x, y, width, height);
  canvasCtx.restore();
}

function redrawCanvas() {
  resizeCanvas();
  canvasCtx.clearRect(0, 0, els.overlayCanvas.width, els.overlayCanvas.height);
  if (
    state.promptMaskImage &&
    state.promptFrameIndex !== null &&
    state.promptFrameIndex === state.currentFrame
  ) {
    canvasCtx.drawImage(
      state.promptMaskImage,
      0,
      0,
      els.overlayCanvas.width,
      els.overlayCanvas.height,
    );
  }
  if (state.currentMode === "points") {
    state.points.forEach(drawPoint);
  }
  if (state.currentMode === "bbox") {
    drawBox(state.box);
    drawBox(state.draftBox, true);
  }
}

function getNormalizedPosition(event) {
  const rect = els.overlayCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
  };
}

function findPointIndexAtEvent(event) {
  const position = getNormalizedPosition(event);
  const x = position.x * els.overlayCanvas.width;
  const y = position.y * els.overlayCanvas.height;
  const hitRadius = 12;
  for (let index = state.points.length - 1; index >= 0; index -= 1) {
    const point = state.points[index];
    const pointX = point.x * els.overlayCanvas.width;
    const pointY = point.y * els.overlayCanvas.height;
    if (Math.hypot(pointX - x, pointY - y) <= hitRadius) {
      return index;
    }
  }
  return -1;
}

function commitPointEdit() {
  updatePointsUI();
  saveCurrentPointKeyframe();
  updateKeyframeUI();
  redrawCanvas();
}

async function setFrame(frameIndex) {
  if (!state.videoUrl) {
    return;
  }
  hidePointContextMenu();
  saveCurrentPointKeyframe();
  state.currentFrame = clampFrame(frameIndex);
  loadCurrentPointKeyframe();
  updateFrameUI();
  els.sourceVideo.pause();
  const targetTime = state.currentFrame / state.fps;
  const seekPromise = new Promise((resolve) => {
    const onSeeked = () => {
      els.sourceVideo.removeEventListener("seeked", onSeeked);
      resolve();
    };
    els.sourceVideo.addEventListener("seeked", onSeeked, {once: true});
  });
  els.sourceVideo.currentTime = targetTime;
  await seekPromise;
  redrawCanvas();
}

async function jumpToKeyframe(direction) {
  if (!state.keyframeMode) {
    throw new Error("请先打开关键帧开关。");
  }
  saveCurrentPointKeyframe();
  const frames = knownKeyframeFrames();
  if (frames.length === 0) {
    throw new Error("还没有可跳转的关键帧。");
  }

  const target =
    direction < 0
      ? frames.filter((frame) => frame < state.currentFrame).pop()
      : frames.find((frame) => frame > state.currentFrame);
  if (target === undefined) {
    throw new Error(direction < 0 ? "当前帧前面没有关键帧。" : "当前帧后面没有关键帧。");
  }
  await setFrame(target);
  setStatus("idle", `已跳转到关键帧 ${target}。`);
}

async function deleteCurrentKeyframe() {
  if (!state.keyframeMode) {
    throw new Error("请先打开关键帧开关。");
  }
  const frameIndex = state.currentFrame;
  const frameWasConfirmed = state.keyframeFrames.includes(frameIndex);
  const key = frameKey(frameIndex);
  const hadLocalPoints = state.points.length > 0 || key in state.pointKeyframes;

  if (!frameWasConfirmed) {
    if (!hadLocalPoints) {
      throw new Error("当前帧没有可删除的关键帧点组。");
    }
    delete state.pointKeyframes[key];
    state.points = [];
    updatePointsUI();
    updateKeyframeUI();
    redrawCanvas();
    setStatus("idle", `已删除当前帧 ${frameIndex} 的点组。`);
    return;
  }

  if (!state.sessionId) {
    throw new Error("没有活动会话。");
  }
  setStatus("prompting", `正在删除关键帧 ${frameIndex}...`);
  const response = await fetch("/api/keyframe/delete", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      sessionId: state.sessionId,
      frameIndex,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "删除关键帧失败。");
  }

  delete state.pointKeyframes[key];
  state.points = [];
  updatePointsUI();
  updateKeyframeUI();
  redrawCanvas();
  await applySessionSnapshot(payload.session);
  if (payload.frameIndex !== null && payload.frameIndex !== undefined) {
    await setFrame(Number(payload.frameIndex));
  }
  const hasRemainingKeyframes = !(payload.deleted && payload.frameIndex === null);
  setStatus(
    hasRemainingKeyframes ? "prompted" : "idle",
    hasRemainingKeyframes
      ? `已删除关键帧 ${frameIndex}。`
      : `已删除关键帧 ${frameIndex}，当前没有已确认提示。`,
  );
}

async function uploadVideo(file) {
  stopPolling();
  showLoadingOverlay();
  try {
    setLoadingState("准备视频", `正在上传 ${file.name} ...`, 2);
    const payload = await uploadVideoWithProgress(file);
    state.sessionId = payload.sessionId;
    state.lastSnapshotUpdatedAt = 0;
    state.videoUrl = payload.videoUrl;
    state.videoName = file.name;
    state.frameCount = payload.frameCount;
    state.fps = payload.fps;
    state.width = payload.width;
    state.height = payload.height;
    state.samInference = payload.samInference || null;
    state.currentFrame = 0;
    state.zoom = 1;
    state.fitMode = "contain";
    resetLocalAnnotations();
    resetPromptPreview();
    updateVideoMeta();
    updateFrameUI();
    els.sourceVideo.src = state.videoUrl;
    els.videoWrapper.style.aspectRatio = `${state.width} / ${state.height}`;
    setEditorLoaded(true);
    toggleHidden(els.previewSection, true);
    setLoadingState("准备视频", "正在打开编辑器...", 100);
    setStatus(
      "idle",
      "视频已加载。请选择任意帧，并用点选、框选或文字确认目标。",
    );
    await new Promise((resolve) => {
      if (els.sourceVideo.readyState >= 1) {
        resolve();
        return;
      }
      els.sourceVideo.addEventListener("loadedmetadata", resolve, {once: true});
    });
    applyZoom({keepCenter: false});
    await setFrame(0);
    hideLoadingOverlay();
  } catch (error) {
    hideLoadingOverlay();
    throw error;
  }
}

function uploadVideoWithProgress(file) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("video", file);
    if (state.sessionId) {
      formData.append("previous_session_id", state.sessionId);
    }
    const samMaxPixels = els.samMaxPixelsInput?.value?.trim();
    if (samMaxPixels) {
      formData.append("sam_max_inference_pixels", samMaxPixels);
    }

    stopLoadingAnimation();

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload", true);
    xhr.responseType = "json";

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || event.total <= 0) {
        return;
      }
      const progress = 4 + (event.loaded / event.total) * 68;
      setLoadingState("准备视频", `正在上传 ${file.name} ...`, progress);
    };

    xhr.upload.onload = () => {
      setLoadingIndeterminate(
        "导入视频",
        "正在分析视频并启动 SAM 3.1，请稍等...",
      );
    };

    xhr.onerror = () => {
      stopLoadingAnimation();
      reject(new Error("上传失败。"));
    };

    xhr.onload = () => {
      stopLoadingAnimation();
      const payload = xhr.response ?? JSON.parse(xhr.responseText || "{}");
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(payload.error || "上传失败。"));
        return;
      }
      resolve(payload);
    };

    xhr.send(formData);
  });
}

async function fetchStatus() {
  if (!state.sessionId) {
    return;
  }
  const requestSessionId = state.sessionId;
  const response = await fetch(`/api/status?session_id=${encodeURIComponent(requestSessionId)}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "获取状态失败。");
  }
  if (state.sessionId !== requestSessionId) {
    return;
  }
  await applySessionSnapshot(payload);
}

async function applySessionSnapshot(snapshot) {
  const snapshotUpdatedAt = Number(snapshot.updatedAt || 0);
  if (snapshotUpdatedAt && snapshotUpdatedAt < state.lastSnapshotUpdatedAt) {
    return;
  }
  if (snapshotUpdatedAt) {
    state.lastSnapshotUpdatedAt = snapshotUpdatedAt;
  }

  const previousPromptMaskUrl = state.promptMaskUrl;
  state.frameCount = snapshot.frameCount;
  state.fps = snapshot.fps;
  state.width = snapshot.width;
  state.height = snapshot.height;
  state.samInference = snapshot.samInference || null;
  state.promptFrameIndex = snapshot.promptFrameIndex;
  state.promptMaskUrl = snapshot.promptMaskUrl;
  state.overlayPreviewUrl = snapshot.overlayPreviewUrl;
  state.maskPreviewUrl = snapshot.maskPreviewUrl;
  if (typeof snapshot.keyframeEnabled === "boolean") {
    state.keyframeMode = snapshot.keyframeEnabled;
  }
  state.keyframeFrames = Array.isArray(snapshot.keyframeFrames)
    ? snapshot.keyframeFrames.map((frame) => Number(frame)).filter((frame) => Number.isFinite(frame))
    : [];
  if (snapshot.maskPostprocess) {
    state.maskPostprocessMode = snapshot.maskPostprocess.mode || "videomama";
    state.videomamaMaxResolution =
      snapshot.maskPostprocess.videomamaMaxResolution ?? DEFAULT_VIDEOMAMA_MAX_RESOLUTION;
    state.vitmatteDevice = snapshot.maskPostprocess.vitmatteDevice || "gpu";
    state.trimapErodePx = snapshot.maskPostprocess.trimapErodePx ?? DEFAULT_TRIMAP_ERODE_PX;
    state.trimapDilatePx =
      snapshot.maskPostprocess.trimapDilatePx ?? DEFAULT_TRIMAP_DILATE_PX;
    els.maskPostprocessSelect.value = state.maskPostprocessMode;
    if (els.videomamaMaxResolutionInput) {
      els.videomamaMaxResolutionInput.value = String(state.videomamaMaxResolution);
    }
    if (els.vitmatteDeviceSelect) {
      els.vitmatteDeviceSelect.value = state.vitmatteDevice;
    }
    els.trimapErodeInput.value = String(state.trimapErodePx);
    els.trimapDilateInput.value = String(state.trimapDilatePx);
    syncVideomamaMaxResolution();
    syncTrimapControls();
    updateMaskPostprocessUI();
  }
  updatePromptSummary(snapshot.promptSummary || {});
  updateVideoMeta();
  setStatus(snapshot.status, snapshot.message || "");

  const percent =
    snapshot.progressTotal > 0
      ? Math.round((snapshot.progressCurrent / snapshot.progressTotal) * 100)
      : 0;
  toggleHidden(els.progressShell, false);
  els.progressFill.style.width = `${percent}%`;
  els.progressText.textContent = `${percent}%`;

  if (snapshot.promptMaskUrl && snapshot.promptMaskUrl !== previousPromptMaskUrl) {
    state.promptMaskImage = await loadMaskImage(snapshot.promptMaskUrl);
    redrawCanvas();
  }

  if (!snapshot.promptMaskUrl) {
    state.promptMaskImage = null;
    redrawCanvas();
  }

  if (snapshot.overlayPreviewUrl && snapshot.maskPreviewUrl) {
    toggleHidden(els.previewSection, false);
    if (els.overlayPreviewVideo.getAttribute("src") !== snapshot.overlayPreviewUrl) {
      resetPreviewTimeline(
        els.overlayPreviewVideo,
        els.overlayPreviewToggleButton,
        els.overlayPreviewSlider,
        els.overlayPreviewFrameText,
      );
      els.overlayPreviewVideo.src = snapshot.overlayPreviewUrl;
    }
    if (els.maskPreviewVideo.getAttribute("src") !== snapshot.maskPreviewUrl) {
      resetPreviewTimeline(
        els.maskPreviewVideo,
        els.maskPreviewToggleButton,
        els.maskPreviewSlider,
        els.maskPreviewFrameText,
      );
      els.maskPreviewVideo.src = snapshot.maskPreviewUrl;
    }
    updatePreviewTimeline(
      els.overlayPreviewVideo,
      els.overlayPreviewToggleButton,
      els.overlayPreviewSlider,
      els.overlayPreviewFrameText,
    );
    updatePreviewTimeline(
      els.maskPreviewVideo,
      els.maskPreviewToggleButton,
      els.maskPreviewSlider,
      els.maskPreviewFrameText,
    );
  }

  if (!snapshot.overlayPreviewUrl || !snapshot.maskPreviewUrl) {
    toggleHidden(els.previewSection, true);
  }

  if (snapshot.status === "completed" || snapshot.status === "error") {
    stopPolling();
  }
}

async function applyPrompt() {
  if (!state.sessionId) {
    throw new Error("请先上传视频。");
  }
  saveCurrentPointKeyframe();
  const useKeyframes = state.currentMode === "points" && state.keyframeMode;
  const payload = {
    sessionId: state.sessionId,
    frameIndex: state.currentFrame,
    mode: state.currentMode,
    points: state.points,
    box: state.box,
    textPrompt: els.textPromptInput.value,
    keyframeEnabled: useKeyframes,
  };
  if (state.currentMode === "points" && state.points.length === 0) {
    throw new Error("请先至少添加一个点。");
  }
  if (state.currentMode === "bbox" && !state.box) {
    throw new Error("请先画出框选区域。");
  }
  if (state.currentMode === "text" && !els.textPromptInput.value.trim()) {
    throw new Error("请先输入文字提示。");
  }

  setStatus(
    "prompting",
    useKeyframes
      ? "SAM 3.1 正在生成当前关键帧遮罩..."
      : "SAM 3.1 正在生成当前帧遮罩...",
  );
  const response = await fetch("/api/prompt", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "提示请求失败。");
  }
  state.promptFrameIndex = result.prompt.frameIndex;
  updatePromptSummary(result.prompt.summary);
  await applySessionSnapshot(result.session);
  setStatus(
    "prompted",
    useKeyframes
      ? `已确认关键帧 ${result.prompt.frameIndex}，现在可以继续标注或开始传播。`
      : "已确认当前帧遮罩，现在可以开始传播。",
  );
}

async function resetSession() {
  if (!state.sessionId) {
    return;
  }
  const response = await fetch("/api/reset_session", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({sessionId: state.sessionId}),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "重置失败。");
  }
  resetLocalAnnotations();
  resetPromptPreview();
  toggleHidden(els.previewSection, true);
  await applySessionSnapshot(payload);
  setStatus("idle", "选择已清空。可以在任意帧重新标注。");
}

async function startPropagation() {
  if (!state.sessionId) {
    throw new Error("请先上传视频。");
  }
  saveCurrentPointKeyframe();
  if (state.promptFrameIndex === null) {
    throw new Error("传播前请先确认当前帧遮罩。");
  }
  const useKeyframes = state.keyframeMode;
  const response = await fetch("/api/start_propagation", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      sessionId: state.sessionId,
      previewBitrate: els.previewBitrateInput.value,
      maskPostprocess: getMaskPostprocessPayload(),
      keyframeEnabled: useKeyframes,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "传播启动失败。");
  }
  await applySessionSnapshot(payload);
  setStatus(
    "propagating",
    useKeyframes
      ? "正在使用关键帧向前后双向传播遮罩，并准备预览视频。"
      : "正在向前后双向传播遮罩，并准备预览视频。",
  );
  startPolling();
}

async function exportMask() {
  if (!state.sessionId) {
    throw new Error("没有可导出的活动会话。");
  }
  const response = await fetch("/api/export_mask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      sessionId: state.sessionId,
      bitrate: els.exportBitrateInput.value,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "导出失败。");
  }
  const anchor = document.createElement("a");
  anchor.href = payload.downloadUrl;
  anchor.download = payload.fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  hideContextMenu();
  setStatus("completed", `已导出 ${payload.fileName}`);
}

function hideContextMenu() {
  toggleHidden(els.maskContextMenu, true);
}

function showContextMenu(x, y) {
  hidePointContextMenu();
  els.maskContextMenu.style.left = `${x}px`;
  els.maskContextMenu.style.top = `${y}px`;
  toggleHidden(els.maskContextMenu, false);
}

function hidePointContextMenu() {
  state.pointContextIndex = null;
  toggleHidden(els.pointContextMenu, true);
}

function showPointContextMenu(pointIndex, x, y) {
  hideContextMenu();
  state.pointContextIndex = pointIndex;
  els.pointContextMenu.style.left = `${x}px`;
  els.pointContextMenu.style.top = `${y}px`;
  toggleHidden(els.pointContextMenu, false);
}

function toggleContextPointLabel() {
  const point = state.points[state.pointContextIndex];
  if (!point) {
    hidePointContextMenu();
    return;
  }
  point.label = point.label === 1 ? 0 : 1;
  commitPointEdit();
  hidePointContextMenu();
}

function deleteContextPoint() {
  if (!state.points[state.pointContextIndex]) {
    hidePointContextMenu();
    return;
  }
  state.points.splice(state.pointContextIndex, 1);
  commitPointEdit();
  hidePointContextMenu();
}

function handleFileDrop(event) {
  event.preventDefault();
  els.dropZone.classList.remove("dragover");
  const file = event.dataTransfer.files?.[0];
  if (file) {
    uploadVideo(file).catch((error) => setStatus("error", error.message));
  }
}

els.pickVideoButton.addEventListener("click", () => els.videoInput.click());
els.replaceVideoButton.addEventListener("click", () => els.videoInput.click());
els.zoomInButton.addEventListener("click", () => {
  state.fitMode = state.fitMode || "contain";
  state.zoom = Math.min(5, Number((state.zoom + 0.25).toFixed(2)));
  applyZoom({keepCenter: true});
});
els.zoomOutButton.addEventListener("click", () => {
  state.fitMode = state.fitMode || "contain";
  state.zoom = Math.max(1, Number((state.zoom - 0.25).toFixed(2)));
  applyZoom({keepCenter: true});
});
els.zoomFitButton.addEventListener("click", () => {
  state.fitMode = "contain";
  state.zoom = 1;
  applyZoom({keepCenter: false});
});
els.zoomFitWidthButton.addEventListener("click", () => {
  state.fitMode = "width";
  state.zoom = 1;
  applyZoom({keepCenter: false});
});
els.zoomFitHeightButton.addEventListener("click", () => {
  state.fitMode = "height";
  state.zoom = 1;
  applyZoom({keepCenter: false});
});
els.videoInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) {
    uploadVideo(file).catch((error) => setStatus("error", error.message));
  }
});

["dragenter", "dragover"].forEach((name) => {
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((name) => {
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragover");
  });
});
els.dropZone.addEventListener("drop", handleFileDrop);

els.videoViewport.addEventListener("wheel", (event) => {
  if (!state.videoUrl) {
    return;
  }
  event.preventDefault();
  const rect = els.videoViewport.getBoundingClientRect();
  const anchor = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
  const delta = event.deltaY < 0 ? 0.12 : -0.12;
  state.zoom = Math.max(1, Math.min(5, Number((state.zoom + delta).toFixed(2))));
  applyZoom({anchor});
}, {passive: false});

els.videoViewport.addEventListener("mousedown", (event) => {
  if (event.button !== 1 || !state.videoUrl) {
    return;
  }
  event.preventDefault();
  state.isPanning = true;
  state.panStartX = event.clientX;
  state.panStartY = event.clientY;
  state.panStartScrollLeft = els.videoViewport.scrollLeft;
  state.panStartScrollTop = els.videoViewport.scrollTop;
  els.videoViewport.classList.add("panning");
});

els.videoViewport.addEventListener("auxclick", (event) => {
  if (event.button === 1) {
    event.preventDefault();
  }
});

els.modeTabs.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
els.keyframeModeInput.addEventListener("change", () => {
  setKeyframeMode(els.keyframeModeInput.checked);
});
els.prevKeyframeButton.addEventListener("click", () => {
  jumpToKeyframe(-1).catch((error) => setStatus("error", error.message));
});
els.nextKeyframeButton.addEventListener("click", () => {
  jumpToKeyframe(1).catch((error) => setStatus("error", error.message));
});
els.deleteKeyframeButton.addEventListener("click", () => {
  deleteCurrentKeyframe().catch((error) => setStatus("error", error.message));
});
els.maskPostprocessSelect.addEventListener("change", updateMaskPostprocessUI);
els.videomamaMaxResolutionInput.addEventListener("change", syncVideomamaMaxResolution);
els.trimapErodeInput.addEventListener("input", syncTrimapControls);
els.trimapDilateInput.addEventListener("input", syncTrimapControls);
els.pointButtons.forEach((button) => {
  button.addEventListener("click", () => setPointLabel(Number(button.dataset.pointLabel)));
});

els.frameSlider.addEventListener("input", () => {
  setFrame(Number(els.frameSlider.value)).catch((error) =>
    setStatus("error", error.message),
  );
});
els.frameNumberInput.addEventListener("change", () => {
  setFrame(Number(els.frameNumberInput.value)).catch((error) =>
    setStatus("error", error.message),
  );
});
els.prevFrameButton.addEventListener("click", () => {
  setFrame(state.currentFrame - 1).catch((error) => setStatus("error", error.message));
});
els.nextFrameButton.addEventListener("click", () => {
  setFrame(state.currentFrame + 1).catch((error) => setStatus("error", error.message));
});

els.overlayCanvas.addEventListener("contextmenu", (event) => {
  if (state.currentMode === "points" && state.videoUrl) {
    event.preventDefault();
    const pointIndex = findPointIndexAtEvent(event);
    if (pointIndex >= 0) {
      showPointContextMenu(pointIndex, event.clientX, event.clientY);
      return;
    }
    hidePointContextMenu();
    hideContextMenu();
    const point = getNormalizedPosition(event);
    state.points.push({...point, label: 0});
    commitPointEdit();
  }
});

els.overlayCanvas.addEventListener("pointerdown", (event) => {
  if (!state.videoUrl) {
    return;
  }
  if (state.currentMode === "bbox") {
    if (event.button !== 0) {
      return;
    }
    state.isDrawingBox = true;
    const start = getNormalizedPosition(event);
    state.draftBox = {
      x: start.x,
      y: start.y,
      w: 0,
      h: 0,
      startX: start.x,
      startY: start.y,
    };
    redrawCanvas();
    return;
  }
  if (state.currentMode === "points" && event.button === 0) {
    const point = getNormalizedPosition(event);
    state.points.push({...point, label: state.currentPointLabel});
    updatePointsUI();
    saveCurrentPointKeyframe();
    redrawCanvas();
  }
});

els.overlayCanvas.addEventListener("pointermove", (event) => {
  if (!state.isDrawingBox || state.currentMode !== "bbox" || !state.draftBox) {
    return;
  }
  const point = getNormalizedPosition(event);
  const x1 = Math.min(state.draftBox.startX, point.x);
  const y1 = Math.min(state.draftBox.startY, point.y);
  const x2 = Math.max(state.draftBox.startX, point.x);
  const y2 = Math.max(state.draftBox.startY, point.y);
  state.draftBox = {
    ...state.draftBox,
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
  };
  redrawCanvas();
});

window.addEventListener("pointermove", (event) => {
  if (!state.isPanning) {
    return;
  }
  const dx = event.clientX - state.panStartX;
  const dy = event.clientY - state.panStartY;
  els.videoViewport.scrollLeft = state.panStartScrollLeft - dx;
  els.videoViewport.scrollTop = state.panStartScrollTop - dy;
});

window.addEventListener("pointerup", () => {
  if (state.isPanning) {
    state.isPanning = false;
    els.videoViewport.classList.remove("panning");
  }
  if (!state.isDrawingBox) {
    return;
  }
  state.isDrawingBox = false;
  if (state.draftBox && state.draftBox.w > 0.005 && state.draftBox.h > 0.005) {
    state.box = {
      x: state.draftBox.x,
      y: state.draftBox.y,
      w: state.draftBox.w,
      h: state.draftBox.h,
    };
    els.bboxSummaryText.textContent =
      `左=${state.box.x.toFixed(3)}, 上=${state.box.y.toFixed(3)}, 宽=${state.box.w.toFixed(3)}, 高=${state.box.h.toFixed(3)}`;
  }
  state.draftBox = null;
  redrawCanvas();
});

els.applyPromptButton.addEventListener("click", () => {
  applyPrompt().catch((error) => setStatus("error", error.message));
});
els.resetSessionButton.addEventListener("click", () => {
  resetSession().catch((error) => setStatus("error", error.message));
});
els.propagateButton.addEventListener("click", () => {
  startPropagation().catch((error) => setStatus("error", error.message));
});

els.maskPreviewCard.addEventListener("contextmenu", (event) => {
  if (!state.maskPreviewUrl) {
    return;
  }
  event.preventDefault();
  showContextMenu(event.clientX, event.clientY);
});

previewTimelines.forEach(({video, toggleButton, slider, frameText}) => {
  bindPreviewTimeline(video, toggleButton, slider, frameText);
});

els.exportMaskButton.addEventListener("click", () => {
  exportMask().catch((error) => setStatus("error", error.message));
});

els.togglePointButton.addEventListener("click", () => {
  toggleContextPointLabel();
});
els.deletePointButton.addEventListener("click", () => {
  deleteContextPoint();
});

document.addEventListener("click", (event) => {
  if (!els.maskContextMenu.contains(event.target)) {
    hideContextMenu();
  }
  if (!els.pointContextMenu.contains(event.target)) {
    hidePointContextMenu();
  }
});

window.addEventListener("resize", () => {
  applyZoom({keepCenter: false});
});
setEditorLoaded(false);
setMode("points");
setPointLabel(1);
updateMaskPostprocessUI();
syncVideomamaMaxResolution();
syncTrimapControls();
updateKeyframeUI();
updateZoomUI();
setStatus("idle", "拖入视频即可开始。");
