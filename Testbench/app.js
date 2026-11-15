'use strict';

/*
 * AI Eyes 测试台前端逻辑。
 * 页面只通过同源 HTTP API 获取状态与事件；本文件不包含任何内置对话、假指标或假日志。
 * 显示文本一律使用 textContent / createElement 渲染。
 */
(function () {
  // ---------------------------------------------------------------- 常量
  const POLL_INTERVAL_MS = 500;
  const PREVIEW_INTERVAL_MS = 250;
  const PREVIEW_ERROR_INTERVAL_MS = 1000;
  const HEARTBEAT_INTERVAL_MS = 5000;
  const MAX_RECONNECT_DELAY_MS = 5000;
  const REQUEST_TIMEOUT_MS = 5000;
  const POLL_TIMEOUT_MS = 4000;
  const FRAME_TIMEOUT_MS = 10000;
  const UPLOAD_TIMEOUT_MS = 120000;
  const MAX_LOG_ENTRIES = 1000;
  const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
  const IMAGE_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
  const ACTIVE_RUN_STATUSES = ['starting', 'running', 'stopping'];

  const API = {
    session: '/api/session',
    state: '/api/state',
    events: '/api/events',
    heartbeat: '/api/client/heartbeat',
    runStart: '/api/run/start',
    runStop: '/api/run/stop',
    previewStart: '/api/preview/start',
    previewStop: '/api/preview/stop',
    previewJpg: '/api/preview.jpg',
    uploads: '/api/uploads',
    uploadImage: function (id) { return '/api/uploads/' + encodeURIComponent(String(id)); },
    frame: function (id) { return '/api/frames/' + encodeURIComponent(String(id)); },
    debugModules: '/api/debug/modules',
    debugStart: '/api/debug/start',
    debugStop: '/api/debug/stop',
    debugAction: '/api/debug/action'
  };

  // ---------------------------------------------------------------- 工具
  function $(selector, root) {
    return (root || document).querySelector(selector);
  }

  function $$(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function includes(list, value) {
    return list.indexOf(value) !== -1;
  }

  function clampPercent(value) {
    if (typeof value !== 'number' || !isFinite(value)) return null;
    if (value < 0) return 0;
    if (value > 100) return 100;
    return value;
  }

  function toMillis(value) {
    if (typeof value === 'number' && isFinite(value)) return value;
    if (typeof value === 'string') {
      const parsed = Date.parse(value);
      if (!isNaN(parsed)) return parsed;
    }
    return null;
  }

  function formatDuration(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
  }

  function formatClock(value) {
    const ms = toMillis(value);
    const date = ms === null ? new Date() : new Date(ms);
    return date.toLocaleTimeString('zh-CN', { hour12: false });
  }

  function formatBytes(value) {
    if (typeof value !== 'number' || !isFinite(value) || value < 0) return '—';
    const mib = value / (1024 * 1024);
    if (mib >= 1024) return (mib / 1024).toFixed(2) + ' GB';
    if (mib >= 10) return Math.round(mib) + ' MB';
    return mib.toFixed(1) + ' MB';
  }

  function formatPercent(value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return Math.round(value) + '%';
  }

  function formatTemperature(value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return Math.round(value) + ' °C';
  }

  function formatTiming(value, emptyText) {
    if (typeof value !== 'number' || !isFinite(value)) return emptyText || '—';
    return Math.round(value) + ' ms';
  }

  function makeRequestId() {
    try {
      if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
      }
    } catch (err) {
      /* 回退到本地生成的请求标识 */
    }
    return 'req-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  }

  // ---------------------------------------------------------------- DOM 引用
  const dom = {
    shell: $('.shell'),
    sidebarHeader: $('#sidebarHeader'),
    navButtons: $$('nav button'),
    integrationPage: $('#integration'),
    debugPage: $('#debug'),
    svcInteraction: $('#svcInteraction'),
    svcModel: $('#svcModel'),
    svcAudio: $('#svcAudio'),
    svcCamera: $('#svcCamera'),
    mediaTabs: $$('[data-media]'),
    cameraPanel: $('#cameraPanel'),
    uploadPanel: $('#uploadPanel'),
    modelFrame: $('#modelFrame'),
    modelFrameImg: $('#modelFrameImg'),
    modelFrameStatus: $('#modelFrameStatus'),
    modelFrameMeta: $('#modelFrameMeta'),
    modelFrameIcon: $('#modelFrame .camera-icon'),
    frameSlots: $('#frameSlots'),
    frameSlotButtons: $$('#frameSlots button'),
    previewFrame: $('#previewFrame'),
    previewImg: $('#previewImg'),
    previewIcon: $('#previewFrame .camera-icon'),
    previewStatus: $('#cameraStatus'),
    cameraPreview: $('#cameraPreview'),
    runstate: $('#runstate'),
    conversation: $('#conversation'),
    conversationStatus: $('#conversationStatus'),
    start: $('#start'),
    stop: $('#stop'),
    elapsed: $('#elapsed'),
    modules: $('#modules'),
    moduleTitle: $('#moduleTitle'),
    moduleDesc: $('#moduleDesc'),
    debugNotice: $('#debugNotice'),
    debugForm: $('#debugForm'),
    debugActions: $('#debugActions'),
    debugOutput: $('#debugOutput'),
    debugRun: $('#debugRun'),
    debugStop: $('#debugStop'),
    debugTime: $('#debugTime'),
    timings: {
      asr: $('#timingAsr'),
      vision: $('#timingVision'),
      generation: $('#timingGeneration'),
      tts: $('#timingTts'),
      playbackWait: $('#timingPlaybackWait'),
      acoustic: $('#timingAcoustic')
    },
    metrics: {
      rssRow: $('#metricRssRow'),
      rss: $('#metricRss'),
      barRss: $('#barRss'),
      cpuRow: $('#metricCpuRow'),
      cpu: $('#metricCpu'),
      barCpu: $('#barCpu'),
      gpuRow: $('#metricGpuRow'),
      gpu: $('#metricGpu'),
      barGpu: $('#barGpu'),
      temp: $('#metricTemp'),
      bench: $('#metricBench'),
      status: $('#resourceStatus')
    },
    logs: $('#logs'),
    logFilter: $('#logFilter'),
    clearLogs: $('#clearLogs'),
    confirm: $('#confirm'),
    confirmTitle: $('#confirmTitle'),
    confirmBody: $('#confirmBody'),
    confirmStart: $('#confirmStart'),
    cancelStart: $('#cancelStart')
  };

  const uploadSlots = {};
  $$('.upload-slot').forEach(function (slotElement) {
    const name = slotElement.dataset.slot;
    if (name !== 'A' && name !== 'B') return;
    uploadSlots[name] = {
      name: name,
      root: slotElement,
      box: $('.upload-box', slotElement),
      img: $('.upload-box img', slotElement),
      placeholder: $('.upload-box > span', slotElement),
      filename: $('.filename', slotElement),
      pick: $('.pick', slotElement),
      remove: $('.remove', slotElement),
      input: $('.image-file', slotElement),
      error: $('.upload-error', slotElement),
      imageId: null,
      uploading: false
    };
  });

  // ---------------------------------------------------------------- 运行状态
  const app = {
    clientId: null,
    token: null,
    lastSeq: 0,
    sessionReady: false,
    connected: false,
    everConnected: false,
    connectionAttempted: false,
    owner: true,
    controlLocked: false,
    serverOffsetMs: 0,
    state: null,
    run: { id: null, mode: null, status: 'idle', started_at: null, elapsed_ms: null },
    interaction: 'idle',
    timings: {},
    previewActive: false,
    previewPolling: false,
    previewTimer: null,
    previewUrl: null,
    previewGraceUntil: 0,
    previewErrorUntil: 0,
    pendingPreview: false,
    frames: { model: null, A: null, B: null },
    frameSelected: 'model',
    frameLoadSeq: { model: 0, A: 0, B: 0 },
    turns: new Map(),
    playbackPieces: new Set(),
    playbackBubbles: new Map(),
    playbackStarts: new Map(),
    statusMessage: null,
    pendingRun: null,
    pendingConfirm: null,
    activePage: 'integration',
    modules: [],
    modulesLoaded: false,
    selectedModuleId: null,
    selectedModule: null,
    debugRun: null,
    debugStopping: false,
    debugGraceUntil: 0,
    debugTimer: null,
    lastDebugResult: null,
    finishedDebugRuns: new Set(),
    debugPlaybackStarts: new Map(),
    uploadsInitialized: false,
    logs: []
  };

  // ---------------------------------------------------------------- HTTP
  function ApiError(status, code, message, retryable) {
    const error = new Error(message || '请求失败');
    error.name = 'ApiError';
    error.status = status;
    error.code = code || 'http_error';
    error.retryable = !!retryable;
    return error;
  }

  function isNetworkError(error) {
    return error && error.name === 'ApiError' && error.status === 0;
  }

  async function readJsonSafe(response) {
    try {
      return await response.json();
    } catch (err) {
      return null;
    }
  }

  async function readErrorDetail(response) {
    try {
      const text = await response.text();
      if (!text) return '';
      try {
        const payload = JSON.parse(text);
        if (payload && payload.error && payload.error.message) return String(payload.error.message);
      } catch (err) {
        /* 非 JSON 错误体按文本处理 */
      }
      return text.slice(0, 200);
    } catch (err) {
      return '';
    }
  }

  function authHeaders(extra) {
    const headers = Object.assign({}, extra || {});
    if (app.token) headers['X-Bench-Token'] = app.token;
    if (app.clientId) headers['X-Bench-Client'] = app.clientId;
    return headers;
  }

  async function request(path, options) {
    const opts = options || {};
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timeout = typeof opts.timeout === 'number' ? opts.timeout : REQUEST_TIMEOUT_MS;
    const timer = controller ? setTimeout(function () { controller.abort(); }, timeout) : null;
    try {
      const response = await fetch(path, {
        method: opts.method || 'GET',
        headers: opts.headers || undefined,
        body: opts.body,
        cache: 'no-store',
        credentials: 'same-origin',
        signal: controller ? controller.signal : undefined
      });
      const payload = await readJsonSafe(response);
      if (!response.ok) {
        const info = payload && payload.error ? payload.error : null;
        throw ApiError(
          response.status,
          info && info.code ? info.code : 'http_' + response.status,
          info && info.message ? info.message : '请求失败（HTTP ' + response.status + '）',
          info ? info.retryable : true
        );
      }
      return { response: response, data: payload };
    } catch (error) {
      if (error && error.name === 'ApiError') throw error;
      throw ApiError(0, 'network', '未连接', true);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  function jsonPost(path, payload, timeout) {
    return request(path, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload || {}),
      timeout: timeout || REQUEST_TIMEOUT_MS
    });
  }

  function postWithoutBody(path) {
    return request(path, { method: 'POST', headers: authHeaders() });
  }

  function jsonDelete(path) {
    return request(path, { method: 'DELETE', headers: authHeaders() });
  }

  async function postBinary(path, blob, mimeType, timeout) {
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = controller ? setTimeout(function () { controller.abort(); }, timeout || UPLOAD_TIMEOUT_MS) : null;
    try {
      const response = await fetch(path, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': mimeType }),
        body: blob,
        cache: 'no-store',
        credentials: 'same-origin',
        signal: controller ? controller.signal : undefined
      });
      const payload = await readJsonSafe(response);
      if (!response.ok) {
        const info = payload && payload.error ? payload.error : null;
        throw ApiError(
          response.status,
          info && info.code ? info.code : 'http_' + response.status,
          info && info.message ? info.message : '上传失败（HTTP ' + response.status + '）',
          info ? info.retryable : true
        );
      }
      return payload;
    } catch (error) {
      if (error && error.name === 'ApiError') throw error;
      throw ApiError(0, 'network', '未连接', true);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  async function fetchRaw(path, timeout) {
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = controller ? setTimeout(function () { controller.abort(); }, timeout || REQUEST_TIMEOUT_MS) : null;
    try {
      return await fetch(path, {
        cache: 'no-store',
        credentials: 'same-origin',
        signal: controller ? controller.signal : undefined
      });
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  // ---------------------------------------------------------------- 连接与状态提示
  function readOnly() {
    return !app.connected || !app.owner || app.controlLocked;
  }

  function runIsActive() {
    return includes(ACTIVE_RUN_STATUSES, app.run && app.run.status);
  }

  function debugIsActive() {
    return !!app.debugRun;
  }

  function showStatusMessage(text, duration, kind) {
    app.statusMessage = {
      text: text,
      until: Date.now() + (duration || 6000),
      kind: kind || 'error'
    };
    renderConversationStatus();
  }

  function clearProgressMessage() {
    if (app.statusMessage && app.statusMessage.kind === 'progress') {
      app.statusMessage = null;
    }
  }

  function setConnected(value) {
    if (app.connected === value) return;
    app.connected = value;
    if (value) {
      app.everConnected = true;
      app.previewErrorUntil = 0;
      if (app.previewActive) startPreviewPolling();
    }
    if (!value) {
      stopPreviewPolling();
      app.previewErrorUntil = Date.now() + 5000;
      setPreviewStatus('未连接', true);
    }
    renderServices();
    renderRunState();
    renderMetrics();
    renderButtons();
    renderConversationStatus();
    renderCameraState();
  }

  function setOwner(value) {
    const next = value !== false;
    if (app.owner === next) return;
    app.owner = next;
    renderButtons();
    renderConversationStatus();
  }

  function defaultConversationStatus() {
    if (!app.connected) {
      if (!app.everConnected && !app.connectionAttempted) return '正在连接…';
      return '未连接，重试中';
    }
    if (!app.owner || app.controlLocked) return '另一窗口正在控制，当前页面只读';
    if (runIsActive()) {
      if (app.run.status === 'starting') return '测试启动中…';
      if (app.run.status === 'stopping') return '测试停止中…';
      return '测试运行中';
    }
    if (app.previewActive) return '预览开启中；开始测试前请先停止预览';
    if (debugIsActive()) return '功能调试运行中；开始测试前请先停止调试';
    if (app.run.status === 'completed') return '本轮已完成';
    if (app.run.status === 'interrupted') return '本轮已中断';
    if (app.run.status === 'failed') return '本轮失败';
    if (app.interaction === 'failed') return '交互服务异常';
    return '已连接，等待开始';
  }

  function renderConversationStatus() {
    if (!dom.conversationStatus) return;
    // 连接/控制状态优先级最高，避免过期错误提示掩盖"未连接/只读"。
    if (!app.connected || !app.owner || app.controlLocked) {
      dom.conversationStatus.textContent = defaultConversationStatus();
      return;
    }
    const message = app.statusMessage;
    if (message && message.until > Date.now()) {
      dom.conversationStatus.textContent = message.text;
      return;
    }
    if (message) app.statusMessage = null;
    dom.conversationStatus.textContent = defaultConversationStatus();
  }

  // ---------------------------------------------------------------- 服务状态卡
  function setService(element, text, tone) {
    if (!element) return;
    const dot = document.createElement('i');
    dot.className = 'dot' + (tone === 'ok' ? '' : ' gray') + (tone === 'bad' ? ' bad' : '');
    let className = 'neutral';
    if (tone === 'ok') className = '';
    if (tone === 'bad') className = 'bad';
    element.className = className;
    element.replaceChildren(dot, document.createTextNode(text));
  }

  function interactionPresentation(interaction) {
    switch (interaction) {
      case 'preview': return ['预览中', 'ok'];
      case 'running': return ['测试运行中', 'ok'];
      case 'debug': return ['调试中', 'ok'];
      case 'stopping': return ['停止中', 'neutral'];
      case 'failed': return ['异常', 'bad'];
      default: return ['空闲', 'neutral'];
    }
  }

  function renderServices() {
    if (!app.connected) {
      setService(dom.svcInteraction, '未连接', 'neutral');
      setService(dom.svcModel, '未连接', 'neutral');
      setService(dom.svcAudio, '未连接', 'neutral');
      setService(dom.svcCamera, '未连接', 'neutral');
      return;
    }
    const state = app.state;
    if (!state) {
      setService(dom.svcInteraction, '未知', 'neutral');
      setService(dom.svcModel, '未知', 'neutral');
      setService(dom.svcAudio, '未知', 'neutral');
      setService(dom.svcCamera, '未知', 'neutral');
      return;
    }

    const interaction = interactionPresentation((state.services && state.services.interaction) || 'idle');
    setService(dom.svcInteraction, interaction[0], interaction[1]);

    const model = state.services && state.services.model;
    if (model === 'ready') setService(dom.svcModel, '就绪', 'ok');
    else if (model === 'unavailable') setService(dom.svcModel, '不可用', 'bad');
    else setService(dom.svcModel, '未知', 'neutral');

    const devices = state.devices || {};
    const audioIn = devices.audio_in || {};
    const audioOut = devices.audio_out || {};
    const detailText = [audioIn.detail, audioOut.detail].filter(Boolean).join(' · ');
    if (audioIn.status === 'ready' && audioOut.status === 'ready') {
      setService(dom.svcAudio, '就绪', 'ok');
    } else if (audioIn.status === 'busy' || audioOut.status === 'busy') {
      setService(dom.svcAudio, '占用中', 'neutral');
    } else if (audioIn.status === 'error' || audioOut.status === 'error') {
      setService(dom.svcAudio, detailText || '异常', 'bad');
    } else {
      setService(dom.svcAudio, '未验证', 'neutral');
    }
    dom.svcAudio.title = [
      '麦克风：' + statusLabel(audioIn.status, audioIn.detail),
      '扬声器：' + statusLabel(audioOut.status, audioOut.detail)
    ].join('\n');

    const camera = devices.camera || {};
    if (app.previewActive) setService(dom.svcCamera, '预览中', 'ok');
    else if (camera.status === 'ready') setService(dom.svcCamera, '就绪', 'ok');
    else if (camera.status === 'error') setService(dom.svcCamera, '异常', 'bad');
    else if (camera.status === 'closed') setService(dom.svcCamera, '已关闭', 'neutral');
    else setService(dom.svcCamera, '未知', 'neutral');
    dom.svcCamera.title = '相机：' + statusLabel(camera.status, camera.detail);
  }

  function statusLabel(status, detail) {
    const labels = {
      ready: '就绪',
      error: '异常',
      busy: '占用',
      unknown: '未知',
      closed: '已关闭',
      unavailable: '不可用'
    };
    const label = labels[status] || '未知';
    return detail ? label + '（' + detail + '）' : label;
  }

  // ---------------------------------------------------------------- 运行状态与按钮
  function setRunStateLabel(text, tone) {
    if (!dom.runstate) return;
    const dot = document.createElement('i');
    dot.className = 'dot' + (tone === 'ok' ? '' : ' gray') + (tone === 'bad' ? ' bad' : '');
    dom.runstate.className = 'runstate';
    dom.runstate.replaceChildren(dot, document.createTextNode(text));
  }

  function renderRunState() {
    if (!app.connected) {
      setRunStateLabel('未连接', 'neutral');
      return;
    }
    const status = (app.run && app.run.status) || 'idle';
    switch (status) {
      case 'starting': setRunStateLabel('启动中', 'neutral'); break;
      case 'running': setRunStateLabel('运行中', 'ok'); break;
      case 'stopping': setRunStateLabel('停止中', 'neutral'); break;
      case 'completed': setRunStateLabel('已完成', 'neutral'); break;
      case 'interrupted': setRunStateLabel('已中断', 'bad'); break;
      case 'failed': setRunStateLabel('失败', 'bad'); break;
      default: setRunStateLabel('未开始', 'neutral'); break;
    }
  }

  function serverNow() {
    return Date.now() + (app.serverOffsetMs || 0);
  }

  function currentElapsedMs() {
    const run = app.run || {};
    const status = run.status || 'idle';
    if (status === 'idle') return null;
    const startedAt = toMillis(run.started_at);
    if (includes(ACTIVE_RUN_STATUSES, status) && startedAt !== null) {
      return Math.max(0, serverNow() - startedAt);
    }
    const elapsed = toMillis(run.elapsed_ms);
    if (elapsed !== null) return elapsed;
    if (startedAt !== null) return Math.max(0, serverNow() - startedAt);
    return null;
  }

  function renderElapsed() {
    const elapsed = currentElapsedMs();
    dom.elapsed.textContent = '本轮时长 ' + (elapsed === null ? '00:00' : formatDuration(elapsed));
  }

  function startBlockedReason() {
    if (!app.connected) return '未连接，无法开始测试';
    if (!app.owner || app.controlLocked) return '另一窗口正在控制，当前页面只读';
    if (runIsActive() || app.pendingRun) return '测试正在进行中';
    if (app.previewActive) return '预览开启中，请先停止预览';
    if (debugIsActive() || app.interaction === 'debug') return '功能调试运行中，请先停止调试';
    return '当前状态无法开始测试';
  }

  function canStartRun() {
    if (readOnly()) return false;
    if (runIsActive() || app.pendingRun) return false;
    if (debugIsActive() || app.interaction === 'debug') return false;
    if (app.previewActive) return false;
    return true;
  }

  function canStopRun() {
    if (readOnly()) return false;
    if (app.run && (app.run.status === 'starting' || app.run.status === 'running')) return true;
    return !!(app.pendingRun && app.pendingRun.kind === 'start' && app.pendingRun.runId);
  }

  function canTogglePreview() {
    if (readOnly() || app.pendingPreview) return false;
    if (app.previewActive) return true;
    if (runIsActive() || debugIsActive() || app.interaction === 'debug') return false;
    return true;
  }

  function debugStartBlockedReason() {
    if (!app.connected) return '未连接，无法运行调试';
    if (!app.owner || app.controlLocked) return '另一窗口正在控制，当前页面只读';
    if (runIsActive()) return '综合测试运行中，无法运行调试';
    if (app.previewActive) return '预览开启中，请先停止预览';
    if (debugIsActive()) return '调试运行中';
    if (!app.selectedModule) return '请先选择调试模块';
    const reason = moduleUnsupportedReason(app.selectedModule);
    if (reason) return reason;
    return '';
  }

  function canStartDebug() {
    return !debugStartBlockedReason();
  }

  function canStopDebug() {
    return !readOnly() && debugIsActive() && !app.debugStopping;
  }

  function renderButtons() {
    const blocked = readOnly();

    const startEnabled = canStartRun();
    dom.start.disabled = !startEnabled;
    dom.start.title = startEnabled ? '开始综合测试' : startBlockedReason();

    const stopEnabled = canStopRun();
    dom.stop.disabled = !stopEnabled;
    dom.stop.title = stopEnabled ? '停止本轮测试' : '当前没有可停止的运行';

    let previewDisabled = blocked || app.pendingPreview;
    let previewTitle = blocked ? '未连接或只读，无法操作相机' : '';
    if (!blocked && !app.previewActive) {
      if (runIsActive()) {
        previewDisabled = true;
        previewTitle = '综合测试运行中，无法开启预览';
      } else if (debugIsActive() || app.interaction === 'debug') {
        previewDisabled = true;
        previewTitle = '功能调试运行中，无法开启预览';
      }
    }
    if (!previewDisabled) {
      previewTitle = app.previewActive ? '停止预览并释放相机' : '查看目标机摄像头画面，不触发取图';
    }
    dom.cameraPreview.disabled = previewDisabled;
    dom.cameraPreview.title = previewTitle;
    dom.cameraPreview.textContent = app.previewActive ? '停止预览' : '开启预览';
    dom.cameraPreview.setAttribute('aria-pressed', app.previewActive ? 'true' : 'false');

    const debugReason = debugStartBlockedReason();
    dom.debugRun.disabled = !!debugReason;
    dom.debugRun.title = debugReason || '开始调试当前模块';
    const debugStopEnabled = canStopDebug();
    dom.debugStop.disabled = !debugStopEnabled;
    dom.debugStop.title = debugStopEnabled ? '停止当前调试' : '当前没有可停止的调试';

    $$('button', dom.modules).forEach(function (button) {
      button.disabled = blocked || debugIsActive();
    });
    $$('button', dom.debugActions).forEach(function (button) {
      button.disabled = blocked;
    });

    Object.keys(uploadSlots).forEach(function (name) {
      const slot = uploadSlots[name];
      slot.pick.disabled = blocked || slot.uploading;
      slot.remove.disabled = blocked || slot.uploading || !slot.imageId;
    });

    dom.confirmStart.disabled = blocked;
  }

  // ---------------------------------------------------------------- 指标与耗时
  function setBar(element, percent) {
    if (!element) return;
    if (percent === null || typeof percent !== 'number' || !isFinite(percent)) {
      element.style.width = '0%';
      return;
    }
    element.style.width = Math.max(0, Math.min(100, percent)) + '%';
  }

  function renderMetrics() {
    const state = app.connected ? app.state : null;
    const metrics = state && state.metrics ? state.metrics : null;
    const detail = metrics && metrics.detail ? metrics.detail : {};
    const online = !!metrics;

    dom.metrics.rss.textContent = online ? formatBytes(metrics.worker_rss_bytes) : '—';
    dom.metrics.cpu.textContent = online ? formatPercent(metrics.cpu_percent) : '—';
    dom.metrics.gpu.textContent = online ? formatPercent(metrics.gpu_percent) : '—';
    dom.metrics.temp.textContent = online ? formatTemperature(metrics.temperature_c) : '—';
    dom.metrics.bench.textContent = online ? formatBytes(metrics.bench_rss_bytes) : '—';

    setBar(dom.metrics.barCpu, online ? clampPercent(metrics.cpu_percent) : null);
    setBar(dom.metrics.barGpu, online ? clampPercent(metrics.gpu_percent) : null);

    let memoryShare = null;
    if (online) {
      const parts = [metrics.worker_rss_bytes, metrics.model_rss_bytes, metrics.bench_rss_bytes]
        .filter(function (value) { return typeof value === 'number' && value > 0; });
      const total = parts.reduce(function (sum, value) { return sum + value; }, 0);
      if (total > 0 && typeof metrics.worker_rss_bytes === 'number') {
        memoryShare = (metrics.worker_rss_bytes / total) * 100;
      }
    }
    setBar(dom.metrics.barRss, memoryShare);

    const tooltip = [
      '交互 worker RSS：' + (online ? formatBytes(metrics.worker_rss_bytes) : '—'),
      '模型 RSS：' + (online ? formatBytes(metrics.model_rss_bytes) : '—'),
      '测试台自身 RSS：' + (online ? formatBytes(metrics.bench_rss_bytes) : '—'),
      '',
      '交互 worker CPU：' + (online ? formatPercent(detail.worker_cpu_percent) : '—'),
      '模型 CPU：' + (online ? formatPercent(detail.model_cpu_percent) : '—'),
      '测试台 CPU：' + (online ? formatPercent(detail.bench_cpu_percent) : '—'),
      'CPU 为全核归一化 0-100',
      '内存条形为交互进程 RSS 占已测量进程 RSS 的比例，不是容量上限'
    ].join('\n');
    dom.metrics.rssRow.title = tooltip;
    dom.metrics.cpuRow.title = tooltip;

    if (!app.connected) {
      dom.metrics.status.textContent = '未连接';
    } else if (metrics && typeof metrics.sampled_at === 'number') {
      dom.metrics.status.textContent = '采样 ' + formatClock(metrics.sampled_at);
    } else {
      dom.metrics.status.textContent = '等待采样';
    }
  }

  function renderTimings() {
    const timings = app.timings || {};
    dom.timings.asr.textContent = formatTiming(timings.asr_ms);
    dom.timings.vision.textContent = formatTiming(timings.vision_ms);
    dom.timings.generation.textContent = formatTiming(timings.generation_ms);
    dom.timings.tts.textContent = formatTiming(timings.tts_ms);
    dom.timings.playbackWait.textContent = formatTiming(timings.playback_wait_ms);
    dom.timings.acoustic.textContent = formatTiming(timings.acoustic_first_audio_ms, '未测量');
  }

  // ---------------------------------------------------------------- 会话
  function scrollConversation() {
    dom.conversation.scrollTop = dom.conversation.scrollHeight;
  }

  function createBubble(side, labelText) {
    const row = make('div', 'message ' + (side === 'user' ? 'user-message' : 'ai-message'));
    const wrap = make('div', 'bubble-wrap');
    const label = make('small', '', labelText);
    const bubble = make('p', 'bubble', '');
    const note = make('small', 'interrupt-note', '（已中断）');
    note.classList.add('hidden');
    wrap.appendChild(label);
    wrap.appendChild(bubble);
    wrap.appendChild(note);
    row.appendChild(wrap);
    return { row: row, bubble: bubble, note: note, final: false };
  }

  function createPlaceholder(side, text) {
    const row = make('div', 'message placeholder ' + (side === 'user' ? 'user-message' : 'ai-message'));
    const wrap = make('div', 'bubble-wrap');
    wrap.appendChild(make('p', 'bubble', text));
    row.appendChild(wrap);
    return row;
  }

  function renderConversationPlaceholders() {
    clearNode(dom.conversation);
    dom.conversation.appendChild(createPlaceholder('user', '等待识别…'));
    dom.conversation.appendChild(createPlaceholder('ai', '等待回复…'));
  }

  function removePlaceholders() {
    $$('.message.placeholder', dom.conversation).forEach(function (node) {
      node.parentNode.removeChild(node);
    });
  }

  function getTurn(turnId) {
    const key = String(turnId);
    let turn = app.turns.get(key);
    if (!turn) {
      turn = {
        id: key,
        user: null,
        assistant: null,
        extras: [],
        assistantText: '',
        assistantFinal: false,
        cancelled: false
      };
      app.turns.set(key, turn);
    }
    return turn;
  }

  function ensureUserBubble(turnId) {
    const turn = getTurn(turnId);
    if (!turn.user) {
      removePlaceholders();
      const bubble = createBubble('user', '语音识别');
      turn.user = bubble;
      if (turn.assistant) dom.conversation.insertBefore(bubble.row, turn.assistant.row);
      else dom.conversation.appendChild(bubble.row);
      scrollConversation();
    }
    return turn.user;
  }

  function ensureAssistantBubble(turnId) {
    const turn = getTurn(turnId);
    if (!turn.assistant) {
      removePlaceholders();
      const bubble = createBubble('ai', '模型回复');
      turn.assistant = bubble;
      dom.conversation.appendChild(bubble.row);
      scrollConversation();
    }
    return turn.assistant;
  }

  function appendPlaybackBubble(ev, payload, pieceKey) {
    if (app.playbackPieces.has(pieceKey)) return;
    app.playbackPieces.add(pieceKey);
    const turnId = ev.turn_id ? String(ev.turn_id) : 'playback-' + pieceKey;
    const turn = getTurn(turnId);
    removePlaceholders();
    const bubble = createBubble('ai', '语音播报');
    bubble.bubble.textContent = typeof payload.text === 'string' ? payload.text : '';
    turn.extras.push(bubble);
    dom.conversation.appendChild(bubble.row);
    app.playbackBubbles.set(pieceKey, bubble);
    scrollConversation();
  }

  function markTurnCancelled(turn) {
    turn.cancelled = true;
    if (turn.assistant) turn.assistant.note.classList.remove('hidden');
    turn.extras.forEach(function (bubble) {
      bubble.note.classList.remove('hidden');
    });
  }

  function clearConversation() {
    app.turns.clear();
    app.playbackPieces.clear();
    app.playbackBubbles.clear();
    app.playbackStarts.clear();
    clearNode(dom.conversation);
    renderConversationPlaceholders();
  }

  function applySessionTurn(turn) {
    if (!turn || !turn.turn_id) return;
    const turnId = String(turn.turn_id);
    if (typeof turn.user === 'string' && turn.user) {
      const userBubble = ensureUserBubble(turnId);
      userBubble.bubble.textContent = turn.user;
      userBubble.final = true;
    }
    const hasAssistant = typeof turn.assistant === 'string' && turn.assistant.length > 0;
    if (hasAssistant || turn.cancelled) {
      const turnState = getTurn(turnId);
      const assistantBubble = ensureAssistantBubble(turnId);
      turnState.assistantText = turn.assistant || '';
      assistantBubble.bubble.textContent = turnState.assistantText;
      if (turn.assistant_done) {
        turnState.assistantFinal = true;
        assistantBubble.final = true;
      }
      if (turn.cancelled) markTurnCancelled(turnState);
    }
  }

  // ---------------------------------------------------------------- 会话事件
  function handleAsrPartial(ev, payload) {
    const turnId = payload.turn_id || ev.turn_id;
    if (!turnId) return;
    const turn = getTurn(turnId);
    if (turn.user && turn.user.final) return;
    const bubble = ensureUserBubble(turnId);
    bubble.bubble.textContent = typeof payload.text === 'string' ? payload.text : '';
    scrollConversation();
  }

  function handleAsrFinal(ev, payload) {
    const turnId = payload.turn_id || ev.turn_id;
    if (!turnId) return;
    const bubble = ensureUserBubble(turnId);
    if (bubble.final) return;
    bubble.bubble.textContent = typeof payload.text === 'string' ? payload.text : '';
    bubble.final = true;
    scrollConversation();
  }

  function handleAssistantDelta(ev, payload) {
    const turnId = payload.turn_id || ev.turn_id;
    if (!turnId) return;
    const turn = getTurn(turnId);
    if (turn.assistantFinal || turn.cancelled) return;
    const bubble = ensureAssistantBubble(turnId);
    turn.assistantText += typeof payload.text === 'string' ? payload.text : '';
    bubble.bubble.textContent = turn.assistantText;
    scrollConversation();
  }

  function handleAssistantFinal(ev, payload) {
    const turnId = payload.turn_id || ev.turn_id;
    if (!turnId) return;
    const turn = getTurn(turnId);
    if (turn.assistantFinal) return;
    const bubble = ensureAssistantBubble(turnId);
    if (typeof payload.text === 'string' && payload.text) turn.assistantText = payload.text;
    bubble.bubble.textContent = turn.assistantText;
    turn.assistantFinal = true;
    bubble.final = true;
    if (payload.cancelled) markTurnCancelled(turn);
    scrollConversation();
  }

  function handlePlayback(ev, payload) {
    const pieceId = payload.piece_id !== undefined && payload.piece_id !== null ? String(payload.piece_id) : '';
    if (!pieceId) return;
    if (payload.state === 'started') {
      app.playbackStarts.set(pieceId, Date.now());
      if (payload.text) appendPlaybackBubble(ev, payload, pieceId);
    } else if (payload.state === 'finished') {
      app.playbackStarts.delete(pieceId);
      if (payload.text) appendPlaybackBubble(ev, payload, pieceId);
      if (payload.cancelled) {
        const bubble = app.playbackBubbles.get(pieceId);
        if (bubble) bubble.note.classList.remove('hidden');
      }
    }
  }

  // ---------------------------------------------------------------- 相机取用画面
  function hasABFrames() {
    return !!(app.frames.A || app.frames.B);
  }

  function currentFrameSlot() {
    if (!hasABFrames()) return 'model';
    return app.frameSelected === 'B' ? 'B' : 'A';
  }

  function updateFrameSwitch() {
    const show = hasABFrames();
    dom.frameSlots.classList.toggle('hidden', !show);
    const activeSlot = currentFrameSlot();
    dom.frameSlotButtons.forEach(function (button) {
      const active = button.dataset.frameSlot === activeSlot;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  function showModelFramePlaceholder(text, isError) {
    dom.modelFrameImg.classList.add('hidden');
    dom.modelFrameImg.removeAttribute('src');
    if (dom.modelFrameIcon) dom.modelFrameIcon.classList.remove('hidden');
    dom.modelFrameStatus.classList.remove('hidden');
    dom.modelFrameStatus.textContent = text;
    dom.modelFrameStatus.classList.toggle('error', !!isError);
  }

  function showModelFrameImage(url) {
    dom.modelFrameImg.src = url;
    dom.modelFrameImg.classList.remove('hidden');
    if (dom.modelFrameIcon) dom.modelFrameIcon.classList.add('hidden');
    dom.modelFrameStatus.classList.add('hidden');
    dom.modelFrameStatus.classList.remove('error');
  }

  function clearFrames() {
    ['model', 'A', 'B'].forEach(function (slot) {
      const entry = app.frames[slot];
      if (entry && entry.url) URL.revokeObjectURL(entry.url);
      app.frames[slot] = null;
      app.frameLoadSeq[slot] += 1;
    });
    app.frameSelected = 'model';
    updateFrameSwitch();
    showModelFramePlaceholder('尚无取用画面', false);
    if (dom.modelFrameMeta) dom.modelFrameMeta.textContent = '本轮实际输入';
  }

  async function loadFrameImage(slot) {
    const entry = app.frames[slot];
    if (!entry || !entry.frame || !entry.frame.frame_id) return;
    const seq = app.frameLoadSeq[slot] + 1;
    app.frameLoadSeq[slot] = seq;
    try {
      const response = await fetchRaw(API.frame(entry.frame.frame_id), FRAME_TIMEOUT_MS);
      if (seq !== app.frameLoadSeq[slot]) return;
      if (!response.ok) {
        if (currentFrameSlot() === slot) {
          showModelFramePlaceholder('无法获取取用画面（HTTP ' + response.status + '）', true);
        }
        return;
      }
      const blob = await response.blob();
      if (seq !== app.frameLoadSeq[slot]) return;
      const url = URL.createObjectURL(blob);
      if (entry.url) URL.revokeObjectURL(entry.url);
      entry.url = url;
      if (currentFrameSlot() === slot) showModelFrameImage(url);
    } catch (err) {
      if (seq === app.frameLoadSeq[slot] && currentFrameSlot() === slot) {
        showModelFramePlaceholder('无法获取取用画面', true);
      }
    }
  }

  function refreshModelFrame() {
    updateFrameSwitch();
    const slot = currentFrameSlot();
    const entry = app.frames[slot];
    if (!entry) {
      showModelFramePlaceholder('尚无取用画面', false);
      return;
    }
    if (entry.url) {
      showModelFrameImage(entry.url);
      return;
    }
    loadFrameImage(slot);
  }

  function handleFrame(payload) {
    const slot = payload && payload.slot;
    if (slot !== 'model' && slot !== 'A' && slot !== 'B') return;
    const previous = app.frames[slot];
    if (previous && previous.url) URL.revokeObjectURL(previous.url);
    app.frames[slot] = { frame: payload, url: null };
    app.frameLoadSeq[slot] += 1;
    if (slot === 'A' || slot === 'B') {
      if (app.frameSelected !== 'A' && app.frameSelected !== 'B') {
        app.frameSelected = app.frames.A ? 'A' : slot;
      }
    } else if (!hasABFrames()) {
      app.frameSelected = 'model';
    }
    if (dom.modelFrameMeta) {
      const parts = [];
      if (payload.width && payload.height) parts.push(payload.width + '×' + payload.height);
      if (payload.purpose) parts.push(String(payload.purpose));
      dom.modelFrameMeta.textContent = parts.length ? parts.join(' · ') : '本轮实际输入';
    }
    refreshModelFrame();
  }

  // ---------------------------------------------------------------- 预览
  function previewIdleText() {
    const devices = (app.state && app.state.devices) || {};
    const camera = devices.camera || {};
    if (!app.connected) return '未连接';
    if (camera.status === 'error') return camera.detail || '摄像头异常';
    if (camera.status === 'ready') return '摄像头就绪，预览未开启';
    return '预览未开启';
  }

  function setPreviewStatus(text, isError) {
    dom.previewImg.classList.add('hidden');
    if (dom.previewIcon) dom.previewIcon.classList.remove('hidden');
    dom.previewStatus.classList.remove('hidden');
    dom.previewStatus.textContent = text;
    dom.previewStatus.classList.toggle('error', !!isError);
  }

  function showPreviewError(text) {
    app.previewErrorUntil = Date.now() + 5000;
    setPreviewStatus(text, true);
  }

  function showPreviewImage(url) {
    dom.previewImg.src = url;
    dom.previewImg.classList.remove('hidden');
    if (dom.previewIcon) dom.previewIcon.classList.add('hidden');
    dom.previewStatus.classList.add('hidden');
    dom.previewStatus.classList.remove('error');
  }

  function clearPreviewImage() {
    if (app.previewUrl) {
      URL.revokeObjectURL(app.previewUrl);
      app.previewUrl = null;
    }
    dom.previewImg.removeAttribute('src');
    dom.previewImg.classList.add('hidden');
  }

  function stopPreviewPolling() {
    app.previewPolling = false;
    if (app.previewTimer) {
      clearTimeout(app.previewTimer);
      app.previewTimer = null;
    }
  }

  function schedulePreviewTick(delay) {
    if (app.previewTimer) clearTimeout(app.previewTimer);
    app.previewTimer = setTimeout(previewTick, delay);
  }

  async function previewTick() {
    app.previewTimer = null;
    if (!app.previewPolling || !app.previewActive) return;
    if (document.hidden) {
      schedulePreviewTick(400);
      return;
    }
    let nextDelay = PREVIEW_INTERVAL_MS;
    try {
      const response = await fetchRaw(API.previewJpg, 3000);
      if (!app.previewActive) return;
      if (response.status === 204) {
        setPreviewStatus('等待相机帧…', false);
      } else if (!response.ok) {
        const detail = await readErrorDetail(response);
        showPreviewError(detail || '预览不可用（HTTP ' + response.status + '）');
        nextDelay = PREVIEW_ERROR_INTERVAL_MS;
      } else {
        const blob = await response.blob();
        if (!app.previewActive) return;
        const url = URL.createObjectURL(blob);
        if (app.previewUrl) URL.revokeObjectURL(app.previewUrl);
        app.previewUrl = url;
        showPreviewImage(url);
      }
    } catch (err) {
      if (!app.previewActive) return;
      showPreviewError('预览获取失败，重试中');
      nextDelay = PREVIEW_ERROR_INTERVAL_MS;
    } finally {
      if (app.previewPolling && app.previewActive) schedulePreviewTick(nextDelay);
    }
  }

  function startPreviewPolling() {
    if (app.previewPolling) return;
    app.previewPolling = true;
    schedulePreviewTick(0);
  }

  function setPreviewActive(active) {
    app.previewActive = !!active;
    if (app.previewActive) {
      startPreviewPolling();
      setPreviewStatus('正在获取预览…', false);
    } else {
      stopPreviewPolling();
      clearPreviewImage();
      setPreviewStatus(previewIdleText(), false);
    }
    renderButtons();
    renderConversationStatus();
    renderServices();
  }

  function syncPreviewFromState(preview) {
    if (!preview || Date.now() < app.previewGraceUntil) return;
    const active = !!preview.active;
    if (active !== app.previewActive) setPreviewActive(active);
    if (preview.error) {
      const text = typeof preview.error === 'string'
        ? preview.error
        : (preview.error.message || '预览不可用');
      showPreviewError(text);
    } else if (!active) {
      setPreviewStatus(previewIdleText(), false);
    } else if (typeof preview.last_frame_ts !== 'number') {
      setPreviewStatus('等待相机帧…', false);
    }
  }

  function renderCameraState() {
    if (app.previewActive) return;
    if (Date.now() < app.previewErrorUntil) return;
    setPreviewStatus(previewIdleText(), false);
  }

  async function togglePreview() {
    if (!canTogglePreview()) {
      if (readOnly()) return;
      showStatusMessage('当前状态无法切换预览', 4000);
      return;
    }
    const target = !app.previewActive;
    app.pendingPreview = true;
    renderButtons();
    try {
      if (target) await postWithoutBody(API.previewStart);
      else await postWithoutBody(API.previewStop);
      app.previewGraceUntil = Date.now() + 2500;
      setPreviewActive(target);
      clearProgressMessage();
    } catch (err) {
      if (isNetworkError(err)) {
        setConnected(false);
      } else {
        showPreviewError('预览失败：' + err.message);
      }
    } finally {
      app.pendingPreview = false;
      renderButtons();
    }
  }

  // ---------------------------------------------------------------- 上传
  function setUploadFilename(slot, text, title) {
    slot.filename.textContent = text;
    if (title) slot.filename.title = title;
    else slot.filename.removeAttribute('title');
  }

  function setUploadError(slot, text) {
    slot.error.textContent = text || '';
  }

  function showUploadImage(slot, imageInfo) {
    slot.imageId = imageInfo.image_id;
    slot.img.src = API.uploadImage(imageInfo.image_id);
    slot.img.classList.remove('hidden');
    slot.placeholder.classList.add('hidden');
    const dimensions = imageInfo.width && imageInfo.height
      ? imageInfo.width + '×' + imageInfo.height
      : '';
    const format = imageInfo.format ? String(imageInfo.format).toUpperCase() : '';
    const nameParts = [slot.lastFileName || '已上传图片'];
    if (format) nameParts.push(format);
    if (dimensions) nameParts.push(dimensions);
    setUploadFilename(slot, nameParts.join(' · '), nameParts.join(' · '));
  }

  function clearUploadSlot(slot) {
    slot.imageId = null;
    slot.lastFileName = null;
    slot.img.removeAttribute('src');
    slot.img.classList.add('hidden');
    slot.placeholder.classList.remove('hidden');
    setUploadFilename(slot, '未选择图片', '');
    setUploadError(slot, '');
    slot.input.value = '';
  }

  function adoptUploadsFromState(images) {
    if (app.uploadsInitialized || !images) return;
    app.uploadsInitialized = true;
    ['A', 'B'].forEach(function (name) {
      const slot = uploadSlots[name];
      const info = images[name];
      if (!slot || !info || !info.image_id) return;
      slot.lastFileName = null;
      showUploadImage(slot, info);
    });
    renderButtons();
  }

  async function handleFileSelected(slotName, file) {
    const slot = uploadSlots[slotName];
    if (!slot || !file) return;
    setUploadError(slot, '');
    if (!includes(IMAGE_MIME_TYPES, file.type)) {
      setUploadError(slot, '仅支持 JPEG、PNG、WebP 图片。');
      slot.input.value = '';
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setUploadError(slot, '图片超过 20 MB 上限。');
      slot.input.value = '';
      return;
    }
    const previous = {
      imageId: slot.imageId,
      lastFileName: slot.lastFileName,
      label: slot.filename.textContent,
      title: slot.filename.getAttribute('title') || ''
    };
    slot.uploading = true;
    slot.lastFileName = file.name;
    setUploadFilename(slot, '上传中…', '');
    renderButtons();
    try {
      const result = await postBinary(API.uploads + '?slot=' + encodeURIComponent(slotName), file, file.type, UPLOAD_TIMEOUT_MS);
      if (!result || !result.image_id) throw ApiError(500, 'invalid_response', '服务端未返回图片信息', true);
      showUploadImage(slot, result);
    } catch (err) {
      if (isNetworkError(err)) {
        setConnected(false);
      }
      // 上传失败时服务端仍保留原图，本地也恢复原图，避免显示与真实槽位不一致。
      if (previous.imageId) {
        slot.imageId = previous.imageId;
        slot.lastFileName = previous.lastFileName;
        setUploadFilename(slot, previous.label, previous.title);
      } else {
        slot.imageId = null;
        slot.lastFileName = null;
        slot.img.removeAttribute('src');
        slot.img.classList.add('hidden');
        slot.placeholder.classList.remove('hidden');
        setUploadFilename(slot, '未选择图片', '');
      }
      setUploadError(slot, err.message || '上传失败');
    } finally {
      slot.uploading = false;
      slot.input.value = '';
      renderButtons();
    }
  }

  async function removeUpload(slotName) {
    const slot = uploadSlots[slotName];
    if (!slot || !slot.imageId || slot.uploading) return;
    if (readOnly()) return;
    slot.uploading = true;
    renderButtons();
    try {
      await jsonDelete(API.uploadImage(slot.imageId));
      clearUploadSlot(slot);
    } catch (err) {
      if (isNetworkError(err)) setConnected(false);
      setUploadError(slot, '删除失败：' + (err.message || '未知错误'));
    } finally {
      slot.uploading = false;
      renderButtons();
    }
  }

  // ---------------------------------------------------------------- 调试
  function findModule(moduleId) {
    for (let index = 0; index < app.modules.length; index += 1) {
      if (app.modules[index].id === moduleId) return app.modules[index];
    }
    return null;
  }

  function moduleUnsupportedReason(module) {
    const capabilities = (app.state && app.state.capabilities) || null;
    if (!module || !capabilities) return '';
    if (module.id === 'contract' && capabilities.contract === false) {
      return '未接入（正式配置 contract.enabled=false）';
    }
    if (module.id === 'speaker' && capabilities.speaker === false) {
      return '未接入（声纹能力未启用）';
    }
    return '';
  }

  function refreshDebugNotice() {
    if (!dom.debugNotice) return;
    const reason = app.selectedModule ? moduleUnsupportedReason(app.selectedModule) : '';
    if (reason) {
      dom.debugNotice.textContent = reason;
      dom.debugNotice.classList.remove('hidden');
    } else {
      dom.debugNotice.textContent = '';
      dom.debugNotice.classList.add('hidden');
    }
  }

  function clearDebugOutput(text) {
    clearNode(dom.debugOutput);
    dom.debugOutput.textContent = text;
    dom.debugOutput.dataset.empty = '1';
  }

  function appendDebugNode(node) {
    if (dom.debugOutput.dataset.empty === '1') {
      clearNode(dom.debugOutput);
      dom.debugOutput.dataset.empty = '0';
    }
    dom.debugOutput.appendChild(node);
    dom.debugOutput.scrollTop = dom.debugOutput.scrollHeight;
  }

  function appendDebugRow(text, kind) {
    const row = make('div', 'debug-row ' + (kind || ''));
    row.appendChild(make('time', '', formatClock(Date.now())));
    row.appendChild(make('div', 'debug-text', text));
    appendDebugNode(row);
  }

  function appendDebugLog(payload) {
    const level = String(payload.level || 'INFO').toUpperCase();
    const moduleName = payload.module ? '[' + payload.module + '] ' : '';
    appendDebugRow(moduleName + (payload.message || ''), level === 'WARN' ? 'warn' : 'info');
  }

  function appendDebugFrame(payload) {
    if (!payload || !payload.frame_id) return;
    const row = make('div', 'debug-row');
    row.appendChild(make('time', '', formatClock(Date.now())));
    const body = make('div', 'debug-frame');
    const image = document.createElement('img');
    image.alt = '调试取帧';
    image.src = API.frame(payload.frame_id);
    image.addEventListener('error', function () {
      image.remove();
      const failed = make('div', 'debug-text', '取帧图片加载失败');
      body.appendChild(failed);
    });
    body.appendChild(image);
    const captionParts = [];
    if (payload.width && payload.height) captionParts.push(payload.width + '×' + payload.height);
    if (payload.purpose) captionParts.push(String(payload.purpose));
    if (payload.slot) captionParts.push('槽位 ' + payload.slot);
    if (captionParts.length) body.appendChild(make('div', 'debug-caption', captionParts.join(' · ')));
    row.appendChild(body);
    appendDebugNode(row);
  }

  function appendDebugTiming(payload) {
    const labels = {
      asr_ms: '识别',
      vision_ms: '视觉',
      generation_ms: '生成',
      tts_ms: '合成',
      playback_wait_ms: '播放等待',
      acoustic_first_audio_ms: '声学首声'
    };
    const parts = [];
    Object.keys(labels).forEach(function (key) {
      if (typeof payload[key] === 'number' && isFinite(payload[key])) {
        parts.push(labels[key] + ' ' + Math.round(payload[key]) + ' ms');
      }
    });
    if (parts.length) appendDebugRow('耗时：' + parts.join(' · '), 'info');
  }

  function handleDebugEvent(ev) {
    const payload = ev.payload || {};
    switch (ev.type) {
      case 'asr.final':
        appendDebugRow('识别：' + (payload.text || ''), 'info');
        break;
      case 'assistant.final':
        appendDebugRow('回复：' + (payload.text || '') + (payload.cancelled ? '（已中断）' : ''), 'info');
        break;
      case 'playback':
        if (payload.state === 'started') {
          app.debugPlaybackStarts.set(String(payload.piece_id), Date.now());
          appendDebugRow('播放开始' + (payload.text ? '：' + payload.text : ''), 'info');
        } else if (payload.state === 'finished') {
          const pieceKey = String(payload.piece_id);
          const started = app.debugPlaybackStarts.get(pieceKey);
          app.debugPlaybackStarts.delete(pieceKey);
          const duration = started ? ' · ' + ((Date.now() - started) / 1000).toFixed(1) + ' s' : '';
          appendDebugRow(
            '播放' + (payload.cancelled ? '已取消' : '完成') + (payload.text ? '：' + payload.text : '') + duration,
            'info'
          );
        }
        break;
      case 'error':
        appendDebugRow('错误：' + (payload.message || '未知错误'), 'error');
        break;
      default:
        break;
    }
  }

  async function loadDebugModules() {
    if (app.modulesLoaded) {
      renderModuleButtons();
      return;
    }
    try {
      const result = await request(API.debugModules, { timeout: REQUEST_TIMEOUT_MS });
      const data = result.data;
      app.modules = data && Array.isArray(data.modules) ? data.modules : [];
      app.modulesLoaded = true;
      renderModuleButtons();
      let targetId = app.selectedModuleId;
      if (app.debugRun) targetId = app.debugRun.moduleId;
      let module = targetId ? findModule(targetId) : null;
      if (!module && app.modules.length) module = app.modules[0];
      if (module) selectModule(module.id);
      else {
        app.selectedModule = null;
        app.selectedModuleId = null;
        renderDebugForm(null);
        renderDebugActions(null);
        refreshDebugNotice();
        renderButtons();
      }
    } catch (err) {
      if (isNetworkError(err)) setConnected(false);
      app.modulesLoaded = false;
      clearNode(dom.modules);
      dom.modules.appendChild(make('span', 'small', '无法获取调试模块：' + (err.message || '未知错误')));
    }
  }

  function renderModuleButtons() {
    clearNode(dom.modules);
    app.modules.forEach(function (module) {
      const button = make('button', '', module.name || module.id);
      button.type = 'button';
      button.title = module.desc || module.name || module.id;
      if (app.selectedModuleId === module.id) button.className = 'active';
      button.addEventListener('click', function () {
        if (button.disabled) return;
        selectModule(module.id);
      });
      dom.modules.appendChild(button);
    });
    renderButtons();
  }

  function selectModule(moduleId) {
    const module = findModule(moduleId);
    app.selectedModule = module || null;
    app.selectedModuleId = module ? module.id : null;
    renderModuleButtons();
    if (!module) {
      dom.moduleTitle.textContent = '尚未选择模块';
      dom.moduleDesc.textContent = '选择模块后开始调试。';
      renderDebugForm(null);
      renderDebugActions(null);
      refreshDebugNotice();
      renderButtons();
      return;
    }
    dom.moduleTitle.textContent = module.name || module.id;
    dom.moduleDesc.textContent = module.desc || '';
    renderDebugForm(module);
    renderDebugActions(module);
    refreshDebugNotice();
    if (!debugIsActive()) {
      app.lastDebugResult = null;
      clearDebugOutput('尚未运行。选择模块后，手动开始调试。');
      renderDebugTime();
    }
    renderButtons();
  }

  function renderDebugForm(module) {
    clearNode(dom.debugForm);
    if (!module) return;
    const fields = Array.isArray(module.fields) ? module.fields : [];
    const form = document.createElement('form');
    form.id = 'debugFormElement';
    form.noValidate = false;
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      startDebug();
    });

    fields.forEach(function (field, index) {
      if (!field || !field.name) return;
      const fieldId = 'debugField_' + index + '_' + String(field.name).replace(/[^\w-]/g, '_');
      const type = field.type || 'text';
      const labelText = (field.label || field.name) + (field.required ? ' *' : '');

      if (type === 'checkbox') {
        const label = make('label', 'field-check');
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = fieldId;
        input.dataset.fieldName = field.name;
        input.dataset.fieldType = 'checkbox';
        input.checked = !!field.value;
        if (field.required) input.required = true;
        label.appendChild(input);
        label.appendChild(make('span', '', labelText));
        form.appendChild(label);
        return;
      }

      const label = make('label', 'field', labelText);
      label.htmlFor = fieldId;
      form.appendChild(label);

      if (type === 'select') {
        const select = document.createElement('select');
        select.className = 'input';
        select.id = fieldId;
        select.dataset.fieldName = field.name;
        select.dataset.fieldType = 'select';
        select.required = !!field.required;
        const options = Array.isArray(field.options) ? field.options : [];
        options.forEach(function (option) {
          const optionElement = document.createElement('option');
          optionElement.value = String(option);
          optionElement.textContent = String(option);
          if (field.value !== undefined && field.value !== null && String(field.value) === String(option)) {
            optionElement.selected = true;
          }
          select.appendChild(optionElement);
        });
        form.appendChild(select);
        return;
      }

      if (type === 'textarea') {
        const textarea = document.createElement('textarea');
        textarea.className = 'input';
        textarea.id = fieldId;
        textarea.rows = 3;
        textarea.dataset.fieldName = field.name;
        textarea.dataset.fieldType = 'textarea';
        textarea.required = !!field.required;
        textarea.placeholder = field.placeholder || '';
        textarea.value = field.value !== undefined && field.value !== null ? String(field.value) : '';
        form.appendChild(textarea);
        return;
      }

      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'input';
      input.id = fieldId;
      input.dataset.fieldName = field.name;
      input.dataset.fieldType = 'text';
      input.required = !!field.required;
      input.placeholder = field.placeholder || '';
      input.value = field.value !== undefined && field.value !== null ? String(field.value) : '';
      form.appendChild(input);
    });

    dom.debugForm.appendChild(form);
  }

  function renderDebugActions(module) {
    clearNode(dom.debugActions);
    if (!module) return;
    const actions = Array.isArray(module.actions) ? module.actions : [];
    actions.forEach(function (action) {
      if (!action || !action.name) return;
      const button = make('button', '', action.label || action.name);
      button.type = 'button';
      button.title = '调试动作：' + (action.label || action.name);
      button.addEventListener('click', function () {
        if (button.disabled) return;
        sendDebugAction(module, action);
      });
      dom.debugActions.appendChild(button);
    });
    renderButtons();
  }

  function collectDebugInput() {
    const result = {};
    const form = dom.debugForm.querySelector('form');
    if (!form) return result;
    $$('[data-field-name]', form).forEach(function (control) {
      const name = control.dataset.fieldName;
      if (!name) return;
      if (control.dataset.fieldType === 'checkbox') result[name] = !!control.checked;
      else result[name] = control.value;
    });
    return result;
  }

  async function sendDebugAction(module, action) {
    if (readOnly()) return;
    try {
      await jsonPost(API.debugAction, {
        request_id: makeRequestId(),
        module: module.id,
        action: action.name
      });
      appendDebugRow('已请求动作：' + (action.label || action.name), 'info');
    } catch (err) {
      showDebugError('调试动作', err);
    }
  }

  function showDebugError(prefix, err) {
    if (isNetworkError(err)) {
      setConnected(false);
      return;
    }
    appendDebugRow(prefix + '失败：' + (err && err.message ? err.message : '未知错误'), 'error');
  }

  function renderDebugTime() {
    if (app.debugRun) {
      const elapsed = Math.max(0, Date.now() - app.debugRun.startedAt);
      dom.debugTime.textContent = '开始 ' + formatClock(app.debugRun.startedAt) + ' · 耗时 ' + formatDuration(elapsed);
    } else if (app.lastDebugResult) {
      dom.debugTime.textContent = app.lastDebugResult;
    } else {
      dom.debugTime.textContent = '未开始';
    }
  }

  function beginDebugRun(info) {
    const module = findModule(info.moduleId);
    app.debugRun = {
      runId: String(info.run_id),
      moduleId: info.moduleId,
      moduleName: info.moduleName || (module && module.name) || info.moduleId,
      startedAt: info.startedAt || Date.now()
    };
    app.debugStopping = false;
    clearDebugOutput((info.adopted ? '已接入正在运行的调试：' : '已开始调试：') + app.debugRun.moduleName + '。等待事件…');
    if (module && (!app.selectedModule || app.selectedModule.id !== module.id)) {
      app.selectedModule = module;
      app.selectedModuleId = module.id;
      renderModuleButtons();
      renderDebugForm(module);
      renderDebugActions(module);
      refreshDebugNotice();
    }
    renderDebugTime();
    renderButtons();
  }

  function finishDebugRun(text) {
    if (!app.debugRun) return;
    const runId = app.debugRun.runId;
    const elapsed = Math.max(0, Date.now() - app.debugRun.startedAt);
    app.finishedDebugRuns.add(runId);
    if (app.finishedDebugRuns.size > 50) {
      const first = app.finishedDebugRuns.values().next().value;
      app.finishedDebugRuns.delete(first);
    }
    app.debugRun = null;
    app.debugStopping = false;
    app.lastDebugResult = text + ' · 用时 ' + formatDuration(elapsed);
    appendDebugRow(app.lastDebugResult, 'info');
    renderDebugTime();
    renderButtons();
  }

  function syncDebugState(state) {
    const debug = state ? state.debug : null;
    if (app.debugRun) {
      if (Date.now() < app.debugGraceUntil) return;
      if (!debug) {
        finishDebugRun('调试已结束');
        return;
      }
      if (String(debug.run_id) !== app.debugRun.runId) {
        finishDebugRun('调试已结束');
        return;
      }
      if (debug.status && debug.status !== 'running') {
        if (debug.status === 'completed') finishDebugRun('调试已完成');
        else if (debug.status === 'failed') finishDebugRun('调试失败');
        else finishDebugRun('调试已中断');
      }
      return;
    }
    if (debug && debug.status === 'running' && !app.finishedDebugRuns.has(String(debug.run_id))) {
      beginDebugRun({
        run_id: debug.run_id,
        moduleId: debug.module,
        moduleName: (findModule(debug.module) || {}).name,
        startedAt: toMillis(debug.started_at) || Date.now(),
        adopted: true
      });
    }
  }

  async function startDebug() {
    const module = app.selectedModule;
    if (!module) return;
    if (readOnly() || debugIsActive() || runIsActive() || app.previewActive) return;
    if (moduleUnsupportedReason(module)) {
      refreshDebugNotice();
      renderButtons();
      return;
    }
    const form = dom.debugForm.querySelector('form');
    if (form && typeof form.reportValidity === 'function' && !form.reportValidity()) return;
    const input = collectDebugInput();
    dom.debugRun.disabled = true;
    try {
      const result = await jsonPost(API.debugStart, {
        request_id: makeRequestId(),
        module: module.id,
        input: input
      });
      const data = result.data;
      if (!data || !data.run_id) {
        appendDebugRow('开始调试失败：服务端未返回 run_id', 'error');
        return;
      }
      app.debugGraceUntil = Date.now() + 2500;
      beginDebugRun({
        run_id: data.run_id,
        moduleId: module.id,
        moduleName: module.name || module.id,
        startedAt: Date.now()
      });
    } catch (err) {
      showDebugError('开始调试', err);
    } finally {
      renderButtons();
    }
  }

  async function stopDebug() {
    if (!canStopDebug()) return;
    app.debugStopping = true;
    renderButtons();
    try {
      await jsonPost(API.debugStop, {
        request_id: makeRequestId(),
        run_id: app.debugRun.runId
      });
      app.debugGraceUntil = Date.now() + 2500;
      appendDebugRow('已请求停止调试，等待服务端结束…', 'info');
      dom.debugTime.textContent = '停止中…';
    } catch (err) {
      app.debugStopping = false;
      showDebugError('停止调试', err);
    } finally {
      renderButtons();
    }
  }

  function renderDebugCapabilities() {
    if (!app.selectedModule) return;
    refreshDebugNotice();
    renderButtons();
  }

  // ---------------------------------------------------------------- 日志
  function buildLogRow(entry) {
    const row = make('div', 'logline');
    row.appendChild(make('time', '', entry.time));
    row.appendChild(make('span', 'level' + (entry.level === 'WARN' ? ' warn' : ''), entry.level));
    row.appendChild(make('span', 'module', entry.module));
    row.appendChild(make('span', 'content', entry.message));
    return row;
  }

  function appendLog(payload) {
    const entry = {
      time: formatClock(Date.now()),
      level: String((payload && payload.level) || 'INFO').toUpperCase(),
      module: (payload && payload.module) || '',
      message: (payload && payload.message) || '',
      el: null
    };
    app.logs.push(entry);
    if (dom.logFilter.value === 'all' || entry.level === dom.logFilter.value) {
      entry.el = buildLogRow(entry);
      dom.logs.appendChild(entry.el);
      dom.logs.scrollTop = dom.logs.scrollHeight;
    }
    if (app.logs.length > MAX_LOG_ENTRIES) {
      const removed = app.logs.shift();
      if (removed.el && removed.el.parentNode) removed.el.parentNode.removeChild(removed.el);
    }
  }

  function rebuildLogs() {
    clearNode(dom.logs);
    const filter = dom.logFilter.value;
    app.logs.forEach(function (entry) {
      if (filter === 'all' || entry.level === filter) {
        entry.el = buildLogRow(entry);
        dom.logs.appendChild(entry.el);
      } else {
        entry.el = null;
      }
    });
    dom.logs.scrollTop = dom.logs.scrollHeight;
  }

  // ---------------------------------------------------------------- 事件分发
  function isCurrentRunEvent(ev) {
    if (!ev.run_id) return true;
    if (!app.run || !app.run.id) return false;
    return String(ev.run_id) === String(app.run.id);
  }

  function mergeTimings(payload) {
    if (!payload) return;
    app.timings = Object.assign({}, app.timings || {}, payload);
    renderTimings();
  }

  function handleEvent(ev) {
    if (!ev || typeof ev.type !== 'string') return;
    const payload = ev.payload && typeof ev.payload === 'object' ? ev.payload : {};
    const debugEvent = !!(app.debugRun && ev.run_id && String(ev.run_id) === app.debugRun.runId);

    switch (ev.type) {
      case 'state':
        return;
      case 'log':
        appendLog(payload);
        if (debugEvent) appendDebugLog(payload);
        return;
      case 'frame':
        handleFrame(payload);
        if (debugEvent) appendDebugFrame(payload);
        return;
      case 'timing':
        if (debugEvent) appendDebugTiming(payload);
        else if (isCurrentRunEvent(ev)) mergeTimings(payload);
        return;
      case 'error':
        if (debugEvent) appendDebugRow('错误：' + (payload.message || '未知错误'), 'error');
        else showStatusMessage(payload.message || '运行错误', 8000);
        return;
      default:
        break;
    }

    if (debugEvent) {
      handleDebugEvent(ev);
      return;
    }
    if (!isCurrentRunEvent(ev)) return;

    switch (ev.type) {
      case 'asr.partial':
        handleAsrPartial(ev, payload);
        break;
      case 'asr.final':
        handleAsrFinal(ev, payload);
        break;
      case 'assistant.delta':
        handleAssistantDelta(ev, payload);
        break;
      case 'assistant.final':
        handleAssistantFinal(ev, payload);
        break;
      case 'playback':
        handlePlayback(ev, payload);
        break;
      default:
        break;
    }
  }

  // ---------------------------------------------------------------- 状态应用
  function applyState(state) {
    if (!state || typeof state !== 'object') return;
    const previousStatus = app.run && app.run.status;
    const previousRunId = app.run && app.run.id ? String(app.run.id) : null;
    app.state = state;
    if (typeof state.server_time_ms === 'number') {
      app.serverOffsetMs = state.server_time_ms - Date.now();
    }
    app.run = state.run && typeof state.run === 'object'
      ? state.run
      : { id: null, mode: null, status: 'idle', started_at: null, elapsed_ms: null };
    app.interaction = (state.services && state.services.interaction) || 'idle';
    app.controlLocked = !!(state.control && state.control.locked);
    const runId = app.run.id ? String(app.run.id) : null;
    if (runId && runId !== previousRunId) app.timings = {};
    const incomingTimings = state.timings && typeof state.timings === 'object' ? state.timings : {};
    const mergedTimings = Object.assign({}, app.timings || {});
    Object.keys(incomingTimings).forEach(function (key) {
      if (typeof incomingTimings[key] === 'number') mergedTimings[key] = incomingTimings[key];
    });
    app.timings = mergedTimings;

    if (app.pendingRun) {
      if (app.pendingRun.kind === 'start') {
        if (app.run.status === 'running' || (app.pendingRun.runId && app.run.id && String(app.run.id) === String(app.pendingRun.runId))) {
          app.pendingRun = null;
        }
      } else if (app.pendingRun.kind === 'stop' && !includes(['starting', 'running'], app.run.status)) {
        app.pendingRun = null;
      }
    }

    if (app.run.status !== previousStatus) clearProgressMessage();

    renderServices();
    renderRunState();
    renderElapsed();
    renderMetrics();
    renderTimings();
    renderCameraState();
    syncPreviewFromState(state.preview);
    adoptUploadsFromState(state.images);
    syncDebugState(state);
    renderDebugCapabilities();
    renderButtons();
    renderConversationStatus();
  }

  // ---------------------------------------------------------------- 轮询
  let stateFailures = 0;
  let eventFailures = 0;
  let bootstrapFailures = 0;

  function reconnectDelay(failures) {
    const exponent = Math.max(0, Math.min(failures, 4));
    return Math.min(POLL_INTERVAL_MS * Math.pow(2, exponent), MAX_RECONNECT_DELAY_MS);
  }

  function notePollFailure() {
    setConnected(false);
  }

  async function stateTick() {
    try {
      const result = await request(API.state, { timeout: POLL_TIMEOUT_MS });
      stateFailures = 0;
      setConnected(true);
      applyState(result.data);
    } catch (err) {
      stateFailures += 1;
      notePollFailure();
    } finally {
      setTimeout(stateTick, app.connected ? POLL_INTERVAL_MS : reconnectDelay(stateFailures));
    }
  }

  async function rebuildSession() {
    const result = await request(API.session, { timeout: POLL_TIMEOUT_MS });
    const data = result.data;
    if (!data || typeof data !== 'object') throw ApiError(500, 'invalid_response', '会话数据无效', true);
    if (typeof data.client_id === 'string' && data.client_id) app.clientId = data.client_id;
    if (typeof data.token === 'string' && data.token) app.token = data.token;
    app.lastSeq = typeof data.last_seq === 'number' ? data.last_seq : app.lastSeq;
    clearConversation();
    const turns = Array.isArray(data.turns) ? data.turns : [];
    turns.forEach(applySessionTurn);
    if (!turns.length) renderConversationPlaceholders();
  }

  async function eventsTick() {
    if (!app.sessionReady) {
      setTimeout(eventsTick, 250);
      return;
    }
    const cursor = app.lastSeq;
    try {
      const result = await request(API.events + '?after=' + encodeURIComponent(String(cursor)), { timeout: POLL_TIMEOUT_MS });
      eventFailures = 0;
      setConnected(true);
      const data = result.data || {};
      if (data.reset_required) {
        await rebuildSession();
      } else {
        const events = Array.isArray(data.events) ? data.events : [];
        events.forEach(function (ev) {
          try {
            handleEvent(ev);
          } catch (err) {
            /* 单个事件渲染失败不阻塞后续事件 */
          }
        });
        app.lastSeq = typeof data.next_seq === 'number' ? data.next_seq : cursor;
      }
    } catch (err) {
      eventFailures += 1;
      notePollFailure();
    } finally {
      setTimeout(eventsTick, app.connected ? POLL_INTERVAL_MS : reconnectDelay(eventFailures));
    }
  }

  async function heartbeatTick() {
    if (app.clientId) {
      try {
        const result = await jsonPost(API.heartbeat, { client_id: app.clientId });
        if (result.data && typeof result.data.owner === 'boolean') setOwner(result.data.owner);
      } catch (err) {
        /* 心跳失败由状态轮询判定连接，不在此单独断线 */
      }
    }
    setTimeout(heartbeatTick, HEARTBEAT_INTERVAL_MS);
  }

  async function loadSession() {
    const result = await request(API.session, { timeout: POLL_TIMEOUT_MS });
    const data = result.data;
    if (!data || typeof data !== 'object') throw ApiError(500, 'invalid_response', '会话数据无效', true);
    if (typeof data.client_id === 'string' && data.client_id) app.clientId = data.client_id;
    if (typeof data.token === 'string' && data.token) app.token = data.token;
    app.lastSeq = typeof data.last_seq === 'number' ? data.last_seq : 0;
    clearConversation();
    const turns = Array.isArray(data.turns) ? data.turns : [];
    turns.forEach(applySessionTurn);
    if (!turns.length) renderConversationPlaceholders();
  }

  async function bootstrap() {
    app.connectionAttempted = true;
    try {
      await loadSession();
      bootstrapFailures = 0;
      app.sessionReady = true;
      setConnected(true);
      try {
        const result = await request(API.state, { timeout: POLL_TIMEOUT_MS });
        applyState(result.data);
      } catch (err) {
        /* 首个状态请求失败时由状态轮询链继续重试 */
      }
      stateTick();
      eventsTick();
      heartbeatTick();
    } catch (err) {
      bootstrapFailures += 1;
      notePollFailure();
      renderConversationStatus();
      setTimeout(bootstrap, reconnectDelay(bootstrapFailures));
    }
  }

  // ---------------------------------------------------------------- 对话框与操作
  function openStartConfirm() {
    if (!canStartRun()) {
      showStatusMessage(startBlockedReason(), 5000);
      return;
    }
    app.pendingConfirm = 'start';
    dom.confirmTitle.textContent = '开始综合测试';
    dom.confirmBody.textContent = '将占用目标机默认麦克风、扬声器与相机；已有模型服务不会被重启。';
    dom.confirmStart.textContent = '开始测试';
    if (typeof dom.confirm.showModal === 'function') dom.confirm.showModal();
  }

  function openStopConfirm() {
    if (!canStopRun()) {
      showStatusMessage('当前没有可停止的运行', 5000);
      return;
    }
    app.pendingConfirm = 'stop';
    dom.confirmTitle.textContent = '停止综合测试';
    dom.confirmBody.textContent = '将停止本轮测试并释放音频、相机与预览资源；已生成的会话记录会保留。';
    dom.confirmStart.textContent = '停止测试';
    if (typeof dom.confirm.showModal === 'function') dom.confirm.showModal();
  }

  function closeConfirm() {
    app.pendingConfirm = null;
    if (dom.confirm.open) dom.confirm.close();
  }

  async function performStart() {
    app.pendingRun = { kind: 'start', runId: null, until: Date.now() + 4000 };
    renderButtons();
    try {
      const result = await jsonPost(API.runStart, { request_id: makeRequestId() });
      const runId = result.data && result.data.run_id ? String(result.data.run_id) : null;
      app.pendingRun = { kind: 'start', runId: runId, until: Date.now() + 4000 };
      clearConversation();
      clearFrames();
      app.timings = {};
      renderTimings();
      app.run = {
        id: runId,
        mode: app.run ? app.run.mode : null,
        status: 'starting',
        started_at: serverNow(),
        elapsed_ms: 0
      };
      app.statusMessage = { text: '测试启动中…', until: Date.now() + 8000, kind: 'progress' };
      renderRunState();
      renderElapsed();
      renderConversationStatus();
    } catch (err) {
      app.pendingRun = null;
      if (isNetworkError(err)) setConnected(false);
      else showStatusMessage('开始测试失败：' + err.message, 8000);
    } finally {
      renderButtons();
    }
  }

  async function performStop() {
    const runId = (app.run && app.run.id)
      || (app.pendingRun && app.pendingRun.runId)
      || null;
    if (!runId) {
      showStatusMessage('当前没有可停止的运行', 5000);
      return;
    }
    app.pendingRun = { kind: 'stop', runId: String(runId), until: Date.now() + 4000 };
    renderButtons();
    try {
      await jsonPost(API.runStop, { request_id: makeRequestId(), run_id: runId });
      app.run = Object.assign({}, app.run, { status: 'stopping' });
      app.statusMessage = { text: '测试停止中…', until: Date.now() + 8000, kind: 'progress' };
      renderRunState();
      renderConversationStatus();
    } catch (err) {
      app.pendingRun = null;
      if (isNetworkError(err)) setConnected(false);
      else showStatusMessage('停止测试失败：' + err.message, 8000);
    } finally {
      renderButtons();
    }
  }

  async function confirmPendingAction() {
    const action = app.pendingConfirm;
    closeConfirm();
    if (!action) return;
    if (readOnly()) {
      showStatusMessage('未连接或只读，无法操作', 5000);
      return;
    }
    if (action === 'start') await performStart();
    else if (action === 'stop') await performStop();
  }

  // ---------------------------------------------------------------- 页面交互
  function setSidebarCollapsed(collapsed) {
    dom.shell.classList.toggle('collapsed', collapsed);
    dom.sidebarHeader.setAttribute('aria-expanded', String(!collapsed));
    dom.sidebarHeader.setAttribute('aria-label', collapsed ? '展开侧边栏' : '折叠侧边栏');
    dom.sidebarHeader.title = collapsed ? '展开侧边栏' : '折叠侧边栏';
  }

  function toggleSidebar() {
    setSidebarCollapsed(!dom.shell.classList.contains('collapsed'));
  }

  function switchPage(page) {
    app.activePage = page;
    dom.navButtons.forEach(function (button) {
      button.classList.toggle('active', button.dataset.page === page);
    });
    dom.integrationPage.classList.toggle('hidden', page !== 'integration');
    dom.debugPage.classList.toggle('hidden', page !== 'debug');
    if (page === 'debug') loadDebugModules();
  }

  function tickClock() {
    renderElapsed();
    renderDebugTime();
    if (app.statusMessage && app.statusMessage.until <= Date.now()) {
      app.statusMessage = null;
      renderConversationStatus();
    }
    if (app.pendingRun && app.pendingRun.until <= Date.now()) {
      app.pendingRun = null;
      renderButtons();
    }
  }

  function wireEvents() {
    dom.sidebarHeader.addEventListener('click', toggleSidebar);
    dom.sidebarHeader.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
        event.preventDefault();
        toggleSidebar();
      }
    });

    dom.navButtons.forEach(function (button) {
      const label = button.dataset.page === 'integration' ? '综合测试' : '功能调试';
      button.title = label;
      button.setAttribute('aria-label', label);
      button.addEventListener('click', function () {
        switchPage(button.dataset.page);
      });
    });

    dom.mediaTabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        const isCamera = tab.dataset.media === 'camera';
        dom.mediaTabs.forEach(function (item) {
          item.classList.toggle('active', item === tab);
        });
        dom.cameraPanel.classList.toggle('hidden', !isCamera);
        dom.uploadPanel.classList.toggle('hidden', isCamera);
      });
    });

    dom.start.addEventListener('click', openStartConfirm);
    dom.stop.addEventListener('click', openStopConfirm);
    dom.confirmStart.addEventListener('click', confirmPendingAction);
    dom.cancelStart.addEventListener('click', closeConfirm);
    dom.confirm.addEventListener('close', function () {
      app.pendingConfirm = null;
    });
    dom.confirm.addEventListener('cancel', function () {
      app.pendingConfirm = null;
    });

    dom.cameraPreview.addEventListener('click', togglePreview);
    dom.frameSlotButtons.forEach(function (button) {
      button.addEventListener('click', function () {
        app.frameSelected = button.dataset.frameSlot === 'B' ? 'B' : 'A';
        refreshModelFrame();
      });
    });

    Object.keys(uploadSlots).forEach(function (name) {
      const slot = uploadSlots[name];
      slot.pick.addEventListener('click', function () {
        if (slot.pick.disabled) return;
        slot.input.click();
      });
      slot.input.addEventListener('change', function () {
        const file = slot.input.files && slot.input.files[0];
        if (file) handleFileSelected(name, file);
        else slot.input.value = '';
      });
      slot.remove.addEventListener('click', function () {
        if (slot.remove.disabled) return;
        removeUpload(name);
      });
    });

    dom.debugRun.addEventListener('click', startDebug);
    dom.debugStop.addEventListener('click', stopDebug);

    dom.logFilter.addEventListener('change', rebuildLogs);
    dom.clearLogs.addEventListener('click', function () {
      app.logs = [];
      clearNode(dom.logs);
    });

    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) renderElapsed();
    });
  }

  function init() {
    if (dom.modelFrameMeta) dom.modelFrameMeta.textContent = '本轮实际输入';
    if (dom.debugOutput) dom.debugOutput.dataset.empty = '1';
    wireEvents();
    renderConversationPlaceholders();
    renderServices();
    renderRunState();
    renderElapsed();
    renderTimings();
    renderMetrics();
    renderButtons();
    renderConversationStatus();
    renderDebugTime();
    updateFrameSwitch();
    setInterval(tickClock, 1000);
    bootstrap();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
