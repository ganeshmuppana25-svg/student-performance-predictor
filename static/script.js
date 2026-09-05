/* Student Performance Predictor — frontend logic (Phase 2 UI/UX) */
"use strict";

const FEATURE_LABELS = {
  attendance: "Attendance (%)",
  study_hours: "Study Hours per Day",
  assignment_score: "Assignment Score (%)",
  previous_exam_score: "Previous Exam Score (%)",
  internal_test_score: "Internal/Test Score (%)",
  sleep_hours: "Sleep Hours",
  screen_time: "Daily Screen Time (hours)",
  assignments_completed: "Assignments Completed (%)",
  class_participation: "Class Participation (%)",
  previous_backlogs: "Previous Backlogs",
};
const FEATURE_KEYS = Object.keys(FEATURE_LABELS);
const HISTORY_KEY = "spp_history";
const THEME_KEY = "spp_theme";
const MAX_HISTORY = 10;

// Presets only fill the existing inputs — they never affect prediction logic.
const PRESETS = {
  excellent: { attendance: 94, study_hours: 6.5, assignment_score: 90,
    previous_exam_score: 88, internal_test_score: 85, sleep_hours: 7.5,
    screen_time: 2, assignments_completed: 96, class_participation: 85,
    previous_backlogs: 0 },
  average: { attendance: 78, study_hours: 3.5, assignment_score: 70,
    previous_exam_score: 65, internal_test_score: 66, sleep_hours: 7,
    screen_time: 5, assignments_completed: 75, class_participation: 55,
    previous_backlogs: 1 },
  risk: { attendance: 55, study_hours: 1.5, assignment_score: 40,
    previous_exam_score: 38, internal_test_score: 40, sleep_hours: 4.5,
    screen_time: 9, assignments_completed: 35, class_participation: 20,
    previous_backlogs: 5 },
};

// ---------- Helpers ----------
function $(id) { return document.getElementById(id); }

function getFormValues() {
  const values = {};
  for (const key of FEATURE_KEYS) {
    values[key] = $(key).value.trim();
  }
  return values;
}

function validateClient(values) {
  const errors = [];
  for (const key of FEATURE_KEYS) {
    const v = values[key];
    if (v === "") { errors.push(`${FEATURE_LABELS[key]} is required.`); continue; }
    const num = Number(v);
    if (!isFinite(num)) {
      errors.push(`${FEATURE_LABELS[key]} must be a valid number.`);
      continue;
    }
    const input = $(key);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    if (num < min || num > max) {
      errors.push(`${FEATURE_LABELS[key]} must be between ${min} and ${max}.`);
    }
  }
  return errors;
}

function showError(message) {
  const el = $("form-error");
  el.textContent = message;
  el.hidden = false;
}

function clearError() {
  const el = $("form-error");
  el.textContent = "";
  el.hidden = true;
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistoryEntry(entry) {
  const history = loadHistory();
  history.unshift(entry);
  while (history.length > MAX_HISTORY) history.pop();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

// ---------- Theme ----------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = $("theme-toggle");
  if (btn) btn.textContent = theme === "dark" ? "🌙 Dark" : "☀️ Light";
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
}

function initTheme() {
  let theme = "light";
  try { theme = localStorage.getItem(THEME_KEY) || "light"; } catch { /* ignore */ }
  if (theme !== "dark" && theme !== "light") theme = "light";
  applyTheme(theme);
  $("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") === "dark"
      ? "dark" : "light";
    applyTheme(current === "dark" ? "light" : "dark");
  });
}

// ---------- Sliders: sync range <-> number ----------
function initSliders() {
  for (const key of FEATURE_KEYS) {
    const range = $(`${key}-range`);
    const num = $(key);
    if (!range || !num) continue;
    range.addEventListener("input", () => {
      num.value = range.value;
      clearError();
    });
    num.addEventListener("input", () => {
      const v = parseFloat(num.value);
      if (isFinite(v)) range.value = String(v);
    });
    num.addEventListener("change", () => {
      const min = parseFloat(num.min), max = parseFloat(num.max);
      const v = parseFloat(num.value);
      if (isFinite(v) && v >= min && v <= max) range.value = String(v);
    });
  }
}

// ---------- Presets ----------
function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;
  for (const key of FEATURE_KEYS) {
    const num = $(key);
    const range = $(`${key}-range`);
    const value = preset[key];
    if (value === undefined) continue;
    num.value = String(value);
    if (range) range.value = String(value);
  }
  clearError();
}

// ---------- Rendering ----------
function renderResult(data, values) {
  const card = $("result-card");
  card.hidden = false;
  card.classList.remove("show", "pred-poor", "pred-average", "pred-good", "pred-excellent");
  const cls = String(data.prediction).toLowerCase();
  card.classList.add("show", `pred-${cls}`);

  $("result-prediction").textContent = data.prediction;
  $("result-confidence").textContent = `${data.confidence}%`;
  $("result-explanation").textContent = data.explanation;

  const summary = $("input-summary");
  summary.innerHTML = "";
  for (const key of FEATURE_KEYS) {
    const li = document.createElement("li");
    li.textContent = `${FEATURE_LABELS[key]}: ${values[key]}`;
    summary.appendChild(li);
  }

  // Animate the confidence bar from 0% to the real value.
  const bar = $("confidence-bar");
  bar.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { bar.style.width = `${data.confidence}%`; });
  });

  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderHistory() {
  const history = loadHistory();
  const list = $("history-list");
  const empty = $("history-empty");
  list.innerHTML = "";
  if (history.length === 0) {
    list.hidden = true;
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  list.hidden = false;
  history.forEach((entry) => {
    const cls = String(entry.prediction).toLowerCase();
    const card = document.createElement("div");
    card.className = `hist-card hp-${cls}`;
    const when = new Date(entry.time).toLocaleString();
    const top = document.createElement("div");
    top.className = "hist-top";
    top.innerHTML =
      `<span class="hist-pred">${entry.prediction}</span>` +
      `<span class="hist-conf">${entry.confidence}%</span>`;
    const meta = document.createElement("div");
    meta.className = "hist-meta";
    meta.textContent = when;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "hist-restore";
    btn.textContent = "↺ Restore inputs";
    btn.addEventListener("click", () => restoreEntry(entry));
    card.appendChild(top);
    card.appendChild(meta);
    card.appendChild(btn);
    list.appendChild(card);
  });
}

function restoreEntry(entry) {
  if (!entry || !entry.inputs) return;
  for (const key of FEATURE_KEYS) {
    const v = entry.inputs[key];
    if (v === undefined || v === "") continue;
    $(key).value = String(v);
    const range = $(`${key}-range`);
    if (range) range.value = String(v);
  }
  clearError();
  $("input-section").scrollIntoView({ behavior: "smooth", block: "start" });
}


function fmtPercent(x) { return `${(x * 100).toFixed(1)}%`; }

function renderMetrics(metrics) {
  $("m-accuracy").textContent = fmtPercent(metrics.accuracy);
  $("m-precision").textContent = fmtPercent(metrics.precision);
  $("m-recall").textContent = fmtPercent(metrics.recall);
  $("m-f1").textContent = fmtPercent(metrics.f1_score);
  $("metrics-extra").textContent =
    `${metrics.model} · trained on ${metrics.training_rows} rows · evaluated on ` +
    `${metrics.test_rows} unseen test rows · tree depth ${metrics.tree_depth} with ` +
    `${metrics.tree_leaves} decision leaves · classes: ${metrics.classes.join(", ")}`;
}

function renderDatasetInfo(data) {
  const stats = $("dataset-stats");
  const dist = data.class_distribution;
  stats.innerHTML = "";
  const items = [
    { label: "Total rows", value: String(data.total_rows) },
    { label: "Features (10)", value: data.features.length },
    { label: "Target classes (4)", value: data.classes.join(", ") },
  ];
  items.forEach((it) => {
    const div = document.createElement("div");
    div.className = "dataset-stat";
    div.innerHTML = `<span class="ds-label">${it.label}</span>` +
      `<span class="ds-value">${it.value}</span>`;
    stats.appendChild(div);
  });

  const distBlock = $("dataset-dist");
  distBlock.innerHTML = "";
  const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
  Object.keys(dist).forEach((k) => {
    const pct = Math.round((dist[k] / total) * 100);
    const row = document.createElement("div");
    row.className = "dist-bar-row";
    row.innerHTML =
      `<span class="ds-name">${k}</span>` +
      `<span class="dist-track"><span class="dist-bar b-${k.toLowerCase()}" style="width:0%"></span></span>` +
      `<span class="ds-count">${dist[k]} (${pct}%)</span>`;
    distBlock.appendChild(row);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        row.querySelector(".dist-bar").style.width = `${pct}%`;
      });
    });
  });
}

// ---------- Actions ----------
async function handlePredict(event) {
  event.preventDefault();
  clearError();
  $("result-card").hidden = true;

  const values = getFormValues();
  const errors = validateClient(values);
  if (errors.length > 0) {
    showError(errors.join(" "));
    return;
  }

  const btn = $("predict-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Predicting…';
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || `Server error (HTTP ${res.status}).`);
      return;
    }
    renderResult(data, values);
    saveHistoryEntry({
      prediction: data.prediction,
      confidence: data.confidence,
      inputs: values,
      time: new Date().toISOString(),
    });
    renderHistory();
  } catch (err) {
    showError("Could not reach the server. Is the Flask app running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "🔮 Predict Performance";
  }
}

function handleReset() {
  for (const key of FEATURE_KEYS) {
    $(key).value = "";
    const range = $(`${key}-range`);
    if (range) range.value = "0";
  }
  clearError();
  $("result-card").hidden = true;
  $("attendance").focus();
}

function handleClearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
}

// ---------- Card reveal on scroll (subtle, reduced-motion safe) ----------
function initReveal() {
  const cards = document.querySelectorAll(".card.reveal");
  if (!("IntersectionObserver" in window)) {
    cards.forEach((c) => c.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    });
  }, { threshold: 0.08 });
  cards.forEach((c) => io.observe(c));
}

// ---------- Init ----------
async function init() {
  initTheme();
  initSliders();
  $("predict-form").addEventListener("submit", handlePredict);
  $("reset-btn").addEventListener("click", handleReset);
  $("clear-history-btn").addEventListener("click", handleClearHistory);
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });
  renderHistory();

  // Reveal cards as they enter the viewport.
  if ("IntersectionObserver" in window) initReveal();
  else document.querySelectorAll(".card.reveal").forEach((c) => c.classList.add("in"));

  try {
    const res = await fetch("/api/metrics");
    if (res.ok) renderMetrics(await res.json());
  } catch { /* metrics stay as dashes if unreachable */ }

  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const h = await res.json();
      if (!h.model_loaded) {
        showError("Model not loaded on server. Run train_model.py and restart the app.");
      }
    }
  } catch { /* ignore */ }

  try {
    const res = await fetch("/api/metrics");
    if (res.ok) {
      const m = await res.json();
      if (m.dataset) renderDatasetInfo(m.dataset);
    }
  } catch { /* ignore */ }
}

document.addEventListener("DOMContentLoaded", init);

