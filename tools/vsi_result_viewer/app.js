const state = {
  manifest: null,
  run: null,
  rows: [],
  filtered: [],
  page: 1,
};

const labels = {
  correct: "正确",
  partial: "部分",
  wrong: "错误",
  empty: "空回答",
  unknown: "未知",
};

const els = {
  runSelect: document.querySelector("#runSelect"),
  runMeta: document.querySelector("#runMeta"),
  summary: document.querySelector("#summary"),
  typeFilter: document.querySelector("#typeFilter"),
  statusFilter: document.querySelector("#statusFilter"),
  datasetFilter: document.querySelector("#datasetFilter"),
  minScore: document.querySelector("#minScore"),
  searchInput: document.querySelector("#searchInput"),
  resetButton: document.querySelector("#resetButton"),
  typeGrid: document.querySelector("#typeGrid"),
  resultCount: document.querySelector("#resultCount"),
  sortSelect: document.querySelector("#sortSelect"),
  pageSize: document.querySelector("#pageSize"),
  resultBody: document.querySelector("#resultBody"),
  prevPage: document.querySelector("#prevPage"),
  nextPage: document.querySelector("#nextPage"),
  pageInfo: document.querySelector("#pageInfo"),
  rowTemplate: document.querySelector("#rowTemplate"),
};

function text(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function scoreText(value) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function pct(num, den) {
  if (!den) return "0.0%";
  return `${((num / den) * 100).toFixed(1)}%`;
}

function setOptions(select, values, current = "") {
  const first = select.querySelector("option[value='']");
  select.replaceChildren(first || new Option("全部", ""));
  values.forEach((value) => select.append(new Option(value, value)));
  select.value = current;
}

function statusBadge(status) {
  const span = document.createElement("span");
  span.className = `status status-${status}`;
  span.textContent = labels[status] || status;
  return span;
}

async function loadManifest() {
  const res = await fetch("./data/runs.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`data/runs.json ${res.status}`);
  state.manifest = await res.json();
  els.runSelect.replaceChildren();
  state.manifest.runs.forEach((run) => {
    const option = new Option(`${run.label} (${run.total})`, run.id);
    els.runSelect.append(option);
  });
  if (state.manifest.runs.length) {
    await loadRun(state.manifest.runs[0].id);
  }
}

async function loadRun(runId) {
  const meta = state.manifest.runs.find((run) => run.id === runId);
  if (!meta) return;
  const res = await fetch(`./data/${meta.data_file}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${meta.data_file} ${res.status}`);
  state.run = await res.json();
  state.rows = state.run.samples;
  state.page = 1;
  els.runSelect.value = runId;
  renderRunMeta();
  refreshFilterOptions();
  renderTypeGrid();
  applyFilters();
}

function renderRunMeta() {
  const run = state.run.run;
  const overall = state.run.metrics?.overall;
  const pieces = [
    `${state.run.summary.total} samples`,
    overall === undefined ? "" : `overall ${Number(overall).toFixed(3)}`,
    run.model_args || "",
  ].filter(Boolean);
  els.runMeta.textContent = pieces.join(" | ");
}

function renderSummary(rows = state.rows) {
  const total = rows.length;
  const counts = rows.reduce((acc, row) => {
    acc[row.status] = (acc[row.status] || 0) + 1;
    return acc;
  }, {});
  const avg = total
    ? rows.reduce((sum, row) => sum + (typeof row.score === "number" ? row.score : 0), 0) / total
    : 0;
  const items = [
    ["总数", total],
    ["正确", `${counts.correct || 0} (${pct(counts.correct || 0, total)})`],
    ["部分得分", `${counts.partial || 0} (${pct(counts.partial || 0, total)})`],
    ["错误", `${counts.wrong || 0} (${pct(counts.wrong || 0, total)})`],
    ["空回答", `${counts.empty || 0} (${pct(counts.empty || 0, total)})`],
    ["平均单题分", avg.toFixed(3)],
  ];
  els.summary.replaceChildren(
    ...items.map(([name, value]) => {
      const div = document.createElement("div");
      div.className = "metric";
      div.innerHTML = `<span>${name}</span><strong>${value}</strong>`;
      return div;
    }),
  );
}

function refreshFilterOptions() {
  const types = [...new Set(state.rows.map((row) => row.question_type).filter(Boolean))].sort();
  const datasets = [...new Set(state.rows.map((row) => row.dataset).filter(Boolean))].sort();
  setOptions(els.typeFilter, types);
  setOptions(els.datasetFilter, datasets);
  els.statusFilter.value = "";
  els.minScore.value = "";
  els.searchInput.value = "";
}

function renderTypeGrid() {
  const summary = state.run.summary.by_type || [];
  els.typeGrid.replaceChildren(
    ...summary.map((item) => {
      const div = document.createElement("button");
      div.type = "button";
      div.className = "type-stat";
      div.dataset.type = item.name;
      div.innerHTML = `
        <h3>${item.name}</h3>
        <dl>
          <div><dt>Total</dt><dd>${item.total}</dd></div>
          <div><dt>Correct</dt><dd>${item.correct}</dd></div>
          <div><dt>Avg</dt><dd>${item.avg_score.toFixed(3)}</dd></div>
        </dl>
      `;
      div.addEventListener("click", () => {
        els.typeFilter.value = item.name;
        state.page = 1;
        applyFilters();
      });
      return div;
    }),
  );
}

function applyFilters() {
  const type = els.typeFilter.value;
  const status = els.statusFilter.value;
  const dataset = els.datasetFilter.value;
  const minScore = els.minScore.value === "" ? null : Number(els.minScore.value);
  const query = els.searchInput.value.trim().toLowerCase();

  state.filtered = state.rows.filter((row) => {
    if (type && row.question_type !== type) return false;
    if (status && row.status !== status) return false;
    if (dataset && row.dataset !== dataset) return false;
    if (minScore !== null && (typeof row.score !== "number" || row.score < minScore)) return false;
    if (query) {
      const haystack = [
        row.id,
        row.doc_id,
        row.question_type,
        row.dataset,
        row.scene_name,
        row.question,
        row.ground_truth,
        row.prediction,
      ]
        .map(text)
        .join("\n")
        .toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  });

  sortRows();
  renderSummary(state.filtered);
  renderTable();
}

function sortRows() {
  const [field, direction] = els.sortSelect.value.split(":");
  const sign = direction === "desc" ? -1 : 1;
  state.filtered.sort((a, b) => {
    let av;
    let bv;
    if (field === "score") {
      av = typeof a.score === "number" ? a.score : -1;
      bv = typeof b.score === "number" ? b.score : -1;
    } else if (field === "type") {
      av = a.question_type;
      bv = b.question_type;
    } else {
      av = Number(a.id);
      bv = Number(b.id);
    }
    if (av < bv) return -1 * sign;
    if (av > bv) return 1 * sign;
    return 0;
  });
}

function renderTable() {
  const pageSize = Number(els.pageSize.value);
  const totalPages = Math.max(1, Math.ceil(state.filtered.length / pageSize));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * pageSize;
  const rows = state.filtered.slice(start, start + pageSize);

  els.resultBody.replaceChildren();
  const fragment = document.createDocumentFragment();
  rows.forEach((row) => {
    const clone = els.rowTemplate.content.cloneNode(true);
    clone.querySelector("[data-field='id']").textContent = text(row.id);
    clone.querySelector("[data-field='type']").textContent = row.question_type;
    const statusCell = clone.querySelector("[data-field='status']");
    statusCell.replaceChildren(statusBadge(row.status));
    clone.querySelector("[data-field='score']").textContent = scoreText(row.score);
    clone.querySelector("[data-field='question']").textContent = text(row.question);
    clone.querySelector("[data-field='groundTruth']").textContent = text(row.ground_truth);
    clone.querySelector("[data-field='prediction']").textContent = text(row.prediction);
    clone.querySelector("[data-field='scene']").textContent = `${text(row.dataset)} / ${text(row.scene_name)}`;
    clone.querySelector("[data-field='prompt']").textContent = text(row.prompt);
    clone.querySelector("[data-field='options']").textContent = text(row.options);
    fragment.append(clone);
  });
  els.resultBody.append(fragment);

  els.resultCount.textContent = `${state.filtered.length} 条结果`;
  els.pageInfo.textContent = `${state.page} / ${totalPages}`;
  els.prevPage.disabled = state.page <= 1;
  els.nextPage.disabled = state.page >= totalPages;
}

function bindEvents() {
  els.runSelect.addEventListener("change", (event) => loadRun(event.target.value));
  [els.typeFilter, els.statusFilter, els.datasetFilter, els.minScore, els.searchInput].forEach((el) => {
    el.addEventListener("input", () => {
      state.page = 1;
      applyFilters();
    });
  });
  els.sortSelect.addEventListener("input", () => {
    state.page = 1;
    applyFilters();
  });
  els.pageSize.addEventListener("input", () => {
    state.page = 1;
    renderTable();
  });
  els.resetButton.addEventListener("click", () => {
    refreshFilterOptions();
    state.page = 1;
    applyFilters();
  });
  els.prevPage.addEventListener("click", () => {
    state.page -= 1;
    renderTable();
  });
  els.nextPage.addEventListener("click", () => {
    state.page += 1;
    renderTable();
  });
}

bindEvents();
loadManifest().catch((error) => {
  els.runMeta.textContent = `数据加载失败: ${error.message}`;
});
