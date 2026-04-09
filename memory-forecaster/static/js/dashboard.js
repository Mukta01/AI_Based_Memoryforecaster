/* =============================================================
   dashboard.js — Client-Side Logic for AI Memory Forecaster
   ============================================================= */

// ============================
// Tab Navigation
// ============================
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    tabBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    tabContents.forEach(tc => {
      tc.classList.remove('active');
      if (tc.id === 'tab-' + target) {
        tc.classList.add('active');
      }
    });
  });
});

// ============================
// State
// ============================
let statusData = {};
let decisionsChart = null;

// ============================
// API Helpers
// ============================
async function apiGet(url) {
  try {
    const res = await fetch(url);
    return await res.json();
  } catch (e) {
    console.error('API GET error:', url, e);
    return null;
  }
}

async function apiPost(url) {
  try {
    const res = await fetch(url, { method: 'POST' });
    return await res.json();
  } catch (e) {
    console.error('API POST error:', url, e);
    return null;
  }
}

// ============================
// Pipeline Stepper
// ============================
function updateStepper(status) {
  const steps = ['collect', 'features', 'train', 'simulate', 'evaluate'];
  let lastDone = -1;

  steps.forEach((step, idx) => {
    const el = document.getElementById('pip-' + step);
    const arrow = el.previousElementSibling;
    if (status[step]) {
      el.classList.add('done');
      el.classList.remove('active');
      if (arrow && arrow.classList.contains('pip-arrow')) {
        arrow.classList.add('done');
      }
      lastDone = idx;
    } else {
      el.classList.remove('done', 'active');
      if (arrow && arrow.classList.contains('pip-arrow')) {
        arrow.classList.remove('done');
      }
    }
  });

  // Mark the next step as active
  if (lastDone + 1 < steps.length) {
    const nextEl = document.getElementById('pip-' + steps[lastDone + 1]);
    if (!nextEl.classList.contains('done')) {
      nextEl.classList.add('active');
    }
  }
}

// ============================
// Live Memory
// ============================
async function updateLiveMemory() {
  const data = await apiGet('/api/live');
  if (!data) return;

  const pct = data.mem_pct || 0;
  const used = data.used_mb || 0;
  const avail = data.avail_mb || 0;
  const total = data.total_mb || 0;

  document.getElementById('live-mem-pct').textContent = pct.toFixed(1) + '%';
  document.getElementById('gauge-pct').textContent = pct.toFixed(1);
  document.getElementById('live-used').textContent = used.toFixed(0) + ' MB';
  document.getElementById('live-avail').textContent = avail.toFixed(0) + ' MB';
  document.getElementById('live-total').textContent = total.toFixed(0) + ' MB';

  // Update gauge circle
  const circle = document.getElementById('gauge-circle');
  const circumference = 2 * Math.PI * 60; // r=60
  const offset = circumference * (1 - pct / 100);
  circle.style.strokeDashoffset = offset;

  // Colour the gauge based on usage
  if (pct > 95) {
    circle.style.stroke = 'var(--rose)';
  } else if (pct > 85) {
    circle.style.stroke = 'var(--amber)';
  } else {
    circle.style.stroke = 'var(--cyan)';
  }
}

// ============================
// Load Status & Populate Data
// ============================
async function loadStatus() {
  const data = await apiGet('/api/status');
  if (!data) return;
  statusData = data;
  updateStepper(data);

  // Overview stats
  if (data.collect_rows) {
    document.getElementById('stat-rows').textContent = data.collect_rows.toLocaleString();
    document.getElementById('collect-rows').textContent = data.collect_rows.toLocaleString();
  }
  if (data.feature_rows) {
    document.getElementById('stat-features').textContent = data.feature_cols || '—';
    document.getElementById('feat-rows').textContent = data.feature_rows.toLocaleString();
    document.getElementById('feat-cols').textContent = data.feature_cols || '—';
  }
  if (data.train) {
    document.getElementById('stat-models').textContent = '2';
  }
  if (data.decision_count) {
    document.getElementById('stat-decisions').textContent = data.decision_count.toLocaleString();
  }

  // Load sub-sections if data exists
  if (data.collect) loadRawData();
  if (data.features) loadFeatureData();
  if (data.train) loadTrainingPlots();
  if (data.simulate) loadDecisionData();
  if (data.evaluate) loadEvaluationData();
}

// ============================
// Raw Data Table
// ============================
async function loadRawData() {
  const data = await apiGet('/api/data/memory_log');
  if (!data || !data.rows || data.rows.length === 0) return;

  const headers = data.columns || Object.keys(data.rows[0]);
  // Show only key columns for readability
  const showCols = ['timestamp', 'used_mb', 'avail_mb', 'mem_pct', 'name1', 'rss1', 'name2', 'rss2'];
  const filteredHeaders = showCols.filter(c => headers.includes(c));

  let html = '<table class="data-table"><thead><tr>';
  filteredHeaders.forEach(h => { html += '<th>' + h + '</th>'; });
  html += '</tr></thead><tbody>';

  const displayRows = data.rows.slice(0, 20);
  displayRows.forEach(row => {
    html += '<tr>';
    filteredHeaders.forEach(h => {
      let val = row[h];
      if (h === 'timestamp' && typeof val === 'number') {
        val = new Date(val * 1000).toLocaleTimeString();
      }
      if (typeof val === 'number') val = val.toFixed ? val.toFixed(2) : val;
      html += '<td>' + (val !== null && val !== undefined ? val : '—') + '</td>';
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  if (data.rows.length > 20) {
    html += '<p class="metric-sub text-center mt-20">Showing 20 of ' + data.rows.length + ' rows</p>';
  }

  document.getElementById('raw-data-table').innerHTML = html;
  if (window.lucide) lucide.createIcons();
}

// ============================
// Feature Data
// ============================
async function loadFeatureData() {
  const data = await apiGet('/api/data/features');
  if (!data) return;

  if (data.rows) {
    document.getElementById('feat-rows').textContent = (data.total_rows || data.rows.length).toLocaleString();
  }
  if (data.columns) {
    document.getElementById('feat-cols').textContent = data.columns.length;

    let html = '<div class="feature-list">';
    data.columns.forEach(col => {
      const highlight = col === 'y' ? 'style="border-color:var(--cyan);color:var(--cyan);"' : '';
      html += '<span class="feature-chip" ' + highlight + '>' + col + '</span>';
    });
    html += '</div>';
    document.getElementById('features-column-list').innerHTML = html;
    if (window.lucide) lucide.createIcons();
  }
  if (data.dropped) {
    document.getElementById('feat-dropped').textContent = data.dropped;
  }
}

// ============================
// Training Plots
// ============================
function loadTrainingPlots() {
  showPlot('rf-results-img', 'rf-results-empty', '/api/plots/rf_results.png');
  showPlot('rf-importances-img', 'rf-importances-empty', '/api/plots/rf_importances.png');
  showPlot('lstm-results-img', 'lstm-results-empty', '/api/plots/lstm_results.png');
  showPlot('lstm-loss-img', 'lstm-loss-empty', '/api/plots/lstm_loss.png');

  // Load metrics if available
  loadTrainingMetrics();
}

async function loadTrainingMetrics() {
  const data = await apiGet('/api/status');
  if (!data) return;

  if (data.rf_metrics) {
    document.getElementById('rf-mae').textContent = data.rf_metrics.mae.toFixed(2);
    document.getElementById('rf-rmse').textContent = data.rf_metrics.rmse.toFixed(2);
    document.getElementById('rf-mape').textContent = data.rf_metrics.mape.toFixed(2);
  }
  if (data.lstm_metrics) {
    document.getElementById('lstm-mae').textContent = data.lstm_metrics.mae.toFixed(2);
    document.getElementById('lstm-rmse').textContent = data.lstm_metrics.rmse.toFixed(2);
    document.getElementById('lstm-mape').textContent = data.lstm_metrics.mape.toFixed(2);
  }
}

function showPlot(imgId, emptyId, src) {
  const img = document.getElementById(imgId);
  const empty = document.getElementById(emptyId);
  img.src = src + '?t=' + Date.now();
  img.style.display = 'block';
  img.onerror = () => { img.style.display = 'none'; if (empty) empty.style.display = 'block'; };
  if (empty) empty.style.display = 'none';
}

// ============================
// Decision Data
// ============================
async function loadDecisionData() {
  const data = await apiGet('/api/data/decisions');
  if (!data) return;

  // Show timeline plot
  showPlot('decisions-timeline-img', 'decisions-timeline-empty', '/api/plots/decisions_timeline.png');

  // Build donut chart
  if (data.summary) {
    renderDecisionDonut(data.summary);
    renderDecisionTable(data.summary, data.total || 0);
  }
}

function renderDecisionDonut(summary) {
  const canvas = document.getElementById('decisions-donut-chart');
  if (!canvas) return;

  if (decisionsChart) decisionsChart.destroy();

  const labels = Object.keys(summary);
  const values = Object.values(summary);
  const colorMap = {
    'none': '#64748b',
    'prealloc': '#38bdf8',
    'swap_early': '#f59e0b',
    'throttle_oom': '#f43f5e'
  };
  const colors = labels.map(l => colorMap[l] || '#94a3b8');

  decisionsChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: '#0a0f1e',
        borderWidth: 3,
        hoverBorderColor: '#e2e8f0',
        hoverBorderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#94a3b8',
            padding: 16,
            font: { family: 'Inter', size: 12 },
            usePointStyle: true,
            pointStyleWidth: 12,
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.95)',
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(6,182,212,0.3)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
        }
      }
    }
  });
}

function renderDecisionTable(summary, total) {
  const container = document.getElementById('decision-summary-table');
  let html = '<div class="data-table-wrap"><table class="data-table"><thead><tr><th>Action</th><th>Count</th><th>Percentage</th></tr></thead><tbody>';
  Object.entries(summary).forEach(([action, count]) => {
    const pct = total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
    const badgeClass = {
      'none': '',
      'prealloc': 'badge-cyan',
      'swap_early': 'badge-amber',
      'throttle_oom': 'badge-rose'
    }[action] || '';
    html += '<tr><td><span class="badge ' + badgeClass + '">' + action + '</span></td>';
    html += '<td>' + count.toLocaleString() + '</td>';
    html += '<td>' + pct + '%</td></tr>';
  });
  html += '</tbody></table></div>';
  container.innerHTML = html;
  if (window.lucide) lucide.createIcons();
}

// ============================
// Evaluation Data
// ============================
async function loadEvaluationData() {
  const data = await apiGet('/api/status');
  if (!data) return;

  // Show comparison plot
  showPlot('comparison-chart-img', 'comparison-chart-empty', '/api/plots/model_comparison.png');

  if (data.rf_metrics) {
    document.getElementById('eval-rf-mae').textContent = data.rf_metrics.mae.toFixed(2);
    document.getElementById('eval-rf-rmse').textContent = data.rf_metrics.rmse.toFixed(2);
    document.getElementById('eval-rf-mape').textContent = data.rf_metrics.mape.toFixed(2);
  }
  if (data.lstm_metrics) {
    document.getElementById('eval-lstm-mae').textContent = data.lstm_metrics.mae.toFixed(2);
    document.getElementById('eval-lstm-rmse').textContent = data.lstm_metrics.rmse.toFixed(2);
    document.getElementById('eval-lstm-mape').textContent = data.lstm_metrics.mape.toFixed(2);
  }

  // Winner
  if (data.rf_metrics && data.lstm_metrics) {
    const winnerCard = document.getElementById('winner-card');
    winnerCard.style.display = 'block';

    const rfMae = data.rf_metrics.mae;
    const lstmMae = data.lstm_metrics.mae;

    if (rfMae < lstmMae) {
      document.getElementById('winner-name').textContent = 'Random Forest';
      document.getElementById('winner-detail').textContent =
        'Wins by ' + (lstmMae - rfMae).toFixed(2) + ' MB MAE';
    } else {
      document.getElementById('winner-name').textContent = 'LSTM';
      document.getElementById('winner-detail').textContent =
        'Wins by ' + (rfMae - lstmMae).toFixed(2) + ' MB MAE';
    }
  }
}

// ============================
// Run Pipeline Step
// ============================
async function runPipeline(mode) {
  const btnMap = {
    'collect': 'btn-collect',
    'features': 'btn-features',
    'train': 'btn-train',
    'simulate': 'btn-simulate',
    'evaluate': 'btn-evaluate'
  };

  const btn = document.getElementById(btnMap[mode]);
  if (!btn) return;

  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';

  const result = await apiPost('/api/' + mode);

  btn.disabled = false;
  btn.innerHTML = originalText;

  if (result && result.success) {
    // Refresh everything
    await loadStatus();

    // Switch to the relevant tab content based on mode
    if (mode === 'collect') loadRawData();
    if (mode === 'features') loadFeatureData();
    if (mode === 'train') loadTrainingPlots();
    if (mode === 'simulate') loadDecisionData();
    if (mode === 'evaluate') loadEvaluationData();
    
    // Refresh icons in case any new ones were added
    if (window.lucide) lucide.createIcons();
  } else {
    const msg = result && result.error ? result.error : 'Unknown error';
    alert('Pipeline error (' + mode + '): ' + msg);
  }
}

// ============================
// Initialization
// ============================
document.addEventListener('DOMContentLoaded', () => {
  loadStatus();
  updateLiveMemory();

  // Initialize icons
  if (window.lucide) lucide.createIcons();

  // Poll live memory every 3 seconds
  setInterval(updateLiveMemory, 3000);
});
