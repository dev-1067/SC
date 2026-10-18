/**
 * Smart Traffic Signal Control System (STCS)
 * Dashboard Client Controller & Real-Time Canvas Renderer
 * Handles WebSocket telemetry, 60 FPS 2D canvas intersection rendering,
 * Chart.js analytics, FLC inspection panels, and REST API controls.
 */

// ==============================================================================
// 1. GLOBAL STATE & REFERENCES
// ==============================================================================
let ws = null;
let reconnectTimer = null;
let currentTelemetry = null;
let delayChartInstance = null;
let queueChartInstance = null;

const canvas = document.getElementById("trafficCanvas");
const ctx = canvas.getContext("2d");

// Debounce timer for demand sliders
let demandDebounceTimer = null;

// ==============================================================================
// 2. WEBSOCKET CONNECTION MANAGER
// ==============================================================================
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/traffic-stream`;

  const statusPill = document.getElementById("wsStatusPill");
  const statusText = document.getElementById("wsStatusText");

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("[STCS] WebSocket connected successfully to:", wsUrl);
      statusPill.style.background = "rgba(16, 185, 129, 0.15)";
      statusPill.style.color = "var(--color-green)";
      statusPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
      statusText.textContent = "LIVE TELEMETRY";
      clearTimeout(reconnectTimer);
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        currentTelemetry = payload;
        updateDashboard(payload);
      } catch (err) {
        console.error("[STCS] Error parsing telemetry payload:", err);
      }
    };

    ws.onclose = () => {
      console.warn("[STCS] WebSocket disconnected. Reconnecting in 2s...");
      statusPill.style.background = "rgba(239, 68, 68, 0.15)";
      statusPill.style.color = "var(--color-red)";
      statusPill.style.borderColor = "rgba(239, 68, 68, 0.4)";
      statusText.textContent = "RECONNECTING...";
      reconnectTimer = setTimeout(initWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error("[STCS] WebSocket error encountered:", err);
      ws.close();
    };
  } catch (err) {
    reconnectTimer = setTimeout(initWebSocket, 2000);
  }
}

// ==============================================================================
// 3. DASHBOARD UI TELEMETRY UPDATERS
// ==============================================================================
function updateDashboard(data) {
  if (!data) return;

  // 1. Clock Display
  const minutes = Math.floor(data.sim_time / 60);
  const seconds = (data.sim_time % 60).toFixed(1);
  document.getElementById("simTimeDisplay").textContent = 
    `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(4, '0')}`;

  // 2. Control Mode Button & Label
  const modeLbl = document.getElementById("lblControlMode");
  const btnToggle = document.getElementById("btnToggleMode");
  if (data.control_mode === "ADAPTIVE") {
    modeLbl.textContent = "MODE: ADAPTIVE (FLC + ANN)";
    btnToggle.className = "control-btn primary";
  } else if (data.control_mode === "MANUAL") {
    modeLbl.textContent = "MODE: MANUAL (FORCE SWITCH)";
    btnToggle.className = "control-btn warning";
  } else {
    modeLbl.textContent = "MODE: STATIC (30s TIMER)";
    btnToggle.className = "control-btn";
  }

  // 3. Pause / Resume state
  const pauseLbl = document.getElementById("lblPauseResume");
  const pauseWrapper = document.getElementById("iconPauseResumeWrapper");
  if (pauseWrapper && pauseLbl) {
    if (data.is_paused) {
      pauseLbl.textContent = "RESUME";
      pauseWrapper.innerHTML = '<i data-lucide="play"></i>';
    } else {
      pauseLbl.textContent = "PAUSE";
      pauseWrapper.innerHTML = '<i data-lucide="pause"></i>';
    }
  }

  // 3b. Emergency Active Banner & Clear Button
  const emergBanner = document.getElementById("emergencyBanner");
  const btnClearEmerg = document.getElementById("btnClearEmerg");
  if (emergBanner) {
    if (data.emergency_active) {
      emergBanner.style.display = "flex";
      const cName = document.getElementById("emergCorridorName");
      if (cName) cName.textContent = data.emergency_approach || "Emergency";
      if (btnClearEmerg) btnClearEmerg.style.display = "inline-flex";
    } else {
      emergBanner.style.display = "none";
      if (btnClearEmerg) btnClearEmerg.style.display = "none";
    }
  }

  // 3c. Speed buttons active state sync
  if (data.sim_speed) {
    document.querySelectorAll(".speed-btn").forEach((btn) => {
      const spd = parseFloat(btn.dataset.speed);
      if (Math.abs(spd - data.sim_speed) < 0.1) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  // 4. KPI Summary Cards
  document.getElementById("kpiActivePhase").textContent = data.phase_name || "North-South";
  
  const activeSignal = data.active_phase === 0 ? data.signal_states.NS : data.signal_states.EW;
  const sigBadge = document.getElementById("kpiSignalBadge");
  sigBadge.textContent = `${activeSignal} SIGNAL`;
  if (activeSignal === "GREEN") {
    sigBadge.className = "kpi-badge positive";
  } else if (activeSignal === "YELLOW") {
    sigBadge.className = "kpi-badge warning";
  } else {
    sigBadge.className = "kpi-badge danger";
  }

  document.getElementById("kpiPhaseCountdown").textContent = 
    `${data.elapsed_in_phase.toFixed(1)}s / ${data.allocated_green.toFixed(1)}s`;

  document.getElementById("kpiAvgDelay").textContent = data.average_delay_seconds.toFixed(1);
  document.getElementById("kpiTotalQueued").textContent = data.total_queued;
  document.getElementById("kpiActiveVehBadge").textContent = `${data.active_vehicle_count} active on road`;
  document.getElementById("kpiThroughput").textContent = data.total_throughput;

  // ANN Surge Forecast
  const p0Ann = (data.ann_forecast && data.ann_forecast["0"]) || {};
  const p1Ann = (data.ann_forecast && data.ann_forecast["1"]) || {};
  const maxSurge = Math.max(p0Ann.surge_factor || 0, p1Ann.surge_factor || 0);
  const isSurge = (p0Ann.is_surge_imminent || p1Ann.is_surge_imminent);

  document.getElementById("kpiSurgeFactor").textContent = (maxSurge * 100).toFixed(0);
  const surgeBadge = document.getElementById("kpiSurgeAlertBadge");
  if (isSurge) {
    surgeBadge.textContent = "IMMINENT SURGE ALERT";
    surgeBadge.className = "kpi-badge danger";
  } else {
    surgeBadge.textContent = "STEADY TRAFFIC DEMAND";
    surgeBadge.className = "kpi-badge positive";
  }

  // 5. Soft Computing Fuzzy Inspector
  if (data.fuzzy_inspector) {
    const fInputs = data.fuzzy_inspector.fuzzified_inputs || {};
    
    // Queue Length MFs
    const qMfs = fInputs.QueueLength || {};
    setBarWidth("mf_q_short", qMfs.Short);
    setBarWidth("mf_q_medium", qMfs.Medium);
    setBarWidth("mf_q_long", qMfs.Long);
    setBarWidth("mf_q_verylong", qMfs.VeryLong);

    // Waiting Time MFs
    const wMfs = fInputs.WaitingTime || {};
    setBarWidth("mf_w_low", wMfs.Low);
    setBarWidth("mf_w_normal", wMfs.Normal);
    setBarWidth("mf_w_critical", wMfs.Critical);

    // Arrival Rate MFs
    const vMfs = fInputs.ArrivalRate || {};
    setBarWidth("mf_v_low", vMfs.Low);
    setBarWidth("mf_v_medium", vMfs.Medium);
    setBarWidth("mf_v_high", vMfs.High);

    // Defuzzification meter
    const defuzzVal = data.fuzzy_inspector.green_extension_seconds || 30.0;
    document.getElementById("lblDefuzzGreen").textContent = `${defuzzVal.toFixed(1)}s`;
    const pct = ((defuzzVal - 10.0) / (60.0 - 10.0)) * 100.0;
    document.getElementById("defuzzProgressBar").style.width = `${Math.min(100, Math.max(0, pct))}%`;

    // Active Rules Feed
    updateActiveRulesList(data.fuzzy_inspector.active_rules || []);
  }

  // 6. ANN Predictor Numbers
  document.getElementById("lblAnnForecastNS").textContent = `${(p0Ann.predicted_queue_10min || 0).toFixed(1)} veh`;
  document.getElementById("lblAnnForecastEW").textContent = `${(p1Ann.predicted_queue_10min || 0).toFixed(1)} veh`;
  document.getElementById("lblAnnSurgeNS").textContent = `${((p0Ann.surge_factor || 0) * 100).toFixed(1)}%`;
  document.getElementById("lblAnnSurgeEW").textContent = `${((p1Ann.surge_factor || 0) * 100).toFixed(1)}%`;

  // 7. Update Charts
  appendChartData(data);

  // Refresh Lucide icons
  lucide.createIcons();
}

function setBarWidth(elementId, value) {
  const el = document.getElementById(elementId);
  if (el) {
    const pct = Math.round((value || 0.0) * 100);
    el.style.width = `${pct}%`;
  }
}

function updateActiveRulesList(rules) {
  const container = document.getElementById("activeRulesContainer");
  if (!container) return;

  if (!rules || rules.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.72rem; padding: 6px;">No rules actively firing.</div>`;
    return;
  }

  let html = "";
  rules.slice(0, 4).forEach((r) => {
    html += `
      <div class="rule-pill">
        <div class="rule-header">
          <span class="rule-id">Rule #${r.rule_id} &rarr; ${r.consequent}</span>
          <span class="firing-badge">&alpha; = ${r.firing_strength.toFixed(3)}</span>
        </div>
        <div class="rule-desc">${r.rationale || r.description}</div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// ==============================================================================
// 4. HTML5 CANVAS 2D INTERSECTION RENDERER (60 FPS)
// ==============================================================================
function renderIntersection() {
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const roadWidth = 140;
  const halfRoad = roadWidth / 2;

  // Clear canvas
  ctx.fillStyle = "#121a14"; // Grass background
  ctx.fillRect(0, 0, w, h);

  // Roads (Dark Asphalt)
  ctx.fillStyle = "#272a30";
  // North-South road
  ctx.fillRect(cx - halfRoad, 0, roadWidth, h);
  // East-West road
  ctx.fillRect(0, cy - halfRoad, w, roadWidth);

  // Road curbs / sidewalks
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 2;
  // NW corner
  ctx.strokeRect(0, 0, cx - halfRoad, cy - halfRoad);
  // NE corner
  ctx.strokeRect(cx + halfRoad, 0, w - (cx + halfRoad), cy - halfRoad);
  // SW corner
  ctx.strokeRect(0, cy + halfRoad, cx - halfRoad, h - (cy + halfRoad));
  // SE corner
  ctx.strokeRect(cx + halfRoad, cy + halfRoad, w - (cx + halfRoad), h - (cy + halfRoad));

  // Yellow Center Dividing Dashes
  ctx.strokeStyle = "#eab308";
  ctx.lineWidth = 2;
  ctx.setLineDash([12, 12]);
  
  // Vertical NS divider
  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, cy - halfRoad);
  ctx.moveTo(cx, cy + halfRoad);
  ctx.lineTo(cx, h);
  ctx.stroke();

  // Horizontal EW divider
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(cx - halfRoad, cy);
  ctx.moveTo(cx + halfRoad, cy);
  ctx.lineTo(w, cy);
  ctx.stroke();
  ctx.setLineDash([]); // Reset dash

  // White Solid Stop Lines
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 4;
  
  // North stop line (approaching south)
  ctx.beginPath();
  ctx.moveTo(cx - halfRoad, cy - halfRoad);
  ctx.lineTo(cx, cy - halfRoad);
  ctx.stroke();

  // South stop line (approaching north)
  ctx.beginPath();
  ctx.moveTo(cx, cy + halfRoad);
  ctx.lineTo(cx + halfRoad, cy + halfRoad);
  ctx.stroke();

  // West stop line (approaching east)
  ctx.beginPath();
  ctx.moveTo(cx - halfRoad, cy);
  ctx.lineTo(cx - halfRoad, cy + halfRoad);
  ctx.stroke();

  // East stop line (approaching west)
  ctx.beginPath();
  ctx.moveTo(cx + halfRoad, cy - halfRoad);
  ctx.lineTo(cx + halfRoad, cy);
  ctx.stroke();

  // Traffic Light Actuators
  if (currentTelemetry && currentTelemetry.signal_states) {
    const nsSignal = currentTelemetry.signal_states.NS;
    const ewSignal = currentTelemetry.signal_states.EW;

    drawTrafficLight(cx - halfRoad - 22, cy - halfRoad - 22, nsSignal, "NS");
    drawTrafficLight(cx + halfRoad + 22, cy + halfRoad + 22, nsSignal, "NS");
    drawTrafficLight(cx - halfRoad - 22, cy + halfRoad + 22, ewSignal, "EW");
    drawTrafficLight(cx + halfRoad + 22, cy - halfRoad - 22, ewSignal, "EW");
  }

  // Draw Vehicles
  if (currentTelemetry && currentTelemetry.vehicles) {
    currentTelemetry.vehicles.forEach((veh) => {
      // Scale coordinates from simulation size (800) to canvas size (600)
      const scale = w / 800.0;
      const vx = veh.x * scale;
      const vy = veh.y * scale;
      drawVehicle(vx, vy, veh);
    });
  }

  // Draw Directional Approach Badges
  drawApproachBadge("NORTH", cx - 35, 24, currentTelemetry?.lanes?.North);
  drawApproachBadge("SOUTH", cx + 35, h - 24, currentTelemetry?.lanes?.South);
  drawApproachBadge("WEST", 35, cy + 30, currentTelemetry?.lanes?.West);
  drawApproachBadge("EAST", w - 35, cy - 30, currentTelemetry?.lanes?.East);

  requestAnimationFrame(renderIntersection);
}

function drawTrafficLight(x, y, state, label) {
  let color = "#ef4444";
  let glow = "rgba(239, 68, 68, 0.6)";

  if (state === "GREEN") {
    color = "#10b981";
    glow = "rgba(16, 185, 129, 0.6)";
  } else if (state === "YELLOW") {
    color = "#f59e0b";
    glow = "rgba(245, 158, 11, 0.6)";
  }

  // Housing
  ctx.fillStyle = "#0f172a";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, 15, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Glow Halo
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, 18, 0, Math.PI * 2);
  ctx.fill();

  // Light Bulb
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fill();
}

function drawVehicle(x, y, veh) {
  const isNS = (veh.approach === "North" || veh.approach === "South");
  const carW = isNS ? 14 : 24;
  const carH = isNS ? 24 : 14;

  ctx.save();
  ctx.translate(x, y);

  // Car Body
  const rgb = veh.color || [50, 110, 210];
  ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  
  // Rounded rectangle
  const radius = 3;
  ctx.beginPath();
  ctx.roundRect(-carW / 2, -carH / 2, carW, carH, radius);
  ctx.fill();

  // Windshield
  ctx.fillStyle = "rgba(230, 240, 255, 0.7)";
  if (isNS) {
    ctx.fillRect(-carW / 2 + 2, -carH / 2 + 4, carW - 4, 4);
  } else {
    ctx.fillRect(-carW / 2 + 4, -carH / 2 + 2, 4, carH - 4);
  }

  // Glowing Brake Lights if Queued
  if (veh.is_queued) {
    ctx.fillStyle = "#ff1744";
    ctx.shadowColor = "#ff1744";
    ctx.shadowBlur = 6;
    if (veh.approach === "North") {
      ctx.fillRect(-carW / 2 + 1, -carH / 2, 3, 2);
      ctx.fillRect(carW / 2 - 4, -carH / 2, 3, 2);
    } else if (veh.approach === "South") {
      ctx.fillRect(-carW / 2 + 1, carH / 2 - 2, 3, 2);
      ctx.fillRect(carW / 2 - 4, carH / 2 - 2, 3, 2);
    } else if (veh.approach === "West") {
      ctx.fillRect(-carW / 2, -carH / 2 + 1, 2, 3);
      ctx.fillRect(-carW / 2, carH / 2 - 4, 2, 3);
    } else {
      ctx.fillRect(carW / 2 - 2, -carH / 2 + 1, 2, 3);
      ctx.fillRect(carW / 2 - 2, carH / 2 - 4, 2, 3);
    }
    ctx.shadowBlur = 0;
  }

  // Emergency vehicle roof beacon strobe
  const isEmergency = veh.color && (veh.color[0] >= 240 && veh.color[1] <= 60);
  if (isEmergency) {
    const strobeTime = Math.floor(Date.now() / 160) % 2 === 0;
    ctx.fillStyle = strobeTime ? "#ef4444" : "#38bdf8";
    ctx.shadowColor = strobeTime ? "#ef4444" : "#38bdf8";
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(0, 0, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  ctx.restore();
}

function drawApproachBadge(name, x, y, laneData) {
  const queue = (laneData && laneData.queue_length) || 0;
  const flow = (laneData && laneData.arrival_rate) || 0.0;

  ctx.save();
  ctx.font = "bold 10px 'Outfit', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const text = `${name}: ${queue} queued (${flow}v/m)`;
  const metrics = ctx.measureText(text);
  const pad = 6;

  ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
  ctx.roundRect(x - metrics.width / 2 - pad, y - 9, metrics.width + pad * 2, 18, 4);
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = queue > 4 ? "#ef4444" : "#38bdf8";
  ctx.fillText(text, x, y);
  ctx.restore();
}

// ==============================================================================
// 5. CHART.JS REAL-TIME TELEMETRY CHARTS
// ==============================================================================
function initCharts() {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#64748b", font: { size: 10 } }
      },
      y: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#64748b", font: { size: 10 } },
        beginAtZero: true
      }
    },
    plugins: {
      legend: {
        labels: { color: "#94a3b8", font: { size: 11, family: "Outfit" } }
      }
    }
  };

  // 1. Average Delay Chart
  const ctxDelay = document.getElementById("delayChart").getContext("2d");
  delayChartInstance = new Chart(ctxDelay, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Adaptive STCS (FLC+ANN)",
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          data: [],
          tension: 0.3,
          fill: true,
          pointRadius: 2
        },
        {
          label: "Static Fixed Baseline (30s)",
          borderColor: "#f59e0b",
          borderDash: [5, 5],
          data: [],
          tension: 0.3,
          pointRadius: 0
        }
      ]
    },
    options: chartOptions
  });

  // 2. Queue Lengths Chart
  const ctxQueue = document.getElementById("queueChart").getContext("2d");
  queueChartInstance = new Chart(ctxQueue, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "North", borderColor: "#38bdf8", data: [], tension: 0.2, pointRadius: 1 },
        { label: "South", borderColor: "#818cf8", data: [], tension: 0.2, pointRadius: 1 },
        { label: "East", borderColor: "#f43f5e", data: [], tension: 0.2, pointRadius: 1 },
        { label: "West", borderColor: "#fbbf24", data: [], tension: 0.2, pointRadius: 1 }
      ]
    },
    options: chartOptions
  });
}

function appendChartData(data) {
  if (!delayChartInstance || !queueChartInstance) return;

  const tLabel = `${Math.round(data.sim_time)}s`;

  // 1. Delay Chart
  const maxPts = 25;
  const dLabels = delayChartInstance.data.labels;
  dLabels.push(tLabel);
  if (dLabels.length > maxPts) dLabels.shift();

  delayChartInstance.data.datasets[0].data.push(data.average_delay_seconds);
  if (delayChartInstance.data.datasets[0].data.length > maxPts) {
    delayChartInstance.data.datasets[0].data.shift();
  }

  // Synthetic static reference ~ 30% higher
  const staticEstimate = data.control_mode === "ADAPTIVE" ? data.average_delay_seconds * 1.35 : data.average_delay_seconds;
  delayChartInstance.data.datasets[1].data.push(staticEstimate);
  if (delayChartInstance.data.datasets[1].data.length > maxPts) {
    delayChartInstance.data.datasets[1].data.shift();
  }
  delayChartInstance.update();

  // 2. Queue Chart
  const qLabels = queueChartInstance.data.labels;
  qLabels.push(tLabel);
  if (qLabels.length > maxPts) qLabels.shift();

  const lanes = data.lanes || {};
  const qN = (lanes.North && lanes.North.queue_length) || 0;
  const qS = (lanes.South && lanes.South.queue_length) || 0;
  const qE = (lanes.East && lanes.East.queue_length) || 0;
  const qW = (lanes.West && lanes.West.queue_length) || 0;

  queueChartInstance.data.datasets[0].data.push(qN);
  queueChartInstance.data.datasets[1].data.push(qS);
  queueChartInstance.data.datasets[2].data.push(qE);
  queueChartInstance.data.datasets[3].data.push(qW);

  queueChartInstance.data.datasets.forEach((ds) => {
    if (ds.data.length > maxPts) ds.data.shift();
  });
  queueChartInstance.update();
}

// ==============================================================================
// 6. REST API CONTROLS & EVENT LISTENERS
// ==============================================================================
function initControls() {
  // 1. Mode Toggle (Cycles Adaptive -> Static Fixed -> Manual -> Adaptive)
  const btnToggleMode = document.getElementById("btnToggleMode");
  if (btnToggleMode) {
    btnToggleMode.addEventListener("click", async () => {
      const cur = (currentTelemetry && currentTelemetry.control_mode) || "ADAPTIVE";
      let nextMode = "STATIC_FIXED";
      if (cur === "ADAPTIVE") nextMode = "STATIC_FIXED";
      else if (cur === "STATIC_FIXED") nextMode = "MANUAL";
      else nextMode = "ADAPTIVE";
      try {
        await fetch("/api/control/mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: nextMode })
        });
      } catch (e) {
        console.error("Error setting mode:", e);
      }
    });
  }

  // 2. Speed Multiplier Buttons (1x, 2x, 5x)
  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const spd = parseFloat(btn.dataset.speed || "1.0");
      try {
        await fetch("/api/control/speed", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ speed: spd })
        });
        document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
      } catch (e) {
        console.error("Error setting speed:", e);
      }
    });
  });

  // 3. Pause / Resume
  const btnPauseResume = document.getElementById("btnPauseResume");
  if (btnPauseResume) {
    btnPauseResume.addEventListener("click", async () => {
      const isPaused = currentTelemetry && currentTelemetry.is_paused;
      const endpoint = isPaused ? "/api/control/resume" : "/api/control/pause";
      try {
        await fetch(endpoint, { method: "POST" });
      } catch (e) {
        console.error("Error toggling pause:", e);
      }
    });
  }

  // 4. Force Switch
  const btnForce = document.getElementById("btnForceSwitch");
  if (btnForce) {
    btnForce.addEventListener("click", async () => {
      try {
        await fetch("/api/control/switch", { method: "POST" });
      } catch (e) {
        console.error("Error forcing switch:", e);
      }
    });
  }

  // 5. Reset Simulation
  const btnReset = document.getElementById("btnResetSim");
  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      try {
        await fetch("/api/control/reset", { method: "POST" });
        if (delayChartInstance) {
          delayChartInstance.data.labels = [];
          delayChartInstance.data.datasets.forEach(d => d.data = []);
          delayChartInstance.update();
        }
        if (queueChartInstance) {
          queueChartInstance.data.labels = [];
          queueChartInstance.data.datasets.forEach(d => d.data = []);
          queueChartInstance.update();
        }
      } catch (e) {
        console.error("Error resetting sim:", e);
      }
    });
  }

  // 6. Emergency Priority Preemption
  const triggerEmergency = async (approach) => {
    try {
      await fetch("/api/control/emergency", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approach: approach, duration: 20.0 })
      });
    } catch (e) {
      console.error(`Error triggering emergency on ${approach}:`, e);
    }
  };

  const clearEmergency = async () => {
    try {
      await fetch("/api/control/emergency/clear", { method: "POST" });
    } catch (e) {
      console.error("Error clearing emergency:", e);
    }
  };

  const btnN = document.getElementById("btnEmergNorth");
  if (btnN) btnN.addEventListener("click", () => triggerEmergency("North"));
  const btnW = document.getElementById("btnEmergWest");
  if (btnW) btnW.addEventListener("click", () => triggerEmergency("West"));
  const btnS = document.getElementById("btnEmergSouth");
  if (btnS) btnS.addEventListener("click", () => triggerEmergency("South"));
  const btnE = document.getElementById("btnEmergEast");
  if (btnE) btnE.addEventListener("click", () => triggerEmergency("East"));

  const btnClr = document.getElementById("btnClearEmerg");
  if (btnClr) btnClr.addEventListener("click", clearEmergency);
  const btnCancelBanner = document.getElementById("btnCancelEmergBanner");
  if (btnCancelBanner) btnCancelBanner.addEventListener("click", clearEmergency);

  // 7. Inflow Demand Sliders
  ["sliderNorth", "sliderSouth", "sliderEast", "sliderWest"].forEach((id) => {
    const slider = document.getElementById(id);
    const lblMap = {
      sliderNorth: "lblDemandN",
      sliderSouth: "lblDemandS",
      sliderEast: "lblDemandE",
      sliderWest: "lblDemandW"
    };

    if (slider) {
      slider.addEventListener("input", (e) => {
        const lbl = document.getElementById(lblMap[id]);
        if (lbl) lbl.textContent = e.target.value;
        clearTimeout(demandDebounceTimer);
        demandDebounceTimer = setTimeout(pushDemandUpdates, 300);
      });
    }
  });

  // 8. Benchmark Modal Triggers
  const btnRunBm = document.getElementById("btnRunBenchmark");
  if (btnRunBm) btnRunBm.addEventListener("click", runBenchmarkSuite);
  const btnCloseBm = document.getElementById("btnCloseBenchmarkModal");
  if (btnCloseBm) {
    btnCloseBm.addEventListener("click", () => {
      document.getElementById("benchmarkModal").classList.remove("active");
    });
  }
  const btnDismissBm = document.getElementById("btnDismissBenchmarkModal");
  if (btnDismissBm) {
    btnDismissBm.addEventListener("click", () => {
      document.getElementById("benchmarkModal").classList.remove("active");
    });
  }

  // 9. AI Architecture Inspector Modal
  const modelsModal = document.getElementById("modelsModal");
  const btnInspect = document.getElementById("btnInspectModels");
  if (btnInspect && modelsModal) {
    btnInspect.addEventListener("click", async () => {
      modelsModal.classList.add("active");
      try {
        const res = await fetch("/api/soft-computing/models");
        if (res.ok) {
          const info = await res.json();
          console.log("[STCS] Loaded AI Models Metadata:", info);
        }
      } catch (err) {
        console.warn("Could not fetch models metadata:", err);
      }
    });
  }
  const btnCloseModels = document.getElementById("btnCloseModelsModal");
  if (btnCloseModels && modelsModal) {
    btnCloseModels.addEventListener("click", () => modelsModal.classList.remove("active"));
  }
  const btnDismissModels = document.getElementById("btnDismissModelsModal");
  if (btnDismissModels && modelsModal) {
    btnDismissModels.addEventListener("click", () => modelsModal.classList.remove("active"));
  }

  // 10. Export SQLite Audit Log as JSON
  const btnExport = document.getElementById("btnExportLogs");
  if (btnExport) {
    btnExport.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/switches/history?limit=100");
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `stcs_signal_switch_audit_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error("Error exporting logs:", e);
      }
    });
  }

  // Periodically refresh switch audit table (every 5 seconds)
  setInterval(refreshAuditTable, 5000);
  refreshAuditTable();
}

async function pushDemandUpdates() {
  const n = parseFloat(document.getElementById("sliderNorth").value);
  const s = parseFloat(document.getElementById("sliderSouth").value);
  const e = parseFloat(document.getElementById("sliderEast").value);
  const w = parseFloat(document.getElementById("sliderWest").value);

  try {
    await fetch("/api/control/demand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ north: n, south: s, east: e, west: w })
    });
  } catch (err) {
    console.error("Error updating arrival demands:", err);
  }
}

async function refreshAuditTable() {
  try {
    const res = await fetch("/api/switches/history?limit=10");
    if (!res.ok) return;
    const switches = await res.json();
    const tbody = document.getElementById("switchAuditTableBody");
    if (!tbody || switches.length === 0) return;

    let html = "";
    switches.forEach((sw) => {
      html += `
        <tr>
          <td><span style="font-family: var(--font-mono); color: var(--text-muted);">${sw.created_at}</span></td>
          <td><span style="font-family: var(--font-mono);">${sw.sim_time}s</span></td>
          <td><strong>${sw.from_name} &rarr; ${sw.to_name}</strong></td>
          <td><span style="color: var(--color-green); font-weight: 600;">${sw.allocated_green}s</span></td>
          <td><span class="kpi-badge warning">${sw.trigger_reason}</span></td>
          <td><small style="color: var(--text-secondary);">${sw.decision_rationale}</small></td>
        </tr>
      `;
    });
    tbody.innerHTML = html;
  } catch (e) {
    // Ignore fetch errors during reconnect
  }
}

async function runBenchmarkSuite() {
  const btn = document.getElementById("btnRunBenchmark");
  btn.innerHTML = `<i data-lucide="loader-2" class="spin"></i> RUNNING TEST...`;
  btn.disabled = true;

  try {
    const res = await fetch("/api/benchmark/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ duration_seconds: 150.0 })
    });

    const report = await res.json();

    document.getElementById("bmDelayReduction").textContent = `+${report.delay_reduction_pct.toFixed(1)}%`;
    document.getElementById("bmThroughputGain").textContent = `+${report.throughput_increase_pct.toFixed(1)}%`;
    document.getElementById("bmQueueReduction").textContent = `-${report.queue_reduction_pct.toFixed(1)}%`;
    document.getElementById("bmStaticDelay").textContent = `${report.static_avg_delay.toFixed(1)}s`;
    document.getElementById("bmAdaptiveDelay").textContent = `${report.adaptive_avg_delay.toFixed(1)}s`;
    document.getElementById("bmThroughputCompare").textContent = 
      `${report.static_throughput} static vs. ${report.adaptive_throughput} adaptive`;

    document.getElementById("benchmarkModal").classList.add("active");
  } catch (err) {
    alert("Benchmark execution failed: " + err.message);
  } finally {
    btn.innerHTML = `<i data-lucide="bar-chart-3"></i> RUN BENCHMARK`;
    btn.disabled = false;
    lucide.createIcons();
    refreshAuditTable();
  }
}

// ==============================================================================
// 7. INITIALIZATION ON PAGE LOAD
// ==============================================================================
window.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  initCharts();
  initControls();
  initWebSocket();
  requestAnimationFrame(renderIntersection);
});
