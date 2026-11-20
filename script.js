/**
 * Smart Traffic AI - Frontend Application & Canvas Visualizer
 * Communicates via fetch() REST APIs with Python Flask backend,
 * renders 60 FPS 4-way road intersection physics, and updates Chart.js analytics.
 */

// Global State
let simulationState = null;
let pollInterval = null;
let waitTimeChart = null;
let queuesChart = null;

// Canvas Configuration
const canvas = document.getElementById('intersectionCanvas');
const ctx = canvas.getContext('2d');

// Center coordinates of the 4-way intersection
const cx = canvas.width / 2;  // 400
const cy = canvas.height / 2; // 250
const roadWidth = 90;         // 45px per lane
const halfRoad = roadWidth / 2;

// Vehicle Color Palette
const VEHICLE_COLORS = {
  car: ['#38bdf8', '#818cf8', '#94a3b8', '#34d399', '#f472b6', '#cbd5e1'],
  bus: '#fbbf24',
  truck: '#fb923c'
};

// ==========================================================
// 1. Chart.js Initialization
// ==========================================================
function initCharts() {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'sans-serif', size: 10 } }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(51, 65, 85, 0.25)' },
        ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 10 }
      },
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(51, 65, 85, 0.25)' },
        ticks: { color: '#64748b', font: { size: 9 } }
      }
    }
  };

  // 1. Waiting Time Chart
  const ctxWait = document.getElementById('chartWaitTime').getContext('2d');
  waitTimeChart = new Chart(ctxWait, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Avg Delay (s/veh)',
        data: [],
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.15)',
        borderWidth: 2,
        fill: true,
        tension: 0.35,
        pointRadius: 0
      }]
    },
    options: chartOptions
  });

  // 2. Queue Lengths Chart
  const ctxQueues = document.getElementById('chartQueues').getContext('2d');
  queuesChart = new Chart(ctxQueues, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Vehicles Queued',
        data: [],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.12)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 0
      }]
    },
    options: chartOptions
  });
}

// ==========================================================
// 2. Canvas 4-Way Road Intersection Rendering Engine
// ==========================================================
function drawIntersection(state) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background Grass / Terrain
  ctx.fillStyle = '#0a0f1d';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Asphalt Roads
  ctx.fillStyle = '#1e293b';

  // North-South Road
  ctx.fillRect(cx - halfRoad, 0, roadWidth, canvas.height);
  // East-West Road
  ctx.fillRect(0, cy - halfRoad, canvas.width, roadWidth);

  // Intersection Box (Center)
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(cx - halfRoad, cy - halfRoad, roadWidth, roadWidth);

  // 2. Road Borders (Solid White Lines)
  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 2.5;

  // North-West Corner
  ctx.beginPath();
  ctx.moveTo(0, cy - halfRoad); ctx.lineTo(cx - halfRoad, cy - halfRoad);
  ctx.lineTo(cx - halfRoad, 0);
  ctx.stroke();

  // North-East Corner
  ctx.beginPath();
  ctx.moveTo(cx + halfRoad, 0); ctx.lineTo(cx + halfRoad, cy - halfRoad);
  ctx.lineTo(canvas.width, cy - halfRoad);
  ctx.stroke();

  // South-West Corner
  ctx.beginPath();
  ctx.moveTo(0, cy + halfRoad); ctx.lineTo(cx - halfRoad, cy + halfRoad);
  ctx.lineTo(cx - halfRoad, canvas.height);
  ctx.stroke();

  // South-East Corner
  ctx.beginPath();
  ctx.moveTo(cx + halfRoad, canvas.height); ctx.lineTo(cx + halfRoad, cy + halfRoad);
  ctx.lineTo(canvas.width, cy + halfRoad);
  ctx.stroke();

  // 3. Lane Dividers (Dashed Yellow Lines)
  ctx.strokeStyle = '#fbbf24';
  ctx.lineWidth = 2;
  ctx.setLineDash([12, 10]);

  // North Lane Divider
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, cy - halfRoad - 20); ctx.stroke();
  // South Lane Divider
  ctx.beginPath(); ctx.moveTo(cx, cy + halfRoad + 20); ctx.lineTo(cx, canvas.height); ctx.stroke();
  // West Lane Divider
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(cx - halfRoad - 20, cy); ctx.stroke();
  // East Lane Divider
  ctx.beginPath(); ctx.moveTo(cx + halfRoad + 20, cy); ctx.lineTo(canvas.width, cy); ctx.stroke();

  ctx.setLineDash([]); // Reset line dash

  // 4. Zebra Crossings (Pedestrian Crosswalks)
  drawCrosswalk(cx - halfRoad, cy - halfRoad - 22, roadWidth, 16, true);
  drawCrosswalk(cx - halfRoad, cy + halfRoad + 6, roadWidth, 16, true);
  drawCrosswalk(cx - halfRoad - 22, cy - halfRoad, 16, roadWidth, false);
  drawCrosswalk(cx + halfRoad + 6, cy - halfRoad, 16, roadWidth, false);

  // 5. White Stop Lines
  ctx.strokeStyle = '#f8fafc';
  ctx.lineWidth = 4;
  // North Stop Line (for Southbound incoming lane: right-hand traffic => x from cx to cx-halfRoad)
  ctx.beginPath(); ctx.moveTo(cx - halfRoad, cy - halfRoad - 24); ctx.lineTo(cx, cy - halfRoad - 24); ctx.stroke();
  // South Stop Line (for Northbound incoming lane => x from cx to cx+halfRoad)
  ctx.beginPath(); ctx.moveTo(cx, cy + halfRoad + 24); ctx.lineTo(cx + halfRoad, cy + halfRoad + 24); ctx.stroke();
  // East Stop Line (for Westbound incoming lane => y from cy-halfRoad to cy)
  ctx.beginPath(); ctx.moveTo(cx + halfRoad + 24, cy - halfRoad); ctx.lineTo(cx + halfRoad + 24, cy); ctx.stroke();
  // West Stop Line (for Eastbound incoming lane => y from cy to cy+halfRoad)
  ctx.beginPath(); ctx.moveTo(cx - halfRoad - 24, cy); ctx.lineTo(cx - halfRoad - 24, cy + halfRoad); ctx.stroke();

  // 6. Draw 4 Traffic Signal Heads
  if (state && state.signals) {
    drawSignalHead(cx - halfRoad - 18, cy - halfRoad - 40, state.signals.N, 'vertical');
    drawSignalHead(cx + halfRoad + 8, cy + halfRoad + 16, state.signals.S, 'vertical');
    drawSignalHead(cx + halfRoad + 16, cy - halfRoad - 18, state.signals.E, 'horizontal');
    drawSignalHead(cx - halfRoad - 40, cy + halfRoad + 8, state.signals.W, 'horizontal');
  }

  // 7. Draw Moving Vehicles
  if (state && state.vehicles) {
    state.vehicles.forEach(v => drawVehicle(v));
  }
}

function drawCrosswalk(x, y, w, h, isHorizontal) {
  ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
  const stripes = 6;
  if (isHorizontal) {
    const sw = w / stripes;
    for (let i = 0; i < stripes; i += 2) {
      ctx.fillRect(x + i * sw, y, sw, h);
    }
  } else {
    const sh = h / stripes;
    for (let i = 0; i < stripes; i += 2) {
      ctx.fillRect(x, y + i * sh, w, sh);
    }
  }
}

function drawSignalHead(x, y, colorState, orientation) {
  const redActive = colorState === 'red';
  const yellowActive = colorState === 'yellow';
  const greenActive = colorState === 'green';

  ctx.fillStyle = '#0f172a';
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.5;

  if (orientation === 'vertical') {
    ctx.fillRect(x, y, 14, 34);
    ctx.strokeRect(x, y, 14, 34);

    // Red Lamp
    drawLamp(x + 7, y + 7, '#ef4444', redActive);
    // Yellow Lamp
    drawLamp(x + 7, y + 17, '#f59e0b', yellowActive);
    // Green Lamp
    drawLamp(x + 7, y + 27, '#10b981', greenActive);
  } else {
    ctx.fillRect(x, y, 34, 14);
    ctx.strokeRect(x, y, 34, 14);

    // Red Lamp
    drawLamp(x + 7, y + 7, '#ef4444', redActive);
    // Yellow Lamp
    drawLamp(x + 17, y + 7, '#f59e0b', yellowActive);
    // Green Lamp
    drawLamp(x + 27, y + 7, '#10b981', greenActive);
  }
}

function drawLamp(cx, cy, color, isActive) {
  ctx.beginPath();
  ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
  if (isActive) {
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  } else {
    ctx.fillStyle = '#1e293b';
    ctx.fill();
  }
}

function drawVehicle(v) {
  // Convert backend road position to Canvas X/Y coordinates
  let vx, vy, angle;
  const laneOffset = 22; // Offset from road center to lane center

  if (v.direction === 'N') {
    // Southbound: moves from top (y=0) towards bottom (y=500)
    vx = cx - laneOffset;
    vy = v.position;
    angle = Math.PI / 2; // Facing down
  } else if (v.direction === 'S') {
    // Northbound: moves from bottom (y=500) towards top (y=0)
    vx = cx + laneOffset;
    vy = canvas.height - v.position;
    angle = -Math.PI / 2; // Facing up
  } else if (v.direction === 'E') {
    // Westbound: moves from right (x=800) towards left (x=0)
    vx = canvas.width - v.position;
    vy = cy - laneOffset;
    angle = Math.PI; // Facing left
  } else if (v.direction === 'W') {
    // Eastbound: moves from left (x=0) towards right (x=800)
    vx = v.position;
    vy = cy + laneOffset;
    angle = 0; // Facing right
  }

  // Choose vehicle dimensions and color
  const width = v.type === 'bus' ? 36 : (v.type === 'truck' ? 30 : 22);
  const height = 13;
  const color = v.type === 'bus' ? VEHICLE_COLORS.bus : (
    v.type === 'truck' ? VEHICLE_COLORS.truck : VEHICLE_COLORS.car[v.id % VEHICLE_COLORS.car.length]
  );

  ctx.save();
  ctx.translate(vx, vy);
  ctx.rotate(angle);

  // Vehicle Body
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(-width / 2, -height / 2, width, height, 3);
  ctx.fill();

  // Windshield & Roof
  ctx.fillStyle = 'rgba(15, 23, 42, 0.6)';
  ctx.fillRect(-width / 6, -height / 2 + 2, width / 2.5, height - 4);

  // Headlights (Facing Forward)
  ctx.fillStyle = '#fef08a';
  ctx.fillRect(width / 2 - 2, -height / 2 + 1, 2, 3);
  ctx.fillRect(width / 2 - 2, height / 2 - 4, 2, 3);

  // Brake Lights (Red when stopped)
  ctx.fillStyle = v.is_stopped ? '#ef4444' : '#7f1d1d';
  if (v.is_stopped) {
    ctx.shadowColor = '#ef4444';
    ctx.shadowBlur = 8;
  }
  ctx.fillRect(-width / 2, -height / 2 + 1, 2, 3);
  ctx.fillRect(-width / 2, height / 2 - 4, 2, 3);
  ctx.shadowBlur = 0;

  ctx.restore();
}

// ==========================================================
// 3. Telemetry Fetching & UI Synchronization
// ==========================================================
async function fetchState() {
  try {
    const res = await fetch('/api/state');
    if (!res.ok) return;
    const data = await res.json();
    simulationState = data;

    // Update KPI Dashboard Values
    document.getElementById('valTotalVehicles').innerText = data.total_vehicles;
    document.getElementById('valVehiclesWaiting').innerText = data.vehicles_waiting;
    document.getElementById('valVehiclesPassed').innerText = data.vehicles_passed;
    document.getElementById('valAvgWait').innerText = `${data.average_waiting_time}s`;
    document.getElementById('valSimTime').innerText = `${data.simulation_time}s`;

    // Active Phase Text
    const phaseElem = document.getElementById('valActivePhase');
    phaseElem.innerText = data.phase_name.replace('_', ' ');
    if (data.phase_name.includes('GREEN')) {
      phaseElem.style.color = '#10b981';
    } else if (data.phase_name.includes('YELLOW')) {
      phaseElem.style.color = '#f59e0b';
    } else {
      phaseElem.style.color = '#ef4444';
    }

    // Traffic Signal Indicators on Top of Canvas
    setSignalDot('lightN', data.signals.N);
    setSignalDot('lightS', data.signals.S);
    setSignalDot('lightE', data.signals.E);
    setSignalDot('lightW', data.signals.W);

    // AI Decision Display
    document.getElementById('aiDecisionText').innerText = data.ai_decision;

    // Render 4-Way Canvas
    drawIntersection(data);

    // Update Chart.js Series
    if (data.history && data.history.timestamps.length > 0) {
      waitTimeChart.data.labels = data.history.timestamps.map(t => `${t}s`);
      waitTimeChart.data.datasets[0].data = data.history.wait_times;
      waitTimeChart.update('none');

      queuesChart.data.labels = data.history.timestamps.map(t => `${t}s`);
      queuesChart.data.datasets[0].data = data.history.queue_lengths;
      queuesChart.update('none');
    }

    // Update Benchmark Comparison Table
    if (data.baseline_stats) {
      const ai = data.baseline_stats.ai;
      const fixed = data.baseline_stats.fixed;

      if (fixed.samples > 0) {
        document.getElementById('rowFixedWait').innerText = `${fixed.avg_wait}s`;
        document.getElementById('rowFixedQueue').innerText = `${fixed.avg_queue} veh`;
        document.getElementById('rowFixedThroughput').innerText = `${fixed.total_passed} veh`;
      }
      if (ai.samples > 0) {
        document.getElementById('rowAiWait').innerText = `${ai.avg_wait}s`;
        document.getElementById('rowAiQueue').innerText = `${ai.avg_queue} veh`;
        document.getElementById('rowAiThroughput').innerText = `${ai.total_passed} veh`;
      }

      // Calculate empirical improvement
      if (fixed.avg_wait > 0 && ai.avg_wait > 0) {
        const diff = ((fixed.avg_wait - ai.avg_wait) / fixed.avg_wait) * 100;
        document.getElementById('rowGainWait').innerText = `${diff >= 0 ? '-' : '+'}${Math.abs(diff).toFixed(1)}% (${diff >= 0 ? 'Faster' : 'Slower'})`;
      }
      if (fixed.avg_queue > 0 && ai.avg_queue > 0) {
        const diffQ = ((fixed.avg_queue - ai.avg_queue) / fixed.avg_queue) * 100;
        document.getElementById('rowGainQueue').innerText = `${diffQ >= 0 ? '-' : '+'}${Math.abs(diffQ).toFixed(1)}%`;
      }
    }
  } catch (err) {
    console.error('Fetch state error:', err);
  }
}

function setSignalDot(elementId, color) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.className = `signal-light light-${color}`;
}

// ==========================================================
// 4. Button & Control Event Listeners
// ==========================================================
function setupControls() {
  // Start Button
  document.getElementById('btnStart').addEventListener('click', async () => {
    await fetch('/api/start', { method: 'POST' });
  });

  // Pause Button
  document.getElementById('btnPause').addEventListener('click', async () => {
    await fetch('/api/pause', { method: 'POST' });
  });

  // Reset Button
  document.getElementById('btnReset').addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    waitTimeChart.data.labels = [];
    waitTimeChart.data.datasets[0].data = [];
    waitTimeChart.update();
    queuesChart.data.labels = [];
    queuesChart.data.datasets[0].data = [];
    queuesChart.update();
  });

  // Policy Mode Buttons
  const btnAI = document.getElementById('btnModeAI');
  const btnFixed = document.getElementById('btnModeFixed');
  const statusPill = document.getElementById('aiStatusPill');
  const statusText = document.getElementById('aiStatusText');

  btnAI.addEventListener('click', async () => {
    await fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'ai' })
    });
    btnAI.classList.add('btn-active');
    btnFixed.classList.remove('btn-active');
    statusPill.style.background = 'rgba(16, 185, 129, 0.12)';
    statusPill.style.borderColor = 'rgba(16, 185, 129, 0.35)';
    statusPill.style.color = '#10b981';
    statusText.innerText = 'AI Controller Active (Q-Learning)';
  });

  btnFixed.addEventListener('click', async () => {
    await fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'fixed' })
    });
    btnFixed.classList.add('btn-active');
    btnAI.classList.remove('btn-active');
    statusPill.style.background = 'rgba(245, 158, 11, 0.12)';
    statusPill.style.borderColor = 'rgba(245, 158, 11, 0.35)';
    statusPill.style.color = '#f59e0b';
    statusText.innerText = 'Fixed-Timer Mode Active (25s)';
  });

  // Density Buttons
  document.querySelectorAll('.btn-density').forEach(btn => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.btn-density').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const density = btn.getAttribute('data-density');
      await fetch('/api/density', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ density })
      });
    });
  });

  // Train Agent Button
  const btnTrain = document.getElementById('btnTrainAgent');
  const trainFeedback = document.getElementById('trainFeedback');

  btnTrain.addEventListener('click', async () => {
    btnTrain.disabled = true;
    btnTrain.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Training Q-Learning Agent...`;
    trainFeedback.innerText = 'Simulating 30 training episodes with Bellman updates...';

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episodes: 30 })
      });
      const data = await res.json();
      trainFeedback.innerText = `Training complete in ${data.elapsed_time}s! Epsilon: ${data.final_epsilon}. Q-Table updated.`;
      trainFeedback.style.color = '#10b981';
    } catch (err) {
      trainFeedback.innerText = 'Error retraining agent.';
      trainFeedback.style.color = '#ef4444';
    } finally {
      btnTrain.disabled = false;
      btnTrain.innerHTML = `<i class="fas fa-dumbbell"></i> Retrain Q-Learning Agent (30 Ep)`;
    }
  });
}

// ==========================================================
// 5. App Bootstrap
// ==========================================================
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  setupControls();
  drawIntersection(null);

  // Poll backend state every 60ms (~16 requests/sec for smooth animation)
  pollInterval = setInterval(fetchState, 60);
});
