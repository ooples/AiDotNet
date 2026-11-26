namespace AiDotNet.Dashboard.Dashboard;

/// <summary>
/// The self-contained HTML page served by <see cref="TrainingMonitorDashboard{T}.StartWebServer"/>: inline CSS and
/// JavaScript, canvas line charts, no external assets.
/// </summary>
internal static class TrainingDashboardWebPage
{
    internal const string Html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AiDotNet training monitor</title>
<style>
  :root { color-scheme: light dark; --fg:#1d2330; --muted:#6b7280; --bg:#f6f7f9; --card:#ffffff; --line:#d9dde3; --accent:#2563eb; --warn:#b45309; --err:#b91c1c; }
  @media (prefers-color-scheme: dark) { :root { --fg:#e5e7eb; --muted:#9ca3af; --bg:#111318; --card:#1a1d24; --line:#2d323c; --accent:#60a5fa; --warn:#f59e0b; --err:#f87171; } }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--fg); background:var(--bg); }
  header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); background:var(--card); }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  select { font:inherit; padding:4px 6px; }
  .pill { padding:2px 8px; border-radius:10px; border:1px solid var(--line); color:var(--muted); }
  .pill.live { color:var(--accent); border-color:var(--accent); }
  main { padding:16px; display:grid; gap:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
  .card h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
  .progress { height:10px; background:var(--line); border-radius:5px; overflow:hidden; }
  .progress > div { height:100%; width:0; background:var(--accent); }
  .stats { display:flex; flex-wrap:wrap; gap:6px 24px; margin-top:8px; }
  .stats b { font-variant-numeric: tabular-nums; }
  table { border-collapse:collapse; width:100%; font-variant-numeric: tabular-nums; }
  td { padding:2px 8px 2px 0; border-bottom:1px solid var(--line); }
  td:last-child { text-align:right; }
  .charts { display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:16px; }
  .chart canvas { width:100%; height:180px; display:block; }
  .chart .title { display:flex; justify-content:space-between; margin-bottom:4px; }
  .chart .title span:last-child { color:var(--muted); font-variant-numeric: tabular-nums; }
  .two { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; }
  ul { list-style:none; margin:0; padding:0; max-height:220px; overflow:auto; font-family:ui-monospace, Consolas, monospace; font-size:12px; }
  li { padding:2px 0; border-bottom:1px solid var(--line); white-space:pre-wrap; }
  .Warning { color:var(--warn); } .Error { color:var(--err); } .alert { color:var(--err); }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Training monitor</h1>
  <label>Session <select id="session"></select></label>
  <span id="state" class="pill">connecting</span>
  <span id="updated" class="muted"></span>
</header>
<main>
  <section class="card">
    <h2>Progress</h2>
    <div class="progress"><div id="bar"></div></div>
    <div class="stats">
      <span>Step <b id="step">-</b></span>
      <span>Epoch <b id="epoch">-</b></span>
      <span>Elapsed <b id="elapsed">-</b></span>
      <span>Remaining <b id="eta">-</b></span>
      <span>Resources <b id="resource">-</b></span>
    </div>
  </section>
  <section class="two">
    <div class="card"><h2>Current values</h2><table id="current"></table></div>
    <div class="card"><h2>Alerts and issues</h2><ul id="alerts"></ul></div>
  </section>
  <section class="charts" id="charts"></section>
  <section class="card"><h2>Messages</h2><ul id="messages"></ul></section>
</main>
<script>
(function () {
  "use strict";
  var REFRESH_MS = 2000;
  // Prototype-free maps: a metric may be named "__proto__" or "constructor".
  var MAX_POINTS = 4096;
  var state = { sessionId: null, seq: null, series: Object.create(null), charts: Object.create(null) };
  var $ = function (id) { return document.getElementById(id); };

  function num(v) { return v === null || v === undefined ? NaN : Number(v); }
  function fmt(v) {
    v = num(v);
    if (!isFinite(v)) return String(v);
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
    // Trailing zeros are trimmed from a fraction only: 10000 stays 10000, 1.2500 becomes 1.25, 1.0000e+5 1e+5.
    var parts = v.toPrecision(5).split("e");
    if (parts[0].indexOf(".") >= 0) parts[0] = parts[0].replace(/0+$/, "").replace(/\.$/, "");
    return parts.join("e");
  }
  function duration(s) {
    s = Math.max(0, Math.floor(num(s)));
    if (!isFinite(s)) return "-";
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    return (h ? h + "h " : "") + (h || m ? m + "m " : "") + r + "s";
  }
  function text(el, value) { el.textContent = value; }
  function getJson(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.json();
    });
  }

  function loadSessions() {
    return getJson("api/sessions").then(function (sessions) {
      var select = $("session");
      var previous = state.sessionId;
      var ids = sessions.map(function (s) { return s.id; });
      if (select.options.length !== sessions.length || Array.prototype.some.call(select.options, function (o, i) { return o.value !== ids[i]; })) {
        select.innerHTML = "";
        sessions.forEach(function (s) {
          var o = document.createElement("option");
          o.value = s.id;
          o.textContent = s.name + (s.active ? " (running)" : "") + " - " + new Date(s.startTime).toLocaleString();
          select.appendChild(o);
        });
      }
      if (!previous || ids.indexOf(previous) < 0) {
        var running = sessions.filter(function (s) { return s.active; });
        var pick = running.length ? running[running.length - 1] : sessions[sessions.length - 1];
        selectSession(pick ? pick.id : null);
      }
      if (state.sessionId) select.value = state.sessionId;
      return sessions;
    });
  }

  function selectSession(id) {
    state.sessionId = id;
    state.seq = null;
    state.series = Object.create(null);
    state.charts = Object.create(null);
    $("charts").innerHTML = "";
  }

  function chartFor(name) {
    var chart = state.charts[name];
    if (chart) return chart;
    var card = document.createElement("div");
    card.className = "card chart";
    var title = document.createElement("div");
    title.className = "title";
    var label = document.createElement("span");
    label.textContent = name;
    var last = document.createElement("span");
    title.appendChild(label);
    title.appendChild(last);
    var canvas = document.createElement("canvas");
    card.appendChild(title);
    card.appendChild(canvas);
    var container = $("charts");
    var names = Object.keys(state.charts).concat([name]).sort();
    var index = names.indexOf(name);
    var next = names[index + 1] ? state.charts[names[index + 1]] : null;
    container.insertBefore(card, next ? next.card : null);
    chart = { card: card, canvas: canvas, last: last };
    state.charts[name] = chart;
    return chart;
  }

  function draw(name) {
    var points = state.series[name] || [];
    var chart = chartFor(name);
    var canvas = chart.canvas;
    var ratio = window.devicePixelRatio || 1;
    var width = canvas.clientWidth || 340, height = canvas.clientHeight || 180;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    var style = getComputedStyle(document.documentElement);
    var fg = style.getPropertyValue("--muted").trim() || "#888";
    var lineColor = style.getPropertyValue("--line").trim() || "#ccc";
    var accent = style.getPropertyValue("--accent").trim() || "#2563eb";
    var finite = points.filter(function (p) { return isFinite(p[1]); });
    chart.last.textContent = points.length ? "step " + points[points.length - 1][0] + ": " + fmt(points[points.length - 1][1]) : "";
    if (!finite.length) return;
    var minX = finite[0][0], maxX = finite[0][0], minY = finite[0][1], maxY = finite[0][1];
    finite.forEach(function (p) {
      if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
    });
    if (maxX === minX) { maxX = minX + 1; }
    if (maxY === minY) { var pad = Math.abs(maxY) * 0.05 || 1; maxY += pad; minY -= pad; }
    var left = 58, right = 8, top = 6, bottom = 20;
    var w = width - left - right, h = height - top - bottom;
    var sx = function (x) { return left + (x - minX) / (maxX - minX) * w; };
    var sy = function (y) { return top + (1 - (y - minY) / (maxY - minY)) * h; };
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = fg;
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      var yv = minY + (maxY - minY) * i / 4, yp = sy(yv);
      ctx.beginPath(); ctx.moveTo(left, yp); ctx.lineTo(left + w, yp); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(fmt(yv), left - 6, yp);
    }
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(String(minX), left, top + h + 4);
    ctx.textAlign = "right";
    ctx.fillText(String(maxX), left + w, top + h + 4);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    finite.forEach(function (p, k) {
      var x = sx(p[0]), y = sy(p[1]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (finite.length === 1) {
      ctx.fillStyle = accent;
      ctx.beginPath(); ctx.arc(sx(finite[0][0]), sy(finite[0][1]), 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  function renderList(el, items) {
    el.innerHTML = "";
    items.forEach(function (item) {
      var li = document.createElement("li");
      li.textContent = item.text;
      if (item.cls) li.className = item.cls;
      el.appendChild(li);
    });
    el.scrollTop = el.scrollHeight;
  }

  function render(data) {
    var s = data.session;
    var total = num(s.totalSteps), step = num(s.currentStep);
    $("bar").style.width = total > 0 ? Math.min(100, step / total * 100).toFixed(1) + "%" : "0%";
    text($("step"), total > 0 ? step + " / " + total : String(step));
    text($("epoch"), s.totalEpochs > 0 ? s.currentEpoch + " / " + s.totalEpochs : String(s.currentEpoch));
    text($("elapsed"), duration(s.elapsedSeconds));
    text($("eta"), s.active && total > 0 && step > 0 ? duration(data.speed.estimatedRemainingSeconds) : "-");
    var r = data.resource;
    text($("resource"), r ? "CPU " + fmt(r.cpuPercent) + "%, RAM " + fmt(r.memoryMB) + " MB" +
      (r.gpuPercent !== null ? ", GPU " + fmt(r.gpuPercent) + "%" : "") +
      (r.gpuMemoryMB !== null ? ", GPU mem " + fmt(r.gpuMemoryMB) + " MB" : "") : "-");
    var stateEl = $("state");
    stateEl.textContent = s.active ? "running" : "finished";
    stateEl.className = "pill" + (s.active ? " live" : "");

    var table = $("current");
    table.innerHTML = "";
    Object.keys(data.current).sort().forEach(function (k) {
      var tr = document.createElement("tr");
      var a = document.createElement("td"), b = document.createElement("td");
      a.textContent = k; b.textContent = fmt(data.current[k]);
      tr.appendChild(a); tr.appendChild(b); table.appendChild(tr);
    });

    renderList($("alerts"), data.alerts.map(function (a) { return { text: a, cls: "alert" }; })
      .concat(data.issues.map(function (i) { return { text: "issue: " + i, cls: "Warning" }; })));
    renderList($("messages"), data.messages.map(function (m) {
      return { text: new Date(m.time).toLocaleTimeString() + " [" + m.level + "] " + m.message, cls: m.level };
    }));

    Object.keys(data.metrics).forEach(function (name) {
      var series = state.series[name] || (state.series[name] = []);
      data.metrics[name].forEach(function (p) { series.push([num(p[0]), num(p[1])]); });
      if (series.length > MAX_POINTS) series.splice(0, series.length - MAX_POINTS);
    });
    Object.keys(state.series).forEach(draw);
    state.seq = data.seq;
  }

  function refresh() {
    loadSessions().then(function () {
      if (!state.sessionId) { text($("state"), "no sessions"); return null; }
      var id = state.sessionId;
      var url = "api/metrics?session=" + encodeURIComponent(id) + (state.seq !== null ? "&seq=" + state.seq : "");
      return getJson(url).then(function (data) { if (id === state.sessionId) render(data); });
    }).then(function () {
      text($("updated"), "updated " + new Date().toLocaleTimeString());
    }).catch(function (e) {
      var el = $("state"); el.textContent = "disconnected: " + e.message; el.className = "pill";
    }).then(function () { setTimeout(refresh, REFRESH_MS); });
  }

  $("session").addEventListener("change", function (e) { selectSession(e.target.value); });
  window.addEventListener("resize", function () { Object.keys(state.series).forEach(draw); });
  refresh();
})();
</script>
</body>
</html>
""";
}
