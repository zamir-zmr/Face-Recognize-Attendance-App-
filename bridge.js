/* ===== PYTHON/OPENCV BACKEND BRIDGE (falls back to built-in face-api.js when backend is unreachable) ===== */
(() => {
  'use strict';
  const CFG = {
    url: (window.FACE_BACKEND_URL || localStorage.getItem('face_backend_url') || 'http://localhost:8000').replace(/\/$/, ''),
    key: window.FACE_API_KEY || localStorage.getItem('face_api_key') || '',
    width: 480, quality: 0.6, gapMs: 0,
    fingerWidth: 320, fingerQuality: 0.5,     /* smaller/lighter JPEGs in the finger step = faster round-trips */
    pruneSync: window.FACE_PRUNE_SYNC === true          /* opt-in: delete server embeddings for employees removed locally */
  };
  const backend = { ok: false };

  async function api(path, opts = {}, timeout = 4000) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeout);
    try {
      const r = await fetch(CFG.url + path, {
        method: opts.method || 'GET',
        headers: { 'Content-Type': 'application/json', ...(CFG.key ? { 'X-API-Key': CFG.key } : {}) },
        body: opts.body, signal: ctl.signal
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || ('HTTP ' + r.status));
      return j;
    } finally { clearTimeout(t); }
  }

  const statusEl = () => document.getElementById('modelStatus');
  let syncing = false; const syncFailed = new Set();

  /* ---------- finger-challenge health notice ---------- */
  let handsIssue = null, noticeKey = "";
  function paintBadge(el) {
    el.textContent = T("engine.ready") + (handsIssue ? T("engine.fingerOff") : "");
    el.style.color = handsIssue ? "#f59e0b" : "var(--ok)";
    el.style.cursor = handsIssue ? "pointer" : "";
  }
  function fingerHint(h) {
    const e = (h.hands_error || "").toLowerCase();
    if (h.hands_api === "old-engine") return T("engine.hints.oldEngine");
    if (e.includes("hand_landmarker")) return T("engine.hints.modelMissing");
    if (e.includes("libgles") || e.includes("libegl") || e.includes("libgl.")) return T("engine.hints.glLibrary");
    if (e.includes("solutions")) return T("engine.hints.tooNew");
    if (e.includes("import") || e.includes("no module") || e.includes("not installed")) return T("engine.hints.notInstalled", {python: h.python || "?"});
    return T("engine.hints.generic");
  }
  function closeFingerNotice() { const n = document.getElementById("fingerNotice"); if (n) n.remove(); }
  function showFingerNotice(h) {
    closeFingerNotice();
    const box = document.createElement("div");
    box.id = "fingerNotice"; box.setAttribute("role", "alert");
    box.style.cssText = "position:fixed;left:12px;right:12px;bottom:92px;z-index:99999;max-width:520px;margin:0 auto;padding:14px 16px;border-radius:14px;" +
      "background:#0f172a;color:#f8fafc;border:1px solid rgba(255,255,255,.12);border-left:4px solid #f59e0b;box-shadow:0 12px 32px rgba(0,0,0,.45);font:14px/1.45 system-ui,sans-serif";
    const t = document.createElement("div"); t.style.cssText = "font-weight:700;margin-bottom:6px"; t.textContent = T("engine.notice.title");
    const why = document.createElement("div"); why.style.cssText = "color:#cbd5e1;word-break:break-word"; why.textContent = T("engine.notice.reason", {reason: h.hands_error || T("engine.notice.reasonDefault")});
    const fix = document.createElement("div"); fix.style.cssText = "margin-top:6px;color:#fbbf24"; fix.textContent = T("engine.notice.fix", {fix: fingerHint(h)});
    const row = document.createElement("div"); row.style.cssText = "display:flex;gap:8px;margin-top:10px;justify-content:flex-end";
    const mk = (label, fn, primary) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn;
      b.style.cssText = "padding:8px 14px;border-radius:10px;border:1px solid rgba(255,255,255,.15);font-weight:600;cursor:pointer;" + (primary ? "background:#0ea5e9;color:#fff;border-color:transparent" : "background:transparent;color:#f8fafc"); return b; };
    row.append(mk(T("engine.notice.recheck"), () => { closeFingerNotice(); noticeKey = ""; ping(); }, true), mk(T("engine.notice.close"), closeFingerNotice, false));
    box.append(t, why, fix, row);
    document.body.appendChild(box);
  }
  { const el0 = document.getElementById("modelStatus"); if (el0) el0.addEventListener("click", () => { if (handsIssue) showFingerNotice(handsIssue); }); }

  let pingFails = 0;
  async function ping() {
    try {
      const h = await api('/api/health', {}, 6000);
      pingFails = 0;
      backend.ok = !!h.ok;
      const el = statusEl();
      handsIssue = (backend.ok && h.hands === false) ? h : null;
      if (backend.ok && el && !syncing) paintBadge(el);
      if (handsIssue) {
        const key = (h.hands_api || "") + "|" + (h.hands_error || "");
        if (key !== noticeKey) { noticeKey = key; showFingerNotice(h); }
      } else { noticeKey = ""; closeFingerNotice(); }
      /* server lost its embeddings (restart / ephemeral disk) or never had them -> rebuild from stored photos */
      if (backend.ok && h.enrolled < employees.filter(e => e.photo && !syncFailed.has(e.id)).length) syncMissing();
    } catch (e) { if (++pingFails >= 2) backend.ok = false; }   /* one slow ping must not flip the scanner offline */
  }

  /* Legacy face-api descriptors are not portable to the server's SFace model, so re-derive from the saved photos. */
  function photoToFrames(dataUrl) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => {
        const w = im.naturalWidth, h = im.naturalHeight, out = [];
        /* snapshotToDataUrl() squashed the video into 160x160 -> restore 4:3 and 16:9 geometry and upscale */
        const sizes = (w === h && w <= 200) ? [[320, 240], [356, 200], [320, 320]] : [[Math.min(w, 640), Math.round(h * Math.min(w, 640) / w)]];
        for (const [tw, th] of sizes) {
          const c = document.createElement('canvas'); c.width = tw; c.height = th;
          const x = c.getContext('2d'); x.imageSmoothingQuality = 'high'; x.drawImage(im, 0, 0, tw, th);
          out.push(c.toDataURL('image/jpeg', 0.92));
        }
        resolve(out);
      };
      im.onerror = () => reject(new Error('bad photo'));
      if (!/^data:/.test(dataUrl)) im.crossOrigin = 'anonymous';
      im.src = dataUrl;
    });
  }

  async function syncMissing() {
    if (!backend.ok || syncing || window.__enrolling) return;   /* never compete with a live enrollment for the server CPU */
    syncing = true;
    const el = statusEl();
    try {
      const have = new Set((await api('/api/enrolled')).ids);
      const todo = employees.filter(e => e.photo && !have.has(e.id) && !syncFailed.has(e.id));
      let ok = 0; const why = [];
      for (let i = 0; i < todo.length; i++) {
        if (el) { el.textContent = T('sync.progress', {current: i + 1, total: todo.length}); el.style.color = ''; }
        try {
          const images = await photoToFrames(todo[i].photo);
          await api('/api/enroll', { method: 'POST', body: JSON.stringify({ employee_id: todo[i].id, images, force: true, relaxed: true }) }, 60000);
          ok++;
        } catch (err) { syncFailed.add(todo[i].id); why.push((todo[i].name || todo[i].id) + ': ' + (err.name === 'AbortError' ? T('sync.serverTimeout') : err.message)); console.warn('face sync failed', todo[i].id, err.message); }
      }
      if (todo.length) {
        const bad = todo.length - ok;
        toast((ok ? T('sync.synced', {count: ok}) : '') + (ok && bad ? ' · ' : '') + (bad ? T('sync.needReenroll', {count: bad, reason: why[0]}) : ''), bad ? 'err' : 'ok');
      }
    } catch (e) { console.warn('sync error', e.message); }
    finally {
      syncing = false;
      if (el && backend.ok) paintBadge(el);
    }
  }
  window.faceBackendDelete = id => backend.ok ? api('/api/enroll/' + encodeURIComponent(id), { method: 'DELETE' }).catch(() => {}) : null;
  window.faceBackendImport = recs => backend.ok ? api('/api/employees/import', { method: 'POST', body: JSON.stringify({ employees: recs }) }, 90000).catch(() => null) : null;
  ping(); setInterval(ping, 10000);

  const cap = document.createElement('canvas'), cctx = cap.getContext('2d');
  function grab(src, w = CFG.width, q = CFG.quality) {
    const sw = src.videoWidth || src.naturalWidth || src.width, sh = src.videoHeight || src.naturalHeight || src.height;
    const s = Math.min(1, w / sw);
    cap.width = Math.round(sw * s); cap.height = Math.round(sh * s);
    cctx.drawImage(src, 0, 0, cap.width, cap.height);
    return cap.toDataURL('image/jpeg', q);
  }

  /* ---------- LIVE SCAN (server: low-light enhance -> detect -> identify (Step 1/2) -> random finger challenge (Step 2/2)) ---------- */
  const _origRunFaceScan = window.runFaceScan;

  /* ---------- Two-step HUD (Step 1/2 face, Step 2/2 fingers) ---------- */
  const DOT_GREEN = '#0FFF50', DOT_BLUE = '#00BFFF', DOT_RED = '#FF0000';
  const HUD_BLUE = '#00E5FF', HUD_GREEN = '#10B981', HUD_RED = '#f43f5e', FINGER_COLOR = '#10B981';
  const camBox = () => document.getElementById('camBox');
  let hudEl = null, hudBadge = null, hudLabel = null, hudPrompt = null, hudDots = [];
  function ensureHud() {
    if (hudEl && hudEl.isConnected) return hudEl;
    hudEl = document.createElement('div'); hudEl.id = 'stepHud';
    hudEl.style.cssText = 'position:absolute;top:10px;left:0;right:0;z-index:10005;display:none;flex-direction:column;align-items:center;gap:8px;pointer-events:none;padding:0 10px';
    hudBadge = document.createElement('div'); hudBadge.className = 'hud-badge';
    hudBadge.style.cssText = 'display:flex;align-items:center;gap:12px;padding:10px 20px;border-radius:999px;font:800 15px/1.2 system-ui,sans-serif;color:#fff;background:rgba(15,23,42,.85);border:1.5px solid ' + HUD_BLUE + ';transition:border-color .2s';
    hudDots = [0, 1].map(() => { const d = document.createElement('span'); d.className = 'hud-dot idle'; d.style.cssText = 'width:16px;height:16px;border-radius:50%;background:rgba(255,255,255,.25);box-shadow:0 0 0 3px rgba(255,255,255,.12),0 0 10px currentColor;transition:background .2s,box-shadow .2s'; return d; });
    hudLabel = document.createElement('span');
    hudBadge.append(hudDots[0], hudDots[1], hudLabel);
    hudPrompt = document.createElement('div'); hudPrompt.className = 'hud-prompt';
    hudPrompt.style.cssText = 'padding:8px 18px;border-radius:14px;color:#fff;background:rgba(15,23,42,.8);border:1.5px solid rgba(56,189,248,.6);text-align:center;display:none;font-family:system-ui,sans-serif;font-weight:800';
    hudEl.append(hudBadge, hudPrompt);
    camBox().appendChild(hudEl);
    return hudEl;
  }
  function hideHud() { if (hudEl) hudEl.style.display = 'none'; }
  function hud(res) {
    const box = ensureHud();
    const active = !!res.box || res.stage !== 'search';
    if (res.state === 'idle' || (res.state === 'running' && !active)) { box.style.display = 'none'; return; }
    const two = (res.steps || 1) === 2, step = res.step || 1;
    const done = res.state === 'passed', bad = res.state === 'failed' || res.state === 'unknown';
    hudLabel.textContent = done ? (two ? T('hud.verifiedTwoStep') : T('hud.verified'))
      : two ? T('hud.step', {step: step, name: step === 1 ? T('hud.faceVerification') : T('hud.fingerChallenge')}) : T('hud.faceVerification');
    hudBadge.style.borderColor = bad ? HUD_RED : done ? HUD_GREEN : HUD_BLUE;
    hudDots[0].style.background = (step >= 2 || done) ? DOT_GREEN : (bad ? DOT_RED : DOT_BLUE);
    hudDots[1].style.display = two ? '' : 'none';
    hudDots[1].style.background = done ? DOT_GREEN : (step === 2 ? (bad ? DOT_RED : DOT_BLUE) : 'rgba(255,255,255,.25)');
    { const c0 = hudDots[0].style.background, c1 = hudDots[1].style.background;
      hudDots[0].classList.remove('idle'); hudDots[1].classList.toggle('idle', !(done || step === 2));
      hudDots[0].style.boxShadow = '0 0 0 3px rgba(0,0,0,.12),0 0 10px ' + c0;
      hudDots[1].style.boxShadow = (done || step === 2) ? '0 0 0 3px rgba(0,0,0,.12),0 0 10px ' + c1 : 'none'; }
    let prompt = '', big = false;
    if (res.stage === 'finger' && res.target_fingers && !done) {
      const n = res.target_fingers; big = true;
      prompt = T('hud.matchGesture') + (typeof res.count === 'number' ? T('hud.detected', {count: res.count}) : '');
    } else if (bad) prompt = res.message;
    else if (!done) prompt = res.message;
    hudPrompt.textContent = prompt;
    hudPrompt.style.display = prompt ? 'block' : 'none';
    hudPrompt.style.fontSize = big ? '20px' : '14px';
    hudPrompt.style.borderColor = bad ? HUD_RED : (res.stage === 'finger' ? FINGER_COLOR : 'rgba(0,229,255,.6)');
    box.style.display = 'flex';
  }

  /* ---------- Finger tracking overlay: green ring + thin green stem, smoothly interpolated  ---------- */
  let fCanvas = null, fRAF = 0; const fState = new Map();
  const lerp2 = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
  function fingerCanvas(video) {
    if (!fCanvas || !fCanvas.isConnected) { fCanvas = document.createElement('canvas'); fCanvas.id = 'fingerOverlay'; camBox().appendChild(fCanvas); }
    if (fCanvas.width !== video.videoWidth) fCanvas.width = video.videoWidth;
    if (fCanvas.height !== video.videoHeight) fCanvas.height = video.videoHeight;
    return fCanvas;
  }
  function setFingers(list, video) {
    fingerCanvas(video);
    const now = performance.now(), seen = new Set();
    for (const f of list) {
      seen.add(f.n);
      const s = fState.get(f.n);
      if (s) {                                   /* glide from where the dot is now to the new position */
        s.fT = s.cT; s.fB = s.cB; s.tT = f.tip; s.tB = f.base;
        s.dur = Math.min(160, Math.max(40, now - s.t0)); s.t0 = now; s.seen = now;
      } else fState.set(f.n, { fT: f.tip, fB: f.base, tT: f.tip, tB: f.base, cT: f.tip, cB: f.base, t0: now, dur: 120, seen: now });
    }
    for (const [k, s] of fState) if (!seen.has(k) && now - s.seen > 200) fState.delete(k);
    if (!fRAF) fRAF = requestAnimationFrame(paintFingers);
  }
  function paintFingers() {
    fRAF = 0;
    const c = fCanvas;
    if (!c || !c.isConnected) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (!fState.size) return;
    const now = performance.now(), W = c.width, H = c.height;
    const r = Math.max(5, W * 0.011), lw = Math.max(1.6, W * 0.0028);
    const CASING = 'rgba(2,6,23,.55)';                     /* dark casing keeps the line legible on any background */
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const s of fState.values()) {
      const k = Math.min(1, (now - s.t0) / s.dur);
      s.cT = lerp2(s.fT, s.tT, k); s.cB = lerp2(s.fB, s.tB, k);
      const bx = s.cB[0] * W, by = s.cB[1] * H, tx = s.cT[0] * W, ty = s.cT[1] * H;
      /* stem: casing + thin accent line */
      ctx.shadowBlur = 0; ctx.strokeStyle = CASING; ctx.lineWidth = lw + 2.2;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.shadowColor = 'rgba(16,185,129,.55)'; ctx.shadowBlur = 5; ctx.strokeStyle = FINGER_COLOR; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tx, ty); ctx.stroke();
      /* base joint: small solid dot */
      ctx.shadowBlur = 0; ctx.fillStyle = FINGER_COLOR;
      ctx.beginPath(); ctx.arc(bx, by, r * 0.38, 0, Math.PI * 2); ctx.fill();
      /* tip: hollow ring with a small centre dot */
      ctx.strokeStyle = CASING; ctx.lineWidth = lw + 2.2;
      ctx.beginPath(); ctx.arc(tx, ty, r, 0, Math.PI * 2); ctx.stroke();
      ctx.shadowColor = 'rgba(16,185,129,.55)'; ctx.shadowBlur = 5; ctx.strokeStyle = FINGER_COLOR; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.arc(tx, ty, r, 0, Math.PI * 2); ctx.stroke();
      ctx.shadowBlur = 0; ctx.fillStyle = FINGER_COLOR;
      ctx.beginPath(); ctx.arc(tx, ty, r * 0.32, 0, Math.PI * 2); ctx.fill();
    }
    ctx.shadowBlur = 0;
    fRAF = requestAnimationFrame(paintFingers);
  }
  function clearFingers() {
    fState.clear();
    if (fRAF) { cancelAnimationFrame(fRAF); fRAF = 0; }
    if (fCanvas && fCanvas.isConnected) fCanvas.getContext('2d').clearRect(0, 0, fCanvas.width, fCanvas.height);
  }

  /* ---------- Hand-gesture challenge overlay (transparent, right-docked, vector-animated; left OR right hand accepted) ---------- */
  /* gesture captions are loaded from text.json (gesture.names) */
  const GESTURE_IMAGES = { 1: 'finger1.png', 2: 'finger2.png', 3: 'finger3.png', 4: 'finger4.png' };   /* gesture pictures (1-4 fingers) */
  let gEl = null, gN = 0, gState = '';
  Object.keys(GESTURE_IMAGES).forEach(function (k) { const im = new Image(); im.src = GESTURE_IMAGES[k]; });   /* preload so there is no flicker */
  function ensureGesture() {
    if (gEl && gEl.isConnected) return gEl;
    if (!document.getElementById('gestureCss')) {
      const st = document.createElement('style'); st.id = 'gestureCss';
      st.textContent = `
      #gestureCard{position:absolute;right:6px;top:30%;width:128px;z-index:10006;pointer-events:none;display:none;flex-direction:column;align-items:center;gap:5px;
        background:none;border:0;box-shadow:none;padding:0;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#fff;text-align:center;
        text-shadow:0 1px 3px rgba(0,0,0,.85),0 0 8px rgba(0,0,0,.55)}
      #gestureCard .g-wrap{position:relative;width:112px;height:140px;display:flex;align-items:center;justify-content:center}
      #gestureCard .g-ring{position:absolute;left:50%;top:56%;width:86px;height:86px;margin:-43px 0 0 -43px;border-radius:50%;border:2px solid rgba(56,189,248,.8);opacity:0;animation:gRing 2.4s ease-out infinite}
      #gestureCard .g-ring.r2{animation-delay:1.2s}
      #gestureCard .g-svg{position:relative;width:112px;height:140px;filter:drop-shadow(0 2px 4px rgba(0,0,0,.55)) drop-shadow(0 0 8px rgba(56,189,248,.55));animation:gGlow 1.8s ease-in-out infinite}
      #gestureCard .g-hand{display:block;width:100%;height:100%;animation:gFlip 5s ease-in-out infinite}
      #gestureCard .g-img{display:block;width:100%;height:100%;object-fit:contain;transform-origin:bottom center;user-select:none;-webkit-user-drag:none;animation:gImgIn .55s cubic-bezier(.2,.9,.3,1.25) both,gBob 2s ease-in-out .7s infinite}
      #gestureCard .g-cap{font-weight:800;font-size:14px;letter-spacing:.12em;line-height:1.1}
      #gestureCard .g-sub{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;letter-spacing:.06em;line-height:1}
      #gestureCard .g-dot{width:8px;height:8px;border-radius:50%;background:#F59E0B;animation:gBlink 1s ease-in-out infinite}
      #gestureCard .g-hint{font-size:10px;letter-spacing:.06em;opacity:.85;line-height:1.1}
      #gestureCard .g-chk{position:absolute;right:-2px;top:2px;width:42px;height:42px;display:none}
      #gestureCard .g-chk circle{fill:#10B981}
      #gestureCard .g-chk path{fill:none;stroke:#fff;stroke-width:9;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:70;stroke-dashoffset:70}
      #gestureCard.match .g-dot{background:#38BDF8}
      #gestureCard.match .g-ring{border-color:rgba(56,189,248,1);animation-duration:1.1s}
      #gestureCard.ok .g-ring{border-color:rgba(16,185,129,.9);animation-duration:1.4s}
      #gestureCard.ok .g-svg{animation:none;filter:drop-shadow(0 0 12px rgba(16,185,129,.95))}
      #gestureCard.ok .g-hand{animation:none}
            #gestureCard.ok .g-dot{background:#10B981;animation:none}
      #gestureCard.ok .g-chk{display:block}
      #gestureCard.ok .g-chk circle{animation:gPop .4s ease both;transform-origin:50% 50%}
      #gestureCard.ok .g-chk path{animation:gDraw .4s ease .2s forwards}
      @keyframes gFlip{0%,40%{transform:scaleX(1)}50%{transform:scaleX(.02)}58%,92%{transform:scaleX(-1)}100%{transform:scaleX(1)}}
      @keyframes gRing{0%{transform:scale(.55);opacity:.85}100%{transform:scale(1.5);opacity:0}}
      @keyframes gGlow{0%,100%{filter:drop-shadow(0 2px 4px rgba(0,0,0,.55)) drop-shadow(0 0 4px rgba(56,189,248,.35))}50%{filter:drop-shadow(0 2px 4px rgba(0,0,0,.55)) drop-shadow(0 0 14px rgba(56,189,248,.95))}}
      @keyframes gBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}
      @keyframes gBlink{0%,100%{opacity:1}50%{opacity:.35}}
      @keyframes gImgIn{from{transform:translateY(14px) scale(.6);opacity:0}to{transform:translateY(0) scale(1);opacity:1}}
      @keyframes gRise{from{transform:scaleY(.15);opacity:0}to{transform:scaleY(1);opacity:1}}
      @keyframes gPop{0%{transform:scale(.3)}70%{transform:scale(1.15)}100%{transform:scale(1)}}
      @keyframes gDraw{to{stroke-dashoffset:0}}
      @media (max-width:420px){#gestureCard{width:108px;right:4px}#gestureCard .g-wrap{width:96px;height:120px}#gestureCard .g-svg,#gestureCard .g-hand{width:96px;height:120px}#gestureCard .g-cap{font-size:12.5px}}`;
      document.head.appendChild(st);
    }
    gEl = document.createElement('div'); gEl.id = 'gestureCard';
    camBox().appendChild(gEl);
    return gEl;
  }
  /* n = required fingers (1-4), count = fingers currently detected (face is guaranteed present by the server for this frame) */
  function showGesture(n, count) {
    const el = ensureGesture();
    if (gN !== n || !el.firstChild) {
      gN = n; gState = '';
      el.innerHTML = '<div class="g-wrap"><span class="g-ring"></span><span class="g-ring r2"></span><div class="g-svg"><div class="g-hand"><img class="g-img" src="' + GESTURE_IMAGES[n] + '" alt="" draggable="false"></div></div>' +
        '<svg class="g-chk" viewBox="0 0 100 100"><circle cx="50" cy="50" r="46"/><path d="M28 52 L44 68 L73 34"/></svg></div>' +
        '<div class="g-cap">' + (T('gesture.names.' + n) || T('gesture.namesFallback', {n: n})) + '</div><div class="g-sub"><span class="g-dot"></span><span class="g-st"></span></div>' +
        '<div class="g-hint">' + T('gesture.eitherHand') + '</div>';
    }
    if (gState === 'ok') return;
    const matching = typeof count === 'number' && count === n;
    el.classList.toggle('match', matching);
    el.querySelector('.g-st').textContent = matching ? T('gesture.holdSteady') : T('gesture.waiting');
    el.style.display = 'flex';
  }
  function gestureSuccess() {
    const el = ensureGesture();
    if (!el.firstChild && gN) showGesture(gN, gN);
    gState = 'ok'; el.classList.remove('match'); el.classList.add('ok');
    const st = el.querySelector('.g-st'); if (st) st.textContent = T('gesture.verified');
    el.style.display = 'flex';
  }
  function hideGesture() { if (gEl) { gEl.style.display = 'none'; gEl.classList.remove('ok', 'match'); gEl.innerHTML = ''; } gN = 0; gState = ''; }

  window.runFaceScan = async function () {
    if (!backend.ok) { setScanStatus(T('scanner.status.connecting'), 'detecting'); scanGapMs = 700; await ping(); isProcessingFrame = false; return; }
    isProcessingFrame = true;
    const video = document.getElementById('video');
    const overlay = document.getElementById('overlay');
    const guide = document.getElementById('camGuide');
    if (!video.videoWidth) { isProcessingFrame = false; return; }
    if (!employees.some(e => e.descriptor)) { setScanStatus(T('scanner.status.noRecords'), 'fail'); isProcessingFrame = false; return; }

    if (overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
    if (overlay.height !== video.videoHeight) overlay.height = video.videoHeight;
    const ctx = overlay.getContext('2d');
    const draw = (box, color, lw = 2) => {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (!box) return;
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      ctx.strokeRect(box[0] * overlay.width, box[1] * overlay.height, box[2] * overlay.width, box[3] * overlay.height);
    };

    let sid, res = null, fast = false, announced = false, errs = 0;
    for (let i = 0; i < 3 && !sid; i++) {                  /* cold/slow server: retry quietly instead of failing */
      try { sid = (await api('/api/verify/start', { method: 'POST', body: '{}' }, 6000)).session_id; }
      catch (e) { await sleep(500 * (i + 1)); }
    }
    if (!sid) { setScanStatus(T('scanner.status.connecting'), 'detecting'); scanGapMs = 800; ping(); isProcessingFrame = false; return; }

    try {
      while (scanStream) {
        updateFlashlightState(video);
        try {
          res = await api('/api/verify/frame', { method: 'POST', body: JSON.stringify({ session_id: sid, image: fast ? grab(video, CFG.fingerWidth, CFG.fingerQuality) : grab(video, CFG.width) }) }, 8000);
          errs = 0;
        } catch (fe) {                                       /* one dropped/slow frame must not kill the scan */
          errs++;
          if (!scanStream || errs > 5) throw fe;
          if (/session/i.test(fe.message || '')) {           /* session lost (server restart / other worker) -> fresh one, silently */
            sid = (await api('/api/verify/start', { method: 'POST', body: '{}' }, 6000)).session_id;
            fast = false; announced = false; clearFingers();
          }
          setScanStatus(T('scanner.status.reconnecting'), 'detecting');
          await sleep(200 * errs);
          continue;
        }
        if (res.stage !== 'search') isVerifyingLiveness = true;
        guide.classList.toggle('active', !!res.box);
        draw(res.box, '#38bdf8');
        hud(res); fast = res.stage === 'finger';      /* smaller frames while tracking fingers */
        if (res.face_verified && !announced) {        /* Step 1/2 complete -> employee name + "Face Verification Complete" */
          announced = true;
          const who = employees.find(e => e.id === res.employee_id);
          announceFaceVerified(who ? who.name : String(res.employee_id));
        }

        if (res.state === 'running') {
          const two = (res.steps || 1) === 2;
          if (res.stage === 'finger') {                                /* Step 2/2 */
            setFingers(res.face_present === false ? [] : (res.fingers || []), video);
            const n = res.target_fingers;
            showGesture(n, res.count);
            setScanStatus(T(n === 1 ? 'scanner.status.matchGestureOne' : 'scanner.status.matchGestureMany', {n: n}), 'detecting');
          } else {
            hideGesture();                                                     /* Step 1/2 */
            setScanStatus((two ? T('scanner.status.step1Prefix') : '') + res.message, 'detecting');
          }
          await sleep(CFG.gapMs);
          continue;
        }
        break;
      }
    } catch (e) {
      res = null;                                           /* never fall back to face-only: attendance needs BOTH challenges */
      setScanStatus(T('scanner.status.unstable'), 'detecting'); scanGapMs = 500; ping();
    }

    isVerifyingLiveness = false;
    clearFingers();
    if (!scanStream || !res) { hideGesture(); hideHud(); isProcessingFrame = false; return; }
    if (!(res.state === 'passed' && res.face_verified && res.finger_verified && res.stage === 'passed')) hideGesture();

    if (res.state === 'idle') {
      guide.classList.remove('active'); draw(null);
      scanGapMs = 220; setScanStatus(T('scanner.status.scanning'), 'detecting');
    } else if (res.state === 'unknown') {
      draw(res.box, '#f43f5e', 3);
      if (res.message === 'No face records enrolled') { setScanStatus(T('scanner.status.syncingRecords'), 'detecting'); scanGapMs = 2000; syncMissing(); }
      else { setScanStatus(res.message, 'fail'); scanGapMs = 450; }
    } else if (res.state === 'passed' && !(res.face_verified && res.finger_verified && res.stage === 'passed')) {
      draw(res.box, '#f43f5e', 3);
      setScanStatus(T('scanner.status.bothRequired'), 'fail'); toast(T('scanner.status.bothRequired'), 'err');
    } else if (res.state === 'passed') {
      draw(res.box, '#10b981', 3);
      if (res.target_fingers) { showGesture(res.target_fingers, res.target_fingers); }
      gestureSuccess(); await sleep(1100);                  /* SUCCESS / checkmark state before completing verification */
      hideGesture();
      const emp = employees.find(e => e.id === res.employee_id);
      if (emp) { setScanStatus(T('scanner.fp.verified', {name: emp.name}), 'success'); logAttendance(emp, T('methods.faceFinger')); }
      else { setScanStatus(T('scanner.status.employeeMissing'), 'fail'); toast(T('scanner.status.employeeNotFound', {id: res.employee_id}), 'err'); }
    } else {
      draw(res.box, '#f43f5e', 3);
      if (res.status === 'FACE_LOST') { setScanStatus(T('scanner.status.faceLost'), 'fail'); toast(T('scanner.status.faceLostToast', {message: res.message}), 'err'); }
      else { setScanStatus(res.message, 'fail'); toast(res.message, 'err'); }
      await sleep(1500);
    }
    if (res.state !== 'idle') setTimeout(hideHud, 1600);
    isProcessingFrame = false;
  };

  /* ---------- ENROLLMENT (embedding computed & stored on server) ---------- */
  document.getElementById('captureFaceBtn').addEventListener('click', async (e) => {
    if (!backend.ok) return;                                /* let the original face-api handler run */
    e.stopImmediatePropagation();
    const statusEl = document.getElementById('regFaceStatus');
    const id = document.getElementById('fEmpId').value.trim();
    if (!id) { statusEl.textContent = T('enroll.enterIdFirst'); statusEl.className = 'enroll-status err'; return; }
    const video = document.getElementById('regVideo'), preview = document.getElementById('galleryPreview');
    const fromGallery = currentEnrollSource === 'gallery';
    statusEl.textContent = T('enroll.enrolling'); statusEl.className = 'enroll-status';
    window.__enrolling = true;
    try {
      const images = [];
      if (fromGallery) images.push(grab(preview, 640, 0.85));
      else for (let i = 0; i < 4; i++) { images.push(grab(video, 640, 0.85)); await sleep(150); }
      const r = await api('/api/enroll', { method: 'POST', body: JSON.stringify({ employee_id: id, images }) }, 90000);
      capturedDescriptor = new Array(128).fill(0);          /* placeholder flag; the real embedding lives on the server */
      capturedPhoto = fromGallery ? preview.src : snapshotToDataUrl(video);
      statusEl.textContent = T(r.samples > 1 ? 'enroll.capturedMany' : 'enroll.capturedOne', {count: r.samples});
      statusEl.className = 'enroll-status ok';
      updateEnrollBadges();
    } catch (err) {
      statusEl.textContent = err.name === 'AbortError' ? T('enroll.serverSlow') : err.message;
      statusEl.className = 'enroll-status err';
    } finally { window.__enrolling = false; }
  }, true);

  /* ---------- optional: keep server embeddings in sync with local employee list ---------- */
  if (CFG.pruneSync) {
    const _re = window.renderEmployees; let tm = null;
    window.renderEmployees = function () {
      const r = _re.apply(this, arguments);
      clearTimeout(tm);
      tm = setTimeout(() => { if (backend.ok && employees.length) api('/api/prune', { method: 'POST', body: JSON.stringify({ keep: employees.map(e => e.id) }) }).catch(() => {}); }, 1500);
      return r;
    };
  }
})();
