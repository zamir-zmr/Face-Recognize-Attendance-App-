const LS_EMP = 'minmoe_employees';
const LS_LOG = 'minmoe_logs';
const LS_PIN = 'minmoe_admin_pin';
const LS_MODE = 'minmoe_scan_mode';
const FACE_MATCH_THRESHOLD = 0.52;

let employees = [];
let logs = [];
let isAdminUnlocked = false;
let pendingPageTarget = null;
let currentPage = 'scanner';
let activeFilter = 'today';
let manualFlashlightState = false;

/* TOAST THROTTLE / COOLDOWN PREVENT SPAM */
let lastToastMessage = '';
let lastToastTime = 0;

/* ===== FIREBASE SYNC (business data only: employees, face data, attendance logs, photos). flashlight / camera / UI state stay local; PINs live in Ash_Attendance_Pin (+ local copy) ===== */
const FB_CONFIG = {
  apiKey: "AIzaSyA0rirztMO13FyXcKYz1aEB1ERYH-HQUbA",
  authDomain: "sweethouse-e3e49.firebaseapp.com",
  databaseURL: "https://sweethouse-e3e49-default-rtdb.firebaseio.com",
  projectId: "sweethouse-e3e49",
  storageBucket: "sweethouse-e3e49.firebasestorage.app",
  messagingSenderId: "579322038108",
  appId: "1:579322038108:web:166b7fdb103080f56d6399"
};
firebase.initializeApp(FB_CONFIG);
const fdb = firebase.database(), fst = firebase.storage();
const fbKey = s => encodeURIComponent(String(s)).replace(/\./g, '%2E');
const clean = o => JSON.parse(JSON.stringify(o, (k, v) => (v === null || v === '' ? undefined : v)));
const sj = o => JSON.stringify(o, Object.keys(o).sort());
function askConfirm(title, msg, okText){
  return new Promise(res => {
    const bg = document.createElement('div'); bg.className = 'cf-bg';
    bg.innerHTML = '<div class="cf-card"><div class="cf-ico"><svg viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v5M14 11v5"/></svg></div><div class="cf-t"></div><div class="cf-m"></div><div class="cf-row"><button class="cf-no"></button><button class="cf-ok"></button></div></div>';
    bg.querySelector('.cf-t').textContent = title; bg.querySelector('.cf-m').textContent = msg; bg.querySelector('.cf-ok').textContent = okText || T('common.delete'); bg.querySelector('.cf-no').textContent = T('common.cancel');
    document.body.appendChild(bg);
    requestAnimationFrame(() => requestAnimationFrame(() => bg.classList.add('show')));
    const done = v => { bg.classList.remove('show'); setTimeout(() => bg.remove(), 300); res(v); };
    bg.querySelector('.cf-no').onclick = () => done(false);
    bg.querySelector('.cf-ok').onclick = () => done(true);
    bg.addEventListener('click', e => { if(e.target === bg) done(false); });
  });
}
let remoteEmp = {}, empReady = false;
const CACHE_KEY = 'fb_cache_emp_v1';                /* read-only copy of the last Firebase snapshot (never used for writes) */
const purgedBackend = new Set(), importingKeys = new Set(); let tombKeys = [];
function purgeLocal(keys){                        /* remove deleted employees from every browser-side copy */
  const ks = new Set(keys);
  [LS_EMP, LS_EMP + '_bak', CACHE_KEY].forEach(n => { try{
    const arr = JSON.parse(localStorage.getItem(n) || 'null'); if(!Array.isArray(arr)) return;
    localStorage.setItem(n, JSON.stringify(arr.filter(e => !ks.has(fbKey(e.id)))));
  }catch(e){} });
}
function purgeBackend(){                          /* make sure the Python server also dropped them (retries until it is reachable) */
  tombKeys.forEach(k => { if(purgedBackend.has(k) || importingKeys.has(k) || !window.faceBackendDelete) return;
    const p = window.faceBackendDelete(decodeURIComponent(k)); if(p) purgedBackend.add(k); });
}
setInterval(purgeBackend, 15000);
const logCache = new Map();
function rerender(){
  if(currentPage === 'dashboard') renderDashboard();
  if(currentPage === 'employees') renderEmployees();
  if(currentPage === 'reports') renderReports();
}
/* ---- Employee writes are ALWAYS per record. Nothing is ever deleted because it is "missing" from an array. ---- */
const safeId = id => { const s = String(id == null ? '' : id).trim(); if(!s) throw new Error(T('errors.emptyId')); return s; };
function safeRef(path){                           /* refuses collection-level paths, so one bad id can never wipe everything */
  const p = path.split('/');
  if(p.length < 2 || p.some(x => !x)) throw new Error(T('errors.collectionAccess', {path: path}));
  return fdb.ref(p.join('/'));
}
const TS = () => firebase.database.ServerValue.TIMESTAMP;
async function fbPutEmployee(e){                  /* add / edit / import: upsert exactly this employee; the record is saved FIRST */
  const id = safeId(e.id), k = fbKey(id), old = remoteEmp[k] || {};
  e.id = id;
  const newPhoto = /^data:/.test(e.photo || '') && (e.photo !== old.photo || !e.photoUrl);
  await safeRef('deleted_employees/' + k).remove();       /* (re-)adding an ID lifts its delete marker */
  await safeRef('employees/' + k).set({ ...clean(e), updatedAt: TS() });
  const ref = fst.ref('face_images/' + k + '/profile.jpg');
  if(newPhoto){                                           /* Storage copy: background, time-boxed, can never lose the record */
    (async () => {
      try{
        await Promise.race([ref.putString(e.photo, 'data_url'), new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 20000))]);
        const url = await ref.getDownloadURL();
        await fdb.ref('employees/' + k).transaction(cur => cur ? { ...cur, photoUrl: url } : cur);   /* no-op if deleted meanwhile */
      }catch(pe){ console.warn('Storage copy failed', id, pe.message); }
    })();
  } else if(!e.photo && old.photoUrl){ ref.delete().catch(()=>{}); }
}
async function fbDeleteEmployee(rawId){           /* removes ONLY this employee (+ their face data / photo) */
  const id = safeId(rawId), k = fbKey(id);
  const rec = remoteEmp[k];
  if(rec) await safeRef('employees_trash/' + k).set({ ...clean(rec), deletedAt: firebase.database.ServerValue.TIMESTAMP });   /* recoverable copy */
  await safeRef('deleted_employees/' + k).set(firebase.database.ServerValue.TIMESTAMP);
  await safeRef('employees/' + k).remove();
  fst.ref('face_images/' + k + '/profile.jpg').delete().catch(()=>{});
  safeRef('face_profiles/' + k).remove().then(() => fdb.ref('face_profiles').once('value'))
    .then(s => fdb.ref('face_meta').set({ count: s.numChildren(), updated_at: firebase.database.ServerValue.TIMESTAMP })).catch(()=>{});
  purgeLocal([k]);
  if(window.faceBackendDelete) window.faceBackendDelete(id);
}
async function saveEmployees(list){               /* upsert only the given records */
  list = list || [];
  const keys = list.map(e => fbKey(safeId(e.id)));
  keys.forEach(k => { importingKeys.add(k); purgedBackend.delete(k); });   /* no pending server purge may hit an id being (re)added */
  try{
    await Promise.all(keys.map(k => safeRef('deleted_employees/' + k).remove()));   /* lift all delete markers first */
    for(const e of list) await fbPutEmployee(e);
    return true;
  }catch(err){ toast(T('toast.firebaseSyncFailed', {error: err.message}), 'err'); return false; }
  finally{ setTimeout(() => keys.forEach(k => importingKeys.delete(k)), 8000); }
}
function saveLogs(entry){
  if(!entry) return;
  if(!entry.id) entry.id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  fdb.ref('attendance_logs/' + entry.date + '/' + entry.id).set(clean(entry)).catch(err => toast(T('toast.firebaseSyncFailed', {error: err.message}), 'err'));
}
function putLogs(byDate, dates){
  [...new Set(dates || Object.keys(byDate))].forEach(d => {
    for(const [id, l] of [...logCache]) if(l.date === d) logCache.delete(id);
    Object.entries(byDate[d] || {}).forEach(([id, l]) => logCache.set(id, { ...l, id, date: d }));
  });
  logs = [...logCache.values()];
}
let empSeq = 0, wasOnline = null;
setTimeout(() => {                                /* Firebase unreachable at startup: show the last synced data (read-only) instead of an empty list */
  if(empReady) return;
  try{ const c = JSON.parse(localStorage.getItem(CACHE_KEY) || '[]'); if(c.length){ employees = c; rerender(); toast(T('toast.offlineCached'), 'err'); } }catch(e){}
}, 6000);
fdb.ref('.info/connected').on('value', s => {
  if(s.val() === true){ if(wasOnline === false) toast(T('toast.backOnline'), 'ok'); wasOnline = true; }
  else if(wasOnline === true){ toast(T('toast.offlineSync'), 'err'); wasOnline = false; }
});
fdb.ref('employees').on('value', async snap => {
  const seq = ++empSeq;
  remoteEmp = snap.val() || {};
  let tomb = {};
  try{ tomb = (await fdb.ref('deleted_employees').once('value')).val() || {}; }catch(e){}
  if(seq !== empSeq) return;                       /* a newer snapshot arrived while waiting */
  Object.keys(remoteEmp).forEach(k => {            /* deleted employees can never come back (self-heals stale re-adds) */
    if(k && tomb[k] && !(((remoteEmp[k] || {}).updatedAt || 0) > Number(tomb[k]))){   /* only records NOT re-saved after the delete */
      safeRef('employees/' + k).remove().catch(()=>{}); delete remoteEmp[k];
    }
  });
  tombKeys = Object.keys(tomb);
  if(tombKeys.length){ purgeLocal(tombKeys); purgeBackend(); }
  employees = Object.values(remoteEmp).map(e => ({ ...e }));
  try{ localStorage.setItem(CACHE_KEY, JSON.stringify(employees)); }catch(e){}
  if(!empReady){
    empReady = true;
    await migrateLocal(tomb);
    if(localStorage.getItem('fb_migrated2')){      /* old browser copy is stale now - never show/restore it again */
      try{ localStorage.setItem(LS_EMP + '_bak', localStorage.getItem(LS_EMP) || '[]'); }catch(e){}
      localStorage.removeItem(LS_EMP); localStorage.removeItem(LS_LOG);
    }
  }
  rerender();
}, err => toast(T('toast.firebaseReadFailed', {error: err.message}), 'err'));   /* keep last good state; never rebuild from a stale copy */
async function migrateLocal(tomb){           /* one-time upload of data previously kept in this browser */
  if(localStorage.getItem('fb_migrated2')) return;
  try{
    const le = JSON.parse(localStorage.getItem(LS_EMP) || '[]').filter(e => !(tomb || {})[fbKey(e.id)]);
    const ll = JSON.parse(localStorage.getItem(LS_LOG) || '[]');
    let ok = true;
    if(le.length && !Object.keys(remoteEmp).length){ ok = await saveEmployees(le); }
    if(ok){
      await Promise.all(ll.filter(l => l.date).map(l => { l.id = l.id || ('m' + (l.lastUpdated || 0).toString(36) + fbKey(l.empId));
        return fdb.ref('attendance_logs/' + l.date + '/' + l.id).set(clean(l)); }));
      localStorage.setItem('fb_migrated2', '1');
    }
  }catch(e){ toast(T('toast.migrationFailed', {error: e.message}), 'err'); }
}
let todayRef = null, watchedDay = '';
function watchToday(){
  const d = todayStr();
  if(d === watchedDay) return;
  if(todayRef) todayRef.off();
  watchedDay = d;
  todayRef = fdb.ref('attendance_logs/' + d);
  todayRef.on('value', s => { putLogs({ [d]: s.val() || {} }); rerender(); });
}
watchToday(); setInterval(watchToday, 60000);
setInterval(() => { if(currentPage === 'reports' && activeFilter !== 'today') refreshReports(); }, 20000);
async function fetchRange(){
  const t = todayStr(); let a, b;
  if(activeFilter === 'today') a = b = t;
  else if(activeFilter === 'yesterday'){ const y = new Date(); y.setDate(y.getDate() - 1); a = b = y.toISOString().slice(0, 10); }
  else if(activeFilter === 'month'){ a = t.slice(0, 7) + '-01'; b = t.slice(0, 7) + '-31'; }
  else { a = b = document.getElementById('customDate').value; if(!a) return; }
  const snap = await fdb.ref('attendance_logs').orderByKey().startAt(a).endAt(b).once('value');
  const v = snap.val() || {};
  putLogs(v, Object.keys(v).concat(a === b ? [a] : []));
}
async function refreshReports(){
  try{ await fetchRange(); }catch(e){ toast(T('toast.firebaseError', {error: e.message}), 'err'); }
  renderReports();
}

const CHECK_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>';

function toast(msg, type='info'){
  const now = Date.now();
  if (msg === lastToastMessage && (now - lastToastTime) < 3000) {
    return;
  }
  lastToastMessage = msg;
  lastToastTime = now;

  const host = document.getElementById('toastHost');
  const t = document.createElement('div');
  if(type === 'name'){                       /* verified employee name: white text, clean contrast */
    t.className = 'toast name';
    const body = document.createElement('div'); body.className = 't-body';
    const cap = document.createElement('span'); cap.className = 't-cap'; cap.textContent = T('toast.employeeIdentified');
    const main = document.createElement('span'); main.className = 't-main'; main.textContent = msg;
    body.append(cap, main); t.appendChild(body);
  } else if(type === 'verify'){              /* explicit success status with green check */
    t.className = 'toast verify';
    const ic = document.createElement('span'); ic.className = 't-ic'; ic.innerHTML = CHECK_SVG;
    const body = document.createElement('div'); body.className = 't-body';
    const main = document.createElement('span'); main.className = 't-main'; main.textContent = msg;
    body.appendChild(main); t.append(ic, body);
  } else {
    t.className = 'toast ' + (type==='ok'?'ok':type==='err'?'err':'');
    t.textContent = msg;
  }
  host.appendChild(t);
  setTimeout(()=>t.remove(), 3600);
}

/* Step 1/2 finished: show who was recognised, then the explicit success status */
function announceFaceVerified(name){
  toast(name, 'name');
  toast(T('toast.faceVerificationComplete'), 'verify');
}

function todayStr(){ return new Date().toISOString().slice(0,10); }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function initials(name){ return name.split(' ').map(p=>p[0]).slice(0,2).join('').toUpperCase(); }

/* ===== DUAL PIN SECURITY =====
   Firebase: Ash_Attendance_Pin/AppAccess_PIN (app lock) + Ash_Attendance_Pin/DashboardAccess_PIN (admin tabs).
   A copy of both PINs is kept in localStorage so the app still unlocks offline / when Firebase sync fails. */
const PIN_PATH = 'Ash_Attendance_Pin';
const PIN_KEYS = { app: 'AppAccess_PIN', dash: 'DashboardAccess_PIN' };
const PIN_LS = { app: 'ash_pin_app', dash: 'ash_pin_dash' };
const LS_PIN_PEND = 'ash_pin_pending';
const PinStore = (() => {
  const rd = k => { try{ return localStorage.getItem(k) || ''; }catch(e){ return ''; } };
  const wr = (k, v) => { try{ localStorage.setItem(k, v); }catch(e){} };
  const valid = v => /^\d{4,12}$/.test(String(v == null ? '' : v));
  const withTimeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);
  const getPend = () => { try{ return JSON.parse(rd(LS_PIN_PEND) || '{}') || {}; }catch(e){ return {}; } };
  const setPend = o => wr(LS_PIN_PEND, JSON.stringify(o));
  const clearPend = w => { const p = getPend(); delete p[w]; setPend(p); };
  /* local = PIN this device was verified/created with.  remote = what Firebase holds right now (the master copy). */
  const st = { local: { app: '', dash: '' }, remote: { app: '', dash: '' }, loaded: false, online: null, connected: false, sessionOpen: false };
  ['app', 'dash'].forEach(w => { const v = rd(PIN_LS[w]); st.local[w] = valid(v) ? v : ''; });
  const legacy = rd(LS_PIN);                         /* old local-only admin PIN -> keep it, push to Firebase once */
  if(!st.local.dash && valid(legacy)){ st.local.dash = legacy; wr(PIN_LS.dash, legacy); const p = getPend(); p.dash = 'mig'; setPend(p); }
  /* PIN that must be matched: a pending local change wins, otherwise Firebase (once read), otherwise the device copy (offline) */
  const eff = w => { if(getPend()[w] && st.local[w]) return st.local[w]; return st.loaded ? st.remote[w] : st.local[w]; };
  function applyRemote(v){
    const pend = getPend();
    ['app', 'dash'].forEach(w => {
      const rem = valid(v[PIN_KEYS[w]]) ? String(v[PIN_KEYS[w]]) : '';
      st.remote[w] = rem;
      if(pend[w] && rem && (rem === st.local[w] || pend[w] === 'mig')) delete pend[w];
    });
    setPend(pend); st.loaded = true;
  }
  function flush(){                                  /* push PINs that could not be saved earlier */
    const pend = getPend();
    Object.keys(pend).forEach(w => {
      if(!st.local[w]) return;
      try{
        const ref = fdb.ref(PIN_PATH + '/' + PIN_KEYS[w]);
        if(pend[w] === 'set') ref.set(st.local[w]).then(() => { clearPend(w); st.remote[w] = st.local[w]; }).catch(() => {});
        else ref.transaction(cur => cur === null ? st.local[w] : undefined).then(r => { clearPend(w); if(r.committed) st.remote[w] = st.local[w]; else load(true); }).catch(() => {});
      }catch(e){}
    });
  }
  let loading = null;
  function load(force, ms){
    if(loading && !force) return loading;
    loading = withTimeout(fdb.ref(PIN_PATH).once('value'), ms || 7000).then(snap => {
      applyRemote(snap.val() || {}); st.online = true; flush();
      return { ok: true };
    }).catch(err => { st.online = false; return { ok: false, error: err }; })
      .then(r => { loading = null; return r; });
    return loading;
  }
  async function save(w, pin){
    st.local[w] = pin; wr(PIN_LS[w], pin);
    const p = getPend(); p[w] = 'set'; setPend(p);
    try{
      await withTimeout(fdb.ref(PIN_PATH + '/' + PIN_KEYS[w]).set(pin), 7000);
      clearPend(w); st.remote[w] = pin; st.online = true;
      return { synced: true };
    }catch(err){
      st.online = false;                             /* stays pending - Firebase SDK also queues the write */
      return { synced: false, error: err };
    }
  }
  async function check(w, pin){                      /* Firebase is the master: typed PIN must match what Firebase holds */
    if(!(st.connected && st.loaded)) await load(true, 3500);
    const cur = eff(w);
    if(cur && pin === cur){ st.local[w] = pin; wr(PIN_LS[w], pin); return true; }
    return false;
  }
  async function change(w, cur, pin){
    if(eff(w) && !(await check(w, cur))) return { ok: false, reason: 'wrong' };
    const r = await save(w, pin);
    return { ok: true, synced: r.synced };
  }
  try{                                              /* live: if the App PIN is changed in Firebase while the app is open, lock again */
    fdb.ref(PIN_PATH).on('value', snap => {
      applyRemote(snap.val() || {}); st.online = true;
      if(st.sessionOpen && st.remote.app && st.remote.app !== st.local.app && !getPend().app) window.dispatchEvent(new Event('ash-relock'));
    }, () => {});
    fdb.ref('.info/connected').on('value', s => { st.connected = !!s.val(); if(st.connected){ st.online = true; flush(); } else if(st.online) st.online = false; }, () => {});
  }catch(e){}
  window.addEventListener('online', () => { load(true); });
  return { get: eff, has: w => !!eff(w), local: w => st.local[w], remote: w => st.remote[w], pending: w => !!getPend()[w],
    loaded: () => st.loaded, connected: () => st.connected, isOnline: () => st.connected || st.online,
    setSessionOpen: v => { st.sessionOpen = !!v; }, load, save, check, change };
})();

const PAD_KEYS = ['1','2','3','4','5','6','7','8','9','⌫','0','OK'];
const PAD_ICONS = '<svg class="i-lock" viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>'
  + '<svg class="i-ok" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>'
  + '<svg class="i-err" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg><span class="pin-spin"></span>';
class PinPad{
  constructor(card){
    this.card = card;
    this.badge = card.querySelector('.pin-badge'); this.title = card.querySelector('.gate-title'); this.sub = card.querySelector('.gate-sub');
    this.dotsEl = card.querySelector('.pin-dots'); this.dots = [...this.dotsEl.children]; this.msg = card.querySelector('.gate-msg'); this.kp = card.querySelector('.keypad');
    this.buf = ''; this.busy = false; this.fails = 0; this.lockUntil = 0; this.onSubmit = null; this._t = []; this.len = 0;
    PAD_KEYS.forEach(k => {
      const b = document.createElement('button'); b.textContent = k;
      if(k === '⌫' || k === 'OK') b.classList.add('wide');
      b.addEventListener('click', () => this.key(k));
      this.kp.appendChild(b);
    });
  }
  keyboard(isActive){
    document.addEventListener('keydown', e => {
      if(!isActive() || e.ctrlKey || e.metaKey || e.altKey) return;
      if(/^\d$/.test(e.key)) this.key(e.key);
      else if(e.key === 'Backspace') this.key('⌫');
      else if(e.key === 'Enter') this.key('OK');
      else return;
      e.preventDefault();
    });
  }
  later(fn, ms){ const id = setTimeout(fn, ms); this._t.push(id); return id; }
  texts(title, sub, step){
    this.title.textContent = title; this.sub.textContent = sub;
    if(step){ this.card.classList.remove('step'); void this.card.offsetWidth; this.card.classList.add('step'); }
  }
  state(s){ ['st-ok','st-err','st-busy'].forEach(c => this.card.classList.remove(c)); if(s) this.card.classList.add('st-' + s); }
  setLen(n){ this.len = n || 0; this.render(); }
  render(){
    const max = this.len || 12, want = Math.min(12, Math.max(this.len || 4, this.buf.length));
    if(this.dots.length !== want){
      this.dotsEl.innerHTML = '<i></i>'.repeat(want); this.dots = [...this.dotsEl.children];
    }
    this.kp.querySelectorAll('button:not(.wide)').forEach(b => b.classList.toggle('locked', this.buf.length >= max));
    this.dots.forEach((d, i) => d.classList.toggle('on', i < this.buf.length));
  }
  clear(){ this.buf = ''; this.render(); }
  reset(){                                            /* back to a clean idle pad */
    this._t.forEach(clearTimeout); this._t = [];
    this.buf = ''; this.busy = false; this.state(null);
    this.card.classList.remove('shake', 'step'); this.dotsEl.classList.remove('ok', 'err'); this.kp.classList.remove('off');
    this.msg.textContent = ''; this.render();
  }
  loading(text){ this.busy = true; this.state('busy'); this.kp.classList.add('off'); this.msg.style.color = 'var(--sub)'; this.msg.textContent = text || ''; }
  ready(){ this.busy = false; this.state(null); this.kp.classList.remove('off'); this.msg.style.color = ''; this.msg.textContent = ''; }
  key(k){
    if(this.busy) return;
    if(Date.now() < this.lockUntil) return;
    if(k === '⌫'){ this.buf = this.buf.slice(0, -1); }
    else if(k === 'OK'){ this.submit(); return; }
    else if(this.buf.length < (this.len || 12)){
      this.buf += k;
      if(this.len && this.buf.length === this.len){ this.render(); this.later(() => this.submit(), 170); return; }
    }
    this.render();
  }
  submit(){
    if(this.busy || !this.onSubmit) return;
    if(this.buf.length < 4){ this.fail(T('pin.fourDigits'), true); return; }
    const v = this.buf; this.busy = true; this.kp.classList.add('off'); this.state('busy');
    Promise.resolve(this.onSubmit(v)).catch(() => { this.fail(T('pin.wrong')); });
  }
  fail(text, noCount){
    this.busy = true; this.state('err'); this.dotsEl.classList.add('err'); this.kp.classList.remove('off');
    this.msg.style.color = ''; this.msg.textContent = text || T('pin.wrong');
    this.card.classList.remove('shake'); void this.card.offsetWidth; this.card.classList.add('shake');
    try{ navigator.vibrate && navigator.vibrate([60, 40, 90]); }catch(e){}
    if(!noCount && ++this.fails >= 5){
      this.fails = 0; this.lockUntil = Date.now() + 30000;
      const tick = () => {
        const s = Math.ceil((this.lockUntil - Date.now()) / 1000);
        if(s <= 0){ this.msg.textContent = ''; return; }
        this.msg.textContent = T('pin.locked', { s }); this.later(tick, 500);
      };
      this.later(tick, 900);
    }
    this.later(() => {
      this.buf = ''; this.render(); this.state(null); this.dotsEl.classList.remove('err'); this.card.classList.remove('shake'); this.busy = false;
      if(Date.now() >= this.lockUntil) this.later(() => { if(!this.busy) this.msg.textContent = ''; }, 1200);
    }, 750);
  }
  success(){                                          /* animated tick, resolves when the animation has played */
    this.busy = true; this.fails = 0; this.state('ok'); this.dotsEl.classList.add('ok'); this.kp.classList.add('off'); this.msg.textContent = '';
    try{ navigator.vibrate && navigator.vibrate(35); }catch(e){}
    return new Promise(res => this.later(res, 800));
  }
}
function buildPadCard(card, withBrand){
  if(withBrand) card.innerHTML = '<div class="brand"><span class="dot"></span><h1>' + T('gate.brand') + '</h1></div>';
  card.insertAdjacentHTML('beforeend',
    '<div class="pin-badge">' + PAD_ICONS + '</div><div class="gate-title"></div><p class="gate-sub"></p>'
    + '<div class="pin-dots"><i></i><i></i><i></i><i></i></div><div class="gate-msg"></div><div class="keypad"></div>');
}

/* Create-PIN (enter + confirm) or verify flow. Resolves true when finished. */
function runPinFlow(pad, which, texts){
  return new Promise(resolve => {
    const exists = !!PinStore.get(which);
    let first = '';
    pad.reset();
    if(exists){
      pad.setLen(PinStore.get(which).length);
      pad.texts(texts.title, texts.sub, true);
      pad.onSubmit = async v => {
        if(await PinStore.check(which, v)){ await pad.success(); resolve({ created: false }); }
        else { pad.setLen(PinStore.has(which) ? PinStore.get(which).length : 0); pad.fail(T('pin.wrong')); }
      };
    } else {
      pad.setLen(0);
      pad.texts(texts.createTitle, texts.createSub, true);
      pad.onSubmit = async v => {
        if(!first){ first = v; pad.reset(); pad.setLen(first.length); pad.texts(T('pin.confirmTitle'), T('pin.confirmSub'), true); return; }
        if(v !== first){ first = ''; pad.fail(T('pin.mismatch'), true); pad.later(() => { pad.setLen(0); pad.texts(texts.createTitle, texts.createSub, true); }, 780); return; }
        pad.loading(T('pin.saving'));
        const r = await PinStore.save(which, v);
        await pad.success();
        resolve({ created: true, synced: r.synced });
      };
    }
    if(exists && PinStore.isOnline() === false) pad.msg.textContent = '';
  });
}

/* ----- APP LOCK (Welcome screen) ----- */
const appLockEl = document.getElementById('appLock');
const appLockCard = document.getElementById('appLockCard');
buildPadCard(appLockCard, true);
const appPad = new PinPad(appLockCard);
function endAppLock(){
  const root = document.documentElement;
  root.classList.add('ash-lock-fade'); root.classList.remove('ash-locked'); appLockEl.classList.add('hide');
  setTimeout(() => { if(appLockEl.classList.contains('hide')) appLockEl.style.display = 'none'; root.classList.remove('ash-lock-fade'); }, 700);
}
appPad.keyboard(() => !appLockEl.classList.contains('hide') && appLockEl.style.display !== 'none' && document.documentElement.classList.contains('ash-locked'));
function offlineBlock(){                               /* no connection AND nothing saved on this device -> cannot verify, do not open */
  return new Promise(res => {
    appPad.reset(); appPad.state('err'); appPad.kp.classList.add('off'); appPad.busy = true;
    appPad.texts(T('pin.noConnTitle'), T('pin.noConnSub'));
    const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'pin-retry'; btn.textContent = T('pin.retry');
    appLockCard.appendChild(btn);
    btn.addEventListener('click', () => { btn.remove(); appPad.reset(); res(); });
  });
}
let appLockRunning = false;
async function runAppLock(){
  /* Firebase holds the master App PIN.  Same PIN on this device -> opens silently.
     PIN changed in Firebase / app data reset / new device -> the Firebase PIN must be typed.  No PIN anywhere -> one-time registration. */
  if(appLockRunning) return; appLockRunning = true;
  try{
    for(;;){
      appPad.reset(); appPad.texts(T('pin.appTitle'), T('pin.verifying')); appPad.loading(T('pin.verifying'));
      const r = await PinStore.load(true, 5000);
      appPad.ready();
      const remote = PinStore.remote('app'), local = PinStore.local('app');
      if(r.ok){
        if(remote && remote === local){ break; }                                   /* matches Firebase -> open */
        if(!remote && local && PinStore.pending('app')){ break; }                  /* PIN saved offline earlier, now being pushed */
        const res = await runPinFlow(appPad, 'app', { title: T('pin.appTitle'), sub: T('pin.appLockedSub'),
          createTitle: T('pin.appCreateTitle'), createSub: T('pin.appCreateSub') });
        if(res.created && !res.synced) setTimeout(() => toast(T('pin.savedOffline'), 'info'), 900);
        break;
      }
      if(local){ break; }                                                          /* offline: PIN saved on this device is used */
      await offlineBlock();
    }
  }catch(e){ /* never leave the user stuck behind a broken lock */ }
  appLockRunning = false;
  PinStore.setSessionOpen(true);
  endAppLock();
}
window.addEventListener('ash-relock', () => {          /* Firebase App PIN changed while the app is open */
  if(appLockRunning) return;
  PinStore.setSessionOpen(false);
  const root = document.documentElement;
  root.classList.add('ash-locked'); appLockEl.style.display = ''; appLockEl.classList.remove('hide');
  runAppLock();
});
runAppLock();

/* ----- DASHBOARD / ADMIN GATE (Dashboard, Employees, Reports) - unlocked once per app session ----- */
const gateEl = document.getElementById('gate');
const gateCard = gateEl.querySelector('.gate-card');
document.getElementById('gateSub').remove();         /* sub-title now comes from the pad (same position) */
buildPadCard(gateCard, false);
gateCard.insertBefore(gateCard.querySelector('.pin-badge'), gateCard.querySelector('.brand').nextSibling);
gateCard.insertBefore(gateCard.querySelector('.gate-title'), gateCard.querySelector('.pin-badge').nextSibling);
gateCard.insertBefore(gateCard.querySelector('.gate-sub'), gateCard.querySelector('.gate-title').nextSibling);
gateCard.insertBefore(gateCard.querySelector('.pin-dots'), gateCard.querySelector('.gate-sub').nextSibling);
gateCard.insertBefore(gateCard.querySelector('.gate-msg'), gateCard.querySelector('.pin-dots').nextSibling);
gateCard.insertBefore(gateCard.querySelector('.keypad'), gateCard.querySelector('.gate-msg').nextSibling);
['pinDots','gateMsg','keypad'].forEach(id => { const e = document.getElementById(id); if(e) e.remove(); });   /* old static copies */
let gatePad = null, gateRun = 0;
let gateKbBound = false;
function gateVisible(){ return gateEl.style.display === 'flex'; }
function hideGate(){
  gateEl.classList.add('fade-out');
  setTimeout(() => { gateEl.style.display = 'none'; gateEl.classList.remove('fade-out'); }, 260);
}
async function openGate(targetPage){
  pendingPageTarget = targetPage;
  const run = ++gateRun;
  if(!gatePad){ gatePad = new PinPad(gateCard); }
  if(!gateKbBound){ gateKbBound = true; gatePad.keyboard(() => gateVisible()); }
  gatePad.reset();
  gateEl.classList.remove('fade-out'); gateEl.style.display = 'flex';
  gatePad.texts(T('pin.dashTitle'), T('pin.dashSub', { page: T(targetPage === 'settings' ? 'pin.settingsName' : 'nav.' + targetPage).toUpperCase() }));
  if(!(PinStore.connected() && PinStore.loaded())){ gatePad.loading(T('pin.connecting')); await PinStore.load(true, 3500); gatePad.ready(); }
  if(run !== gateRun || !gateVisible()) return;
  const r = await runPinFlow(gatePad, 'dash', {
    title: T('pin.dashTitle'), sub: T('pin.dashSub', { page: T(targetPage === 'settings' ? 'pin.settingsName' : 'nav.' + targetPage).toUpperCase() }),
    createTitle: T('pin.dashCreateTitle'), createSub: T('pin.dashCreateSub') });
  if(run !== gateRun) return;
  isAdminUnlocked = true;                              /* stays unlocked for ALL admin tabs until the app is closed or "Lock Admin" is pressed */
  hideGate();
  setTimeout(() => window.enablePush && window.enablePush(), 600);   /* PWA: ask notification permission after admin login */
  if(r.created && !r.synced) toast(T('pin.savedOffline'), 'info');
  const tgt = pendingPageTarget; pendingPageTarget = null;
  if(tgt === 'settings') openSettings(); else if(tgt) navigateToPage(tgt);
}
function closeGate(){
  gateRun++;
  if(gatePad){ gatePad.onSubmit = null; gatePad.reset(); }
  gateEl.style.display = 'none'; gateEl.classList.remove('fade-out'); pendingPageTarget = null;
}
document.getElementById('gateCancelBtn').addEventListener('click', () => { if(gatePad && gatePad.busy && gatePad.card.classList.contains('st-ok')) return; closeGate(); });

/* Professional result popup (animated tick / cross) */
function showResult(type, title, msg, ms){
  ms = ms || 1900;
  const bg = document.createElement('div'); bg.className = 'rs-bg';
  bg.innerHTML = '<div class="rs-card ' + (type === 'ok' ? 'ok' : 'err') + '"><div class="rs-ico"><svg viewBox="0 0 24 24">'
    + (type === 'ok' ? '<path d="M5 12.5l4.5 4.5L19 7.5"/>' : '<path d="M6 6l12 12M18 6L6 18"/>')
    + '</svg></div><div class="rs-t"></div><div class="rs-m"></div><div class="rs-bar"><i></i></div></div>';
  bg.querySelector('.rs-t').textContent = title || ''; bg.querySelector('.rs-m').textContent = msg || '';
  bg.querySelector('.rs-bar i').style.setProperty('--rs-ms', ms + 'ms'); bg.style.setProperty('--rs-ms', ms + 'ms');
  document.body.appendChild(bg);
  requestAnimationFrame(() => requestAnimationFrame(() => bg.classList.add('show')));
  let gone = false;
  const done = () => { if(gone) return; gone = true; bg.classList.remove('show'); setTimeout(() => bg.remove(), 320); };
  bg.addEventListener('click', done); setTimeout(done, ms);
}

/* ===== SETTINGS (change App PIN / Dashboard PIN - stored in Firebase + local copy) ===== */
const EYE_ON = '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 24 24"><path d="M3 3l18 18M10.6 6.2A9.7 9.7 0 0 1 12 6c6.4 0 10 6 10 6a17 17 0 0 1-3.2 3.9M6.5 7.6A16.6 16.6 0 0 0 2 12s3.6 7 10 7a9.6 9.6 0 0 0 4-.9"/></svg>';
let settingsBg = null;
function pinField(label, id){
  return '<label class="st-f"><span>' + label + '</span><div class="st-in"><input type="password" inputmode="numeric" autocomplete="off" maxlength="12" id="' + id + '" placeholder="&bull;&bull;&bull;&bull;">'
    + '<button type="button" class="st-eye" tabindex="-1">' + EYE_ON + '</button></div></label>';
}
function buildSettingsSection(which, icon, title, sub){
  const sec = document.createElement('section'); sec.className = 'st-sec'; sec.dataset.which = which;
  const has = PinStore.has(which);
  sec.innerHTML = '<div class="st-sh"><div class="st-si">' + icon + '</div><div><div class="st-st">' + title + '</div><div class="st-ss">' + sub + '</div></div><span class="st-chip"></span></div>'
    + (has ? pinField(T('pin.currentPin'), 'stCur_' + which) : '')
    + pinField(T('pin.newPin'), 'stNew_' + which) + pinField(T('pin.confirmNewPin'), 'stCnf_' + which)
    + '<div class="st-msg"></div><button type="button" class="st-btn"><span class="st-bt">' + T(has ? 'pin.update' : 'pin.create') + '</span><i class="st-spin"></i></button>';
  sec.querySelectorAll('.st-in').forEach(w => {
    const inp = w.querySelector('input'), eye = w.querySelector('.st-eye');
    inp.addEventListener('input', () => { inp.value = inp.value.replace(/\D/g, '').slice(0, 12); sec.querySelector('.st-msg').textContent = ''; });
    eye.addEventListener('click', () => { const show = inp.type === 'password'; inp.type = show ? 'text' : 'password'; eye.innerHTML = show ? EYE_OFF : EYE_ON; });
  });
  sec.querySelector('.st-btn').addEventListener('click', () => submitSettingsPin(sec));
  return sec;
}
function setChip(sec){
  const chip = sec.querySelector('.st-chip'), on = PinStore.isOnline();
  chip.className = 'st-chip ' + (on === false ? 'off' : 'on');
  chip.textContent = T(on === false ? 'pin.chipDevice' : 'pin.chipSynced');
}
async function submitSettingsPin(sec){
  const which = sec.dataset.which, msg = sec.querySelector('.st-msg'), btn = sec.querySelector('.st-btn');
  if(btn.classList.contains('busy')) return;
  const cur = (sec.querySelector('#stCur_' + which) || {}).value || '', nw = sec.querySelector('#stNew_' + which).value, cf = sec.querySelector('#stCnf_' + which).value;
  const bad = text => {
    msg.textContent = text; sec.classList.remove('shake'); void sec.offsetWidth; sec.classList.add('shake');
    try{ navigator.vibrate && navigator.vibrate([50, 30, 70]); }catch(e){}
  };
  if(PinStore.has(which) && !cur) return bad(T('pin.enterCurrent'));
  if(!/^\d{4,12}$/.test(nw)) return bad(T('pin.lengthRule'));
  if(nw !== cf) return bad(T('pin.mismatch'));
  if(PinStore.has(which) && nw === PinStore.get(which)) return bad(T('pin.sameAsOld'));
  btn.classList.add('busy');
  let r; try{ r = await PinStore.change(which, cur, nw); }catch(e){ r = { ok: false, reason: 'error' }; }
  btn.classList.remove('busy');
  if(!r.ok){
    bad(r.reason === 'wrong' ? T('pin.currentWrong') : T('pin.failedTitle'));
    if(r.reason === 'wrong') showResult('err', T('pin.wrong'), T('pin.currentWrong'), 1800);
    return;
  }
  showResult('ok', T('pin.updatedTitle'), T(r.synced ? 'pin.updatedSynced' : 'pin.updatedLocal'), 2200);
  sec.querySelectorAll('input').forEach(i => { i.value = ''; });
  setChip(sec); sec.querySelector('.st-bt').textContent = T('pin.update');
  if(!sec.querySelector('#stCur_' + which)){                       /* first PIN just created -> from now on ask for the current PIN */
    sec.querySelector('.st-f').insertAdjacentHTML('beforebegin', pinField(T('pin.currentPin'), 'stCur_' + which));
    const w = sec.querySelector('#stCur_' + which).parentNode, inp = w.querySelector('input'), eye = w.querySelector('.st-eye');
    inp.addEventListener('input', () => { inp.value = inp.value.replace(/\D/g, '').slice(0, 12); msg.textContent = ''; });
    eye.addEventListener('click', () => { const show = inp.type === 'password'; inp.type = show ? 'text' : 'password'; eye.innerHTML = show ? EYE_OFF : EYE_ON; });
  }
}
function openSettings(){
  if(!settingsBg){
    settingsBg = document.createElement('div'); settingsBg.id = 'settingsBg'; settingsBg.className = 'st-bg';
    settingsBg.innerHTML = '<div class="st-card"><div class="st-head"><div class="st-hi"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg></div>'
      + '<div class="st-ht"><div class="st-title">' + T('pin.settingsTitle') + '</div><div class="st-sub">' + T('pin.settingsSub') + '</div></div><button type="button" class="st-x" aria-label="Close"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button></div><div class="st-body"></div></div>';
    document.body.appendChild(settingsBg);
    settingsBg.querySelector('.st-x').addEventListener('click', () => closeSettings());
    settingsBg.addEventListener('click', e => { if(e.target === settingsBg) closeSettings(); });
  }
  const body = settingsBg.querySelector('.st-body'); body.innerHTML = '';
  body.appendChild(buildSettingsSection('app', '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>', T('pin.secAppTitle'), T('pin.secAppSub')));
  body.appendChild(buildSettingsSection('dash', '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/></svg>', T('pin.secDashTitle'), T('pin.secDashSub')));
  body.querySelectorAll('.st-sec').forEach(setChip);
  PinStore.load().then(() => body.querySelectorAll('.st-sec').forEach(setChip));
  requestAnimationFrame(() => requestAnimationFrame(() => settingsBg.classList.add('show')));
  history.pushState({ page: currentPage, settings: true }, '', '#' + currentPage);
}
function closeSettings(fromPop){
  if(settingsBg) settingsBg.classList.remove('show');
  if(!fromPop && history.state && history.state.settings){ try{ history.back(); }catch(e){} }
}
(function mountSettingsButton(){
  const b = document.createElement('button'); b.id = 'settingsBtn'; b.type = 'button'; b.setAttribute('aria-label', 'Settings');
  b.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>';
  b.addEventListener('click', () => { if(isAdminUnlocked) openSettings(); else openGate('settings'); });
  document.body.appendChild(b);
})();

document.getElementById('logoutBtn').addEventListener('click', ()=>{
  isAdminUnlocked = false;
  toast(T('gate.locked'), 'info');
  if(currentPage !== 'scanner') navigateToPage('scanner');
});

/* CLOCK */
function tickClock(){
  const now = new Date();
  document.getElementById('sideClock').innerHTML = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) + 
    '<small>'+now.toLocaleDateString([], {weekday:'short', year:'numeric', month:'short', day:'numeric'})+'</small>';
}
setInterval(tickClock, 1000); tickClock();

/* NAV & TAB SWITCHING */
function navigateToPage(pageName, fromHistory){
  currentPage = pageName;
  
  if(pageName !== 'scanner') {
    stopScannerCamera();
  }

  document.querySelectorAll('.nav-item').forEach(i=> i.classList.toggle('active', i.dataset.page === pageName));
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  
  const targetPageEl = document.getElementById('page-'+pageName);
  if(targetPageEl) {
    targetPageEl.classList.add('active');
  }

  if(pageName==='dashboard') renderDashboard();
  if(pageName==='employees') renderEmployees();
  if(pageName==='reports') refreshReports();

  /* record this page in the browser history so the phone Back button returns to the previous page */
  if(!fromHistory && history.state && history.state.page !== pageName){
    history.pushState({page: pageName}, '', '#' + pageName);
  }
}

/* BACK BUTTON / HISTORY: Back goes to the previous page (or closes an open popup) and never closes the app */
(function initHistory(){
  try{
    history.replaceState({page: 'scanner', root: true}, '', location.pathname + location.search);
    history.pushState({page: 'scanner'}, '', '#scanner');
  }catch(e){ return; }
  window.addEventListener('popstate', function(e){
    const st = e.state || {};
    const keep = () => history.pushState({page: currentPage}, '', '#' + currentPage);   /* re-add the entry that Back just removed */
    const gate = document.getElementById('gate');
    if(gate && gate.style.display === 'flex'){ closeGate(); keep(); return; }
    const stBg = document.getElementById('settingsBg');
    if(stBg && stBg.classList.contains('show')){ closeSettings(true); return; }
    const modal = document.getElementById('empModalBg');
    if(modal && modal.classList.contains('show')){ closeEmpModal(); keep(); return; }
    if(st.root || !st.page){                      /* already at the first screen: stay inside the app */
      keep();
      if(currentPage !== 'scanner') navigateToPage('scanner', true);
      return;
    }
    let page = st.page;
    if(page !== 'scanner' && !isAdminUnlocked) page = 'scanner';
    navigateToPage(page, true);
  });
})();

/* CAMERA BUTTON: fixed size and layout (icon on the left, "Stop" / "Camera" on two lines) - never resizes */
(function lockCameraButton(){
  const st = document.createElement('style'); st.id = 'camBtnLockCss';
  st.textContent = `
  #camToggleBtn{width:84px !important;min-width:84px !important;max-width:84px !important;height:38px !important;min-height:38px !important;max-height:38px !important;
    flex:0 0 84px !important;box-sizing:border-box !important;padding:0 8px !important;display:flex !important;flex-direction:row !important;align-items:center !important;
    justify-content:flex-start !important;gap:6px !important;overflow:hidden !important}
  #camToggleBtn svg{flex:0 0 14px !important;width:14px !important;height:14px !important}
  #camToggleBtn span{display:block !important;width:min-content !important;white-space:normal !important;text-align:left !important;line-height:1.15 !important;font-size:.72rem !important}`;
  document.head.appendChild(st);
})();

document.querySelectorAll('.nav-item').forEach(item=>{
  item.addEventListener('click', ()=>{
    const target = item.dataset.page;
    if(target !== 'scanner' && !isAdminUnlocked){ openGate(target); return; }
    navigateToPage(target);
  });
});

/* FACE API LOADERS */
let faceApiReady = false;
let faceApiLoading = false;
const MODEL_MIRRORS = [
  'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model',
  'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights'
];

async function loadModelsFromUri(url){
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(url),
    faceapi.nets.faceLandmark68Net.loadFromUri(url),
    faceapi.nets.faceRecognitionNet.loadFromUri(url)
  ]);
}

async function loadFaceModels(silent){
  if(faceApiReady) return true;
  if(faceApiLoading) return false;
  faceApiLoading = true;
  setScanModelStatus(T('scanner.models.loading'), 'detecting');

  for(const mirror of MODEL_MIRRORS){
    try{
      await loadModelsFromUri(mirror);
      faceApiReady = true; faceApiLoading = false;
      setScanModelStatus(T('scanner.models.loaded'), 'success');
      return true;
    }catch(e){ console.warn('Mirror failed', mirror); }
  }

  faceApiLoading = false; faceApiReady = false;
  setScanModelStatus(T('scanner.models.failed'), 'fail');
  if(!silent) toast(T('scanner.models.failedToast'), 'err');
  return false;
}

function setScanModelStatus(text, cls){
  const el = document.getElementById('modelStatus');
  if(!el) return;
  el.textContent = text; el.className = 'scan-status ' + (cls||'');
  document.getElementById('retryModelsBtn').style.display = cls==='fail' ? 'inline-block' : 'none';
}
document.getElementById('retryModelsBtn').addEventListener('click', ()=>{ faceApiReady=false; loadFaceModels(false); });

/* DASHBOARD */
function renderDashboard(){
  document.getElementById('statTotalEmp').textContent = employees.length;
  const todays = logs.filter(l=>l.date===todayStr());
  document.getElementById('statInToday').textContent = todays.filter(l=>l.entrance).length;
  document.getElementById('statOutToday').textContent = todays.filter(l=>l.exit).length;
  document.getElementById('statFaceReg').textContent = employees.filter(e=>e.descriptor).length;

  const recent = logs.filter(l=>l.date===todayStr()).sort((a,b)=>(b.lastUpdated||0)-(a.lastUpdated||0)).slice(0,8);
  const body = document.getElementById('dashRecentBody');
  body.innerHTML = recent.length ? recent.map(r=>{
    const status = r.exit ? 'OUT' : 'IN';
    const time = r.exit || r.entrance;
    return `<tr><td><b>${escapeHtml(r.name)}</b></td><td>${escapeHtml(r.empId)}</td><td>${r.date}</td><td>${time}</td>
      <td><span class="badge ${status==='IN'?'in':'out'}">${T('status.' + status)}</span></td></tr>`;
  }).join('') : `<tr><td colspan="5" class="empty">${T('dashboard.noActivity')}</td></tr>`;
}

/* EMPLOYEES */
function renderEmployees(){
  const q = (document.getElementById('empSearch').value || '').trim().replace(/\s+/g,' ').toLowerCase();
  const grid = document.getElementById('empGrid');
  const filtered = employees.filter(e => e.name.toLowerCase().includes(q) || e.id.toLowerCase().includes(q));
  grid.innerHTML = filtered.length ? filtered.map(e=>`
    <div class="emp-card">
      ${e.photo ? `<img class="avatar-lg" src="${e.photo}">` : `<div class="avatar-ph-lg">${initials(e.name)}</div>`}
      <div class="ename">${escapeHtml(e.name)}</div>
      <div class="eid">${escapeHtml(e.id)} · ${escapeHtml(e.dept||'—')}</div>
      <span class="face-tag ${e.descriptor?'yes':'no'}">${e.descriptor ? T('common.enrolled') : T('employees.noFaceData')}</span>
      <div class="emp-actions">
        <button data-edit="${escapeHtml(e.id)}">${T('common.edit')}</button>
        <button class="del" data-del="${escapeHtml(e.id)}">${T('common.delete')}</button>
      </div>
    </div>
  `).join('') : `<div class="empty">${T('employees.noneFound')}</div>`;

  grid.querySelectorAll('[data-del]').forEach(b=>b.addEventListener('click', async ()=>{
    const id = b.dataset.del, de = employees.find(e=>e.id === id);
    if(!id || !de) return;
    if(await askConfirm(T('employees.deleteTitle'), T('employees.deleteMessage', {name: de.name}), T('common.delete'))){
      employees = employees.filter(e=>e.id !== id);                 /* exactly this id */
      renderEmployees(); renderDashboard();
      fbDeleteEmployee(id).then(() => showResult('ok', T('pin.deletedTitle'), T('employees.deleted')))
        .catch(err => showResult('err', T('pin.failedTitle'), T('employees.deleteFailed', {error: err.message}), 2600));
    }
  }));
  grid.querySelectorAll('[data-edit]').forEach(b=>b.addEventListener('click', ()=> openEmpModal(b.dataset.edit)));
}
document.getElementById('empSearch').addEventListener('input', renderEmployees);

/* EMPLOYEE DATA JSON IMPORT & EXPORT */
document.getElementById('exportJsonBtn').addEventListener('click', async ()=>{
  const byId = new Map(employees.map(e => [e.id, e]));              /* what the app shows now */
  try{ Object.values((await fdb.ref('employees').once('value')).val() || {}).forEach(e => byId.set(e.id, { ...byId.get(e.id), ...e })); }catch(e){}   /* + Firebase copy */
  const toData = u => fetch(u).then(r => r.blob()).then(b => new Promise(ok => { const f = new FileReader(); f.onload = () => ok(f.result); f.readAsDataURL(b); })).catch(() => u);
  const cleanEmployees = await Promise.all([...byId.values()].map(async ({ photoUrl, ...e }) => (e.photo && !/^data:/.test(e.photo)) ? { ...e, photo: await toData(e.photo) } : e));
  if(!cleanEmployees.length){ toast(T('employees.exportNone'), 'err'); return; }
  const jsonStr = JSON.stringify(cleanEmployees, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = T('files.employeesBackup', {date: todayStr()});
  a.click();
  toast(T('employees.exported'), 'ok');
});

document.getElementById('importJsonBtn').addEventListener('click', ()=>{
  document.getElementById('importJsonInput').click();
});

document.getElementById('importJsonInput').addEventListener('change', (e)=>{
  const file = e.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = (evt)=>{
    try{
      let imported = JSON.parse(evt.target.result);
      if(imported && !Array.isArray(imported)) imported = Array.isArray(imported.employees) ? imported.employees : Object.values(imported);
      if(!Array.isArray(imported)){ toast(T('employees.invalidJson'), 'err'); return; }
      let addedCount = 0, updatedCount = 0, skipped = 0;
      const seen = new Set(), batch = [];
      imported.forEach(imp=>{
        const id = imp && (imp.id ?? imp.emp_id ?? imp.employee_id) != null ? String(imp.id ?? imp.emp_id ?? imp.employee_id).trim() : '';
        if(!id || seen.has(id)){ skipped++; return; }
        seen.add(id);
        const { emp_id, employee_id, photoUrl, ...rest } = imp;
        const cur = employees.find(e=>e.id === id);
        if(cur) updatedCount++; else addedCount++;
        batch.push({ ...(cur || {}), ...rest, id, name: String(rest.name || (cur && cur.name) || id) });
      });
      if(!batch.length){ toast(T('employees.noValidRecords'), 'err'); e.target.value = ''; return; }
      batch.forEach(r => { const i = employees.findIndex(x => x.id === r.id); if(i >= 0) employees[i] = r; else employees.push(r); });
      renderEmployees(); renderDashboard();                      /* instant UI; Firebase listener takes over */
      saveEmployees(batch).then(ok => {
        if(!ok) return;
        toast(T('employees.importDone', {added: addedCount, updated: updatedCount}) + (skipped ? T('employees.importSkipped', {skipped: skipped}) : ''), 'ok');
        const p = window.faceBackendImport && window.faceBackendImport(batch);      /* server: store face embeddings from photos */
        if(p) p.then(r => { if(r) toast(T('employees.faceServer', {enrolled: r.enrolled}) + (r.failed.length ? T('employees.faceServerFailed', {failed: r.failed.length}) : ''), r.failed.length ? 'err' : 'ok'); }).catch(()=>{});
      });
    }catch(err){
      toast(T('employees.parseFailed'), 'err');
    }
    e.target.value = '';
  };
  reader.readAsText(file);
});

/* ADD / EDIT MODAL */
const empModalBg = document.getElementById('empModalBg');
let editingId = null, regStream = null, capturedDescriptor = null, capturedPhoto = null, capturedFpEnrolled = false, fpCredentialId = null;
let currentEnrollSource = 'cam';

document.getElementById('addEmpBtn').addEventListener('click', ()=> openEmpModal(null));
document.getElementById('empCancelBtn').addEventListener('click', closeEmpModal);

function updateEnrollBadges(){
  const faceBadge = document.getElementById('faceEnrollBadge');
  faceBadge.textContent = capturedDescriptor ? T('common.enrolled') : T('common.notEnrolled');
  faceBadge.classList.toggle('on', !!capturedDescriptor);

  const fpBadge = document.getElementById('fpEnrollBadge');
  const fpIcon = document.getElementById('fpEnrollIcon');
  fpBadge.textContent = capturedFpEnrolled ? T('common.enrolled') : T('common.notEnrolled');
  fpBadge.classList.toggle('on', capturedFpEnrolled);
  fpIcon.classList.toggle('done', capturedFpEnrolled);
}

function openEmpModal(id){
  editingId = id; capturedDescriptor = null; capturedPhoto = null; capturedFpEnrolled = false; fpCredentialId = null;
  currentEnrollSource = 'cam';
  document.getElementById('galleryPreview').style.display = 'none';
  document.getElementById('regVideo').style.display = 'block';
  document.getElementById('regFaceStatus').textContent = '';
  document.getElementById('regFaceStatus').className = 'enroll-status';
  document.getElementById('regFpStatus').textContent = '';
  document.getElementById('regFpStatus').className = 'enroll-status';
  document.getElementById('captureFaceBtn').disabled = true;
  if(id){
    const e = employees.find(x=>x.id===id);
    document.getElementById('empModalTitle').textContent = T('modal.editTitle');
    document.getElementById('fEmpId').value = e.id; document.getElementById('fEmpId').disabled = true;
    document.getElementById('fEmpName').value = e.name;
    document.getElementById('fEmpDept').value = e.dept || '';
    document.getElementById('fEmpPin').value = e.pin || '';
    document.getElementById('fEmpFpTag').value = e.fpTag || '';
    capturedDescriptor = e.descriptor || null; capturedPhoto = e.photo || null;
    capturedFpEnrolled = !!e.fpEnrolled; fpCredentialId = e.fpCredentialId || null;
    if(e.photo){
      document.getElementById('galleryPreview').src = e.photo;
      document.getElementById('galleryPreview').style.display = 'block';
      document.getElementById('regVideo').style.display = 'none';
    }
  } else {
    document.getElementById('empModalTitle').textContent = T('modal.addTitle');
    document.getElementById('fEmpId').disabled = false; document.getElementById('fEmpId').value = '';
    document.getElementById('fEmpName').value = ''; document.getElementById('fEmpDept').value = '';
    document.getElementById('fEmpPin').value = ''; document.getElementById('fEmpFpTag').value = '';
  }
  updateEnrollBadges();
  empModalBg.classList.add('show');
}
function closeEmpModal(){ empModalBg.classList.remove('show'); stopRegCamera(); }

document.getElementById('regCamBtn').addEventListener('click', async ()=>{
  stopRegCamera();
  currentEnrollSource = 'cam';
  document.getElementById('galleryPreview').style.display = 'none';
  document.getElementById('regVideo').style.display = 'block';
  try{
    regStream = await navigator.mediaDevices.getUserMedia({video:{width:320,height:240}});
    document.getElementById('regVideo').srcObject = regStream;
    document.getElementById('captureFaceBtn').disabled = false;
  }catch(e){ toast(T('modal.cameraUnavailable'), 'err'); }
});
function stopRegCamera(){ if(regStream){ regStream.getTracks().forEach(t=>t.stop()); regStream=null; } }

/* CHOOSE FROM GALLERY PHOTO UPLOAD */
document.getElementById('chooseGalleryBtn').addEventListener('click', ()=>{
  document.getElementById('galleryInput').click();
});

document.getElementById('galleryInput').addEventListener('change', (e)=>{
  const file = e.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = (evt)=>{
    stopRegCamera();
    currentEnrollSource = 'gallery';
    const preview = document.getElementById('galleryPreview');
    preview.src = evt.target.result;
    preview.style.display = 'block';
    document.getElementById('regVideo').style.display = 'none';
    document.getElementById('captureFaceBtn').disabled = false;
    document.getElementById('regFaceStatus').textContent = T('modal.photoSelected');
    document.getElementById('regFaceStatus').className = 'enroll-status';
  };
  reader.readAsDataURL(file);
  e.target.value = '';
});

document.getElementById('captureFaceBtn').addEventListener('click', async ()=>{
  const video = document.getElementById('regVideo');
  const preview = document.getElementById('galleryPreview');
  const statusEl = document.getElementById('regFaceStatus');

  let inputSource = currentEnrollSource === 'gallery' ? preview : video;

  if(!faceApiReady){
    capturedPhoto = currentEnrollSource === 'gallery' ? preview.src : snapshotToDataUrl(video);
    statusEl.textContent = T('modal.photoOnly');
    updateEnrollBadges();
    return;
  }
  statusEl.textContent = T('modal.extracting');
  try{
    const det = await faceapi.detectSingleFace(inputSource, new faceapi.TinyFaceDetectorOptions({ inputSize: 128, scoreThreshold: 0.4 }))
      .withFaceLandmarks().withFaceDescriptor();

    if(!det){ 
      statusEl.textContent = T('modal.noFaceInPhoto'); 
      statusEl.className = 'enroll-status err'; 
      return; 
    }

    capturedDescriptor = Array.from(det.descriptor);
    capturedPhoto = currentEnrollSource === 'gallery' ? preview.src : snapshotToDataUrl(video);
    statusEl.textContent = T('modal.faceCaptured'); 
    statusEl.className = 'enroll-status ok';
    updateEnrollBadges();
  }catch(e){
    statusEl.textContent = T('modal.faceExtractFailed');
    statusEl.className = 'enroll-status err';
  }
});

function snapshotToDataUrl(video){
  /* centre-crop to a square (no squashing) so the stored profile photo keeps real face proportions */
  const vw = video.videoWidth || video.naturalWidth || 256, vh = video.videoHeight || video.naturalHeight || 256;
  const sz = Math.min(vw, vh), sx = (vw - sz) / 2, sy = (vh - sz) / 2;
  const c = document.createElement('canvas'); c.width = 256; c.height = 256;
  c.getContext('2d').drawImage(video, sx, sy, sz, sz, 0, 0, 256, 256);
  return c.toDataURL('image/jpeg', 0.85);
}

/* WebAuthn helpers */
function bufToB64(buf){
  return btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');
}
function b64ToBuf(b64){
  const pad = '='.repeat((4 - (b64.length % 4)) % 4);
  const s = (b64 + pad).replace(/-/g,'+').replace(/_/g,'/');
  const raw = atob(s);
  const buf = new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++) buf[i] = raw.charCodeAt(i);
  return buf.buffer;
}
async function platformAuthenticatorAvailable(){
  if(!window.PublicKeyCredential || !PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable) return false;
  try{ return await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable(); }catch(e){ return false; }
}

document.getElementById('captureFpBtn').addEventListener('click', async ()=>{
  const statusEl = document.getElementById('regFpStatus');
  const hintEl = document.getElementById('fpEnrollHint');
  const tag = document.getElementById('fEmpFpTag').value.trim();
  const empId = document.getElementById('fEmpId').value.trim();
  if(!empId){ statusEl.textContent = T('modal.enterIdFirst'); statusEl.className='enroll-status err'; return; }
  if(!tag){ statusEl.textContent = T('modal.enterTagFirst'); statusEl.className='enroll-status err'; return; }
  if(employees.some(e => e.fpTag === tag && e.id !== editingId)){
    statusEl.textContent = T('modal.tagUsed'); statusEl.className='enroll-status err'; return;
  }
  if(!navigator.credentials || !window.PublicKeyCredential || !window.isSecureContext){
    statusEl.textContent = T('modal.webauthnUnsupported'); statusEl.className='enroll-status err'; return;
  }
  if(!(await platformAuthenticatorAvailable())){
    statusEl.textContent = T('modal.noSensor'); statusEl.className='enroll-status err'; return;
  }

  hintEl.textContent = T('modal.followPromptEnroll');
  statusEl.textContent = T('scanner.fp.waiting'); statusEl.className='enroll-status';
  try{
    const challenge = new Uint8Array(32); crypto.getRandomValues(challenge);
    const userId = new Uint8Array(16); crypto.getRandomValues(userId);
    const cred = await navigator.credentials.create({
      publicKey: {
        challenge,
        rp: { name: T('modal.relyingPartyName'), id: location.hostname || undefined },
        user: { id: userId, name: tag, displayName: document.getElementById('fEmpName').value.trim() || tag },
        pubKeyCredParams: [ {type:'public-key', alg:-7}, {type:'public-key', alg:-257} ],
        authenticatorSelection: {
          authenticatorAttachment: 'platform',
          userVerification: 'required',
          residentKey: 'preferred'
        },
        timeout: 30000,
        attestation: 'none'
      }
    });
    capturedFpEnrolled = true;
    fpCredentialId = bufToB64(cred.rawId);
    hintEl.textContent = T('modal.fpEnrolledHint');
    statusEl.textContent = T('modal.fpCaptured'); statusEl.className='enroll-status ok';
    updateEnrollBadges();
  }catch(err){
    hintEl.textContent = T('modal.fpHintReset');
    statusEl.textContent = T('modal.fpEnrollFailed'); statusEl.className='enroll-status err';
  }
});

document.getElementById('empSaveBtn').addEventListener('click', ()=>{
  const id = document.getElementById('fEmpId').value.trim();
  const name = document.getElementById('fEmpName').value.trim();
  const dept = document.getElementById('fEmpDept').value.trim();
  const pin = document.getElementById('fEmpPin').value.trim();
  const fpTag = document.getElementById('fEmpFpTag').value.trim();
  if(!id || !name){ toast(T('modal.idNameRequired'), 'err'); return; }
  if(pin && !/^\d{4}$/.test(pin)){ toast(T('toast.pinFormat'), 'err'); return; }

  const record = { id, name, dept, pin, fpTag, fpEnrolled: fpTag ? capturedFpEnrolled : false, fpCredentialId: fpTag ? fpCredentialId : null, descriptor: capturedDescriptor, photo: capturedPhoto };
  let merged = record;
  if(editingId){
    if(id !== editingId){ toast(T('modal.idLocked'), 'err'); return; }
    const cur = employees.find(e=>e.id===editingId);
    merged = cur ? {...cur, ...record} : record;
    const i = employees.findIndex(e=>e.id===editingId);
    if(i >= 0) employees[i] = merged; else employees.push(merged);
  } else {
    if(employees.some(e=>e.id===id)){ toast(T('modal.idExists'), 'err'); return; }
    employees.push(merged);
  }
  closeEmpModal(); renderEmployees(); renderDashboard();
  saveEmployees([merged]).then(ok => toast(ok ? T('modal.saved') : T('modal.saveNotSynced'), ok ? 'ok' : 'err'));
});

/* LIVE SCANNER & LIVENESS */
let scanStream = null, isProcessingFrame = false, animationFrameId = null;
let scanMode = localStorage.getItem(LS_MODE) || 'IN';
let isVerifyingLiveness = false;

/* FLASHLIGHT / LOW-LIGHT DETECTION LOGIC */
let flashCheckCanvas = document.createElement('canvas');
let flashCheckCtx = flashCheckCanvas.getContext('2d', { willReadFrequently: true });

let lastAvgBrightness = 255, lowBrightnessLatched = false, brightnessTs = 0;
const LOWLIGHT_ON_BELOW = 50;        /* switch screen-light ON below this  */
const LOWLIGHT_OFF_ABOVE = 80;       /* switch OFF only above this (hysteresis: no flicker) */
const LOWLIGHT_ENHANCE_BELOW = 95;   /* digitally brighten frames for the detector below this */

function isNightOrLowLight(videoEl) {
  const hour = new Date().getHours();
  const isNightTime = (hour >= 19 || hour < 6);

  if (videoEl && videoEl.videoWidth > 0) {
    const t = performance.now();
    if (t - brightnessTs > 500) {           /* measure at most twice per second */
      brightnessTs = t;
      flashCheckCanvas.width = 64;
      flashCheckCanvas.height = 48;
      flashCheckCtx.drawImage(videoEl, 0, 0, 64, 48);
      const d = flashCheckCtx.getImageData(0, 0, 64, 48).data;
      let total = 0;
      for (let i = 0; i < d.length; i += 4) total += (d[i] + d[i + 1] + d[i + 2]) / 3;
      lastAvgBrightness = total / (d.length / 4);
      if (!lowBrightnessLatched && lastAvgBrightness < LOWLIGHT_ON_BELOW) lowBrightnessLatched = true;
      else if (lowBrightnessLatched && lastAvgBrightness > LOWLIGHT_OFF_ABOVE) lowBrightnessLatched = false;
    }
  }
  return isNightTime || lowBrightnessLatched;
}

/* Gamma-brightened copy of the frame for face detection in dim rooms (reused canvas) */
const enhCanvas = document.createElement('canvas');
const enhCtx = enhCanvas.getContext('2d', { willReadFrequently: true });
const GAMMA_LUT = (() => { const l = new Uint8ClampedArray(256); for (let i = 0; i < 256; i++) l[i] = Math.pow(i / 255, 0.55) * 255 * 1.08; return l; })();
function getDetectionInput(video) {
  if (lastAvgBrightness >= LOWLIGHT_ENHANCE_BELOW) return video;
  const w = video.videoWidth, h = video.videoHeight;
  if (enhCanvas.width !== w) enhCanvas.width = w;
  if (enhCanvas.height !== h) enhCanvas.height = h;
  enhCtx.drawImage(video, 0, 0, w, h);
  const img = enhCtx.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) { d[i] = GAMMA_LUT[d[i]]; d[i + 1] = GAMMA_LUT[d[i + 1]]; d[i + 2] = GAMMA_LUT[d[i + 2]]; }
  enhCtx.putImageData(img, 0, 0);
  return enhCanvas;
}

/* Cached detector options (created once, bigger input + lower threshold in low light) */
const _optsCache = {};
function getOpts(size, thr) {
  const k = size + '_' + thr;
  return _optsCache[k] || (_optsCache[k] = new faceapi.TinyFaceDetectorOptions({ inputSize: size, scoreThreshold: thr }));
}
function isDimNow() { return lastAvgBrightness < LOWLIGHT_ENHANCE_BELOW; }
function getScanOpts() { return isDimNow() ? getOpts(224, 0.3) : getOpts(128, 0.4); }

/* Cached FaceMatcher (rebuilt only when enrolled descriptors change) */
let cachedMatcher = null, cachedMatcherSig = '';
function getMatcher(enrolled) {
  const sig = enrolled.map(e => e.id + ':' + e.descriptor.length + ':' + e.descriptor[0] + ':' + e.descriptor[63]).join('|');
  if (!cachedMatcher || sig !== cachedMatcherSig) {
    const labeled = enrolled.map(e => new faceapi.LabeledFaceDescriptors(e.id, [new Float32Array(e.descriptor)]));
    cachedMatcher = new faceapi.FaceMatcher(labeled, FACE_MATCH_THRESHOLD);
    cachedMatcherSig = sig;
  }
  return cachedMatcher;
}

/* ---------- TORCH (camera flashlight) CONTROL ---------- */
/* Flash mode persisted in localStorage: 'on' | 'off' | 'auto' */
const FLASH_MODE_KEY = 'flashMode';
function getFlashMode() {
  try { const v = localStorage.getItem(FLASH_MODE_KEY); if (v === 'on' || v === 'off' || v === 'auto') return v; } catch (e) {}
  return 'auto';
}
function saveFlashMode(m) { try { localStorage.setItem(FLASH_MODE_KEY, m); } catch (e) {} }
function flashModeToOverride(m) { return m === 'on' ? true : (m === 'off' ? false : null); }
let torchOverride = flashModeToOverride(getFlashMode());   /* null = AUTO (night/low-light), true/false = forced ON/OFF */
let torchApplied = false;        /* real hardware torch state */
let torchAutoLatched = false;    /* auto turned hardware torch on (stay on: torch itself brightens the frame) */
let torchBusy = false;
let torchUnsupported = false;    /* set once hardware torch proved unavailable -> screen light fallback only */

function getScanTrack() {
  try { return (scanStream && scanStream.getVideoTracks && scanStream.getVideoTracks()[0]) || null; } catch (e) { return null; }
}
function torchSupported() {
  try {
    const t = getScanTrack();
    if (!t || t.readyState !== 'live' || typeof t.getCapabilities !== 'function') return false;
    const caps = t.getCapabilities();
    return !!(caps && caps.torch);
  } catch (e) { return false; }
}
async function applyTorch(state) {
  if (torchBusy) return false;
  const t = getScanTrack();
  if (!t || torchUnsupported) return false;
  if (!torchSupported()) { torchUnsupported = true; return false; }
  torchBusy = true;
  try {
    await t.applyConstraints({ advanced: [{ torch: !!state }] });
    torchApplied = !!state;
    return true;
  } catch (e) {
    torchUnsupported = true;
    return false;
  } finally { torchBusy = false; }
}
function releaseTorchState() {
  torchOverride = flashModeToOverride(getFlashMode()); torchApplied = false; torchAutoLatched = false; torchBusy = false; torchUnsupported = false;
}

function updateFlashlightState(videoEl) {
  const flashEl = document.getElementById('screenFlashlight');
  const bulbBtn = document.getElementById('bulbToggleBtn');
  if (!flashEl) return;

  /* always measure brightness while camera is live */
  const isNight = !!(scanStream && isNightOrLowLight(videoEl));

  let want;
  if (torchOverride !== null) want = torchOverride;                 /* manual override wins, day or night */
  else if (torchApplied && torchAutoLatched) want = true;           /* hold auto torch (avoid ON/OFF flicker) */
  else want = isNight;

  /* hardware torch (only when supported); screen light is the fallback */
  if (scanStream && !torchUnsupported && want !== torchApplied) {
    const auto = torchOverride === null;
    applyTorch(want).then(ok => {
      if (ok) torchAutoLatched = auto && want;
      else updateFlashlightState(videoEl);
    }).catch(() => {});
  }
  if (!scanStream && torchApplied) { torchApplied = false; torchAutoLatched = false; }

  if (want) {
    flashEl.style.display = 'block';
    document.body.classList.add('white-mode');
  } else {
    flashEl.style.display = 'none';
    document.body.classList.remove('white-mode');
  }
  if (bulbBtn) {
    bulbBtn.classList.toggle('active', !!want);
    bulbBtn.dataset.mode = getFlashMode();
    bulbBtn.title = T('scanner.flashTitle', {mode: getFlashMode().toUpperCase()});
  }
}

function setFlashMode(m) {
  if (m !== 'on' && m !== 'off' && m !== 'auto') m = 'auto';
  saveFlashMode(m);
  torchOverride = flashModeToOverride(m);
  torchAutoLatched = false;
  manualFlashlightState = (m === 'on');
  updateFlashlightState(document.getElementById('video'));
}

/* long-press mode popup */
const flashPop = document.createElement('div');
flashPop.id = 'flashModePop';
flashPop.innerHTML = '<button data-m="on">' + T('scanner.flashOn') + '</button><button data-m="off">' + T('scanner.flashOff') + '</button><button data-m="auto">' + T('scanner.flashAuto') + '</button>';
document.body.appendChild(flashPop);
function closeFlashPop() { flashPop.classList.remove('open'); }
function openFlashPop() {
  const btn = document.getElementById('bulbToggleBtn');
  const r = btn.getBoundingClientRect();
  const cur = getFlashMode();
  flashPop.querySelectorAll('button').forEach(b => b.classList.toggle('sel', b.dataset.m === cur));
  flashPop.classList.add('open');
  const half = flashPop.offsetWidth / 2;
  const cx = Math.min(Math.max(r.left + r.width / 2, half + 8), window.innerWidth - half - 8);
  flashPop.style.left = cx + 'px';
  flashPop.style.bottom = (window.innerHeight - r.top + 10) + 'px';
  flashPop.style.top = 'auto';
}
flashPop.addEventListener('click', (e) => {
  const b = e.target.closest('button[data-m]');
  if (!b) return;
  setFlashMode(b.dataset.m);
  closeFlashPop();
});
document.addEventListener('pointerdown', (e) => {
  if (flashPop.classList.contains('open') && !flashPop.contains(e.target) && e.target.id !== 'bulbToggleBtn' && !e.target.closest('#bulbToggleBtn')) closeFlashPop();
}, true);

(function initBulbGestures() {
  const btn = document.getElementById('bulbToggleBtn');
  const LONG_MS = 550;
  let timer = null, longFired = false, startX = 0, startY = 0;
  const clear = () => { if (timer) { clearTimeout(timer); timer = null; } };

  btn.addEventListener('pointerdown', (e) => {
    longFired = false;
    startX = e.clientX; startY = e.clientY;
    clear();
    timer = setTimeout(() => { timer = null; longFired = true; if (navigator.vibrate) { try { navigator.vibrate(30); } catch (er) {} } openFlashPop(); }, LONG_MS);
  });
  btn.addEventListener('pointermove', (e) => {
    if (timer && Math.hypot(e.clientX - startX, e.clientY - startY) > 12) clear();
  });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach(ev => btn.addEventListener(ev, clear));
  btn.addEventListener('contextmenu', (e) => e.preventDefault());

  /* single tap = instant manual toggle ON/OFF */
  btn.addEventListener('click', (e) => {
    if (longFired) { longFired = false; e.preventDefault(); return; }
    if (flashPop.classList.contains('open')) { closeFlashPop(); return; }
    if (!scanStream) { toast(T('scanner.startCameraFirst'), 'info'); return; }
    const isOn = btn.classList.contains('active');
    setFlashMode(isOn ? 'off' : 'on');
  });
})();

function setScanMode(m){
  scanMode = m; localStorage.setItem(LS_MODE, m);
  ['modeIn','modeOut','modeAuto'].forEach(id=>document.getElementById(id).classList.remove('active'));
  document.getElementById('mode'+(m==='IN'?'In':m==='OUT'?'Out':'Auto')).classList.add('active');
}
document.getElementById('modeIn').addEventListener('click', ()=>setScanMode('IN'));
document.getElementById('modeOut').addEventListener('click', ()=>setScanMode('OUT'));
document.getElementById('modeAuto').addEventListener('click', ()=>setScanMode('AUTO'));

const camToggleBtn = document.getElementById('camToggleBtn');

camToggleBtn.addEventListener('click', async ()=>{
  if(scanStream){ stopScannerCamera(); return; }
  try{
    scanStream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:"user"}});
    const video = document.getElementById('video');
    video.srcObject = scanStream; video.style.display = 'block';
    document.getElementById('camPlaceholder').style.display = 'none';
    
    camToggleBtn.classList.add('active');
    camToggleBtn.querySelector('span').textContent = T('scanner.stopCamera');
    document.getElementById('camBox').classList.add('active-scan');

    updateFlashlightState(video);

    if(!faceApiReady) await loadFaceModels(true);
    if(faceApiReady){ setScanStatus(T('scanner.status.scanning'), 'detecting'); startScanLoop(); }
    else setScanStatus(T('scanner.status.modelsUnavailable'), 'fail');
  }catch(e){ toast(T('scanner.cameraBlocked'), 'err'); }
});

let scanGapMs = 100;   /* adaptive delay between scans: longer when no face / unknown face */
function startScanLoop(){
  if(animationFrameId) cancelAnimationFrame(animationFrameId);
  let lastRun = 0;
  const loop = async (ts) => {
    if(!scanStream) return;
    /* skip work while tab is hidden, busy, or inside the throttle window */
    if(!document.hidden && !isProcessingFrame && !isVerifyingLiveness && (ts - lastRun) >= scanGapMs){
      await runFaceScan();
      lastRun = performance.now();
    }
    animationFrameId = requestAnimationFrame(loop);
  };
  animationFrameId = requestAnimationFrame(loop);
}

function stopScannerCamera(){
  if(animationFrameId){ cancelAnimationFrame(animationFrameId); animationFrameId=null; }
  if(scanStream){ scanStream.getTracks().forEach(t=>t.stop()); scanStream=null; }
  const video = document.getElementById('video');
  video.style.display='none'; video.srcObject=null;
  document.getElementById('camPlaceholder').style.display='block';
  
  camToggleBtn.classList.remove('active');
  camToggleBtn.querySelector('span').textContent = T('scanner.startCamera');
  
  document.getElementById('camBox').classList.remove('active-scan');
  document.getElementById('camGuide').classList.remove('active');
  setScanStatus(T('scanner.status.idle'),'');
  
  manualFlashlightState = false; releaseTorchState();
  lowBrightnessLatched = false; lastAvgBrightness = 255; brightnessTs = 0;
  const bulbBtn = document.getElementById('bulbToggleBtn');
  if(bulbBtn) bulbBtn.classList.remove('active');
  updateFlashlightState(null);

  const overlay = document.getElementById('overlay');
  if(overlay && overlay.getContext){
    overlay.getContext('2d').clearRect(0,0,overlay.width||640,overlay.height||480);
  }
  isProcessingFrame = false;
  isVerifyingLiveness = false;
}

/* SCAN STATUS DISPLAYED STRICTLY AT BOTTOM-LEFT */
/* The bottom-left status chip shows ONLY the messages listed in text.json -> scanner.statusAllowed.
   Every other message is hidden here (those still appear in the top toasts / HUD). */
function setScanStatus(text, cls){
  const el = document.getElementById('scanStatus');
  if(!el) return;
  const norm = s => String(s).replace(/^step\s*\d\s*\/\s*\d\s*[\u00b7:\-\u2013]\s*/i, '').toLowerCase().replace(/[\s.\u2026]+$/, '');
  const allowed = T('scanner.statusAllowed');
  const t = norm(text);
  let shown = null;
  const fm = t.match(/^match the gesture \((\d)\s*finger/);        /* finger challenge (1-4) -> "Show N finger(s)" */
  const aliases = T('scanner.statusAliases');                       /* server wording -> text shown in the chip */
  if(fm){ shown = T(fm[1] === '1' ? 'scanner.status.showOne' : 'scanner.status.showMany', {n: fm[1]}); }
  else if(aliases && typeof aliases === 'object' && Object.keys(aliases).some(k => t === norm(k) || t.startsWith(norm(k) + ' '))){
    const k = Object.keys(aliases).find(k => t === norm(k) || t.startsWith(norm(k) + ' '));
    shown = aliases[k];
  }
  else if(Array.isArray(allowed)){
    for(const a of allowed){
      const n = norm(a);
      if(t === n || t.startsWith(n + ' ')){ shown = a; break; }
    }
  }
  if(shown === null){ el.style.display = 'none'; return; }
  el.style.display = '';
  el.textContent = shown;
  el.className = 'scan-status ' + (cls||'');
}

/* LIVENESS HELPERS */
let livenessCanvas = document.createElement('canvas');
let livenessCtx = livenessCanvas.getContext('2d', { willReadFrequently: true });

/* MICRO-MOTION LIVENESS CHECK: real faces show tiny natural motion/texture change
   between frames; a flat printed photo or a static/looping video screen tends to
   show near-zero pixel variance across a short interval. It runs right after the
   face is matched and must also pass before verification completes. */
function getFaceFrameSnapshot(video, box) {
  let sx = Math.max(0, Math.floor(box.x));
  let sy = Math.max(0, Math.floor(box.y));
  let sw = Math.max(1, Math.min(video.videoWidth - sx, Math.floor(box.width)));
  let sh = Math.max(1, Math.min(video.videoHeight - sy, Math.floor(box.height)));
  if (livenessCanvas.width !== sw) livenessCanvas.width = sw;
  if (livenessCanvas.height !== sh) livenessCanvas.height = sh;
  livenessCtx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);   /* draw only the face crop, not the full frame */
  return livenessCtx.getImageData(0, 0, sw, sh).data;
}

async function verifyMicroMotionLiveness(video, box) {
  try {
    const frame1 = getFaceFrameSnapshot(video, box);
    await new Promise(r => setTimeout(r, 140));
    const frame2 = getFaceFrameSnapshot(video, box);

    if (!frame1 || !frame2 || frame1.length !== frame2.length) return true;

    let diffSum = 0, samples = 0;
    for (let i = 0; i < frame1.length; i += 32) {
      diffSum += Math.abs(frame1[i] - frame2[i]);
      samples++;
    }
    const avgDiff = samples ? (diffSum / samples) : 0;

    /* Extremely low variance across the whole face crop over ~140ms is a strong
       signal of a static printed photo or a frozen/looping screen replay. */
    return avgDiff >= 0.35;
  } catch (e) {
    /* If the motion probe itself fails (e.g. canvas restrictions), don't block
       on this secondary signal alone. */
    return true;
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runFaceScan(){
  isProcessingFrame = true;
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const guide = document.getElementById('camGuide');
  if(!video.videoWidth || !video.videoHeight){ isProcessingFrame = false; return; }

  updateFlashlightState(video);

  if(overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
  if(overlay.height !== video.videoHeight) overlay.height = video.videoHeight;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0,0,overlay.width, overlay.height);

  const enrolled = employees.filter(e=>e.descriptor);
  if(!enrolled.length){ setScanStatus(T('scanner.status.noRecords'), 'fail'); isProcessingFrame = false; return; }

  let det;
  try{
    det = await faceapi.detectSingleFace(getDetectionInput(video), getScanOpts())
      .withFaceLandmarks().withFaceDescriptor();
  }catch(e){ isProcessingFrame = false; return; }

  if(!det){ 
    scanGapMs = 220;
    guide.classList.remove('active');
    setScanStatus(T('scanner.status.scanning'), 'detecting'); 
    isProcessingFrame = false; 
    return; 
  }

  guide.classList.add('active');
  scanGapMs = 100;
  const box = det.detection.box;
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.strokeRect(box.x, box.y, box.width, box.height);

  const matcher = getMatcher(enrolled);
  const best = matcher.findBestMatch(det.descriptor);

  if(best.label === 'unknown'){
    setScanStatus(T('scanner.status.notRecognized'), 'fail');
    scanGapMs = 450;
    isProcessingFrame = false;
    return;
  }

  const emp = employees.find(e=>e.id===best.label);

  /* Step 1/2 (straight-face recognition only - no head-pose challenge). */
  isVerifyingLiveness = true;
  let isMotionReal = await verifyMicroMotionLiveness(video, box);
  isVerifyingLiveness = false;

  if (!isMotionReal) {
    ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = 3; ctx.strokeRect(box.x, box.y, box.width, box.height);
    setScanStatus(T('scanner.status.spoof'), 'fail');
    toast(T('scanner.spoofToast'), 'err');
    isProcessingFrame = false;
    return;
  }

  ctx.strokeStyle = '#10b981'; ctx.lineWidth = 3; ctx.strokeRect(box.x, box.y, box.width, box.height);
  /* Legacy face-only path: attendance is NOT recorded here (finger challenge is mandatory, handled by the server flow). */
  setScanStatus(T('scanner.status.serverOffline'), 'fail');
  isProcessingFrame = false;
}

/* VERIFICATION TABS */
document.querySelectorAll('.vtab').forEach(tab=>{
  tab.addEventListener('click', ()=>{
    document.querySelectorAll('.vtab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    const v = tab.dataset.v;
    document.getElementById('vpin').style.display = v==='pin'?'block':'none';
    document.getElementById('vfp').style.display = v==='fp'?'block':'none';
    if(v==='fp') startFpSensor(); else resetFpState();
  });
});

['attPin','fEmpPin'].forEach(id=>{
  const el = document.getElementById(id);
  el.addEventListener('input', ()=>{
    const raw = el.value, digits = raw.replace(/\D/g,'');
    if(digits.length > 4 || digits !== raw) toast(T('toast.pinFormat'), 'err');
    el.value = digits.slice(0,4);
  });
});
document.getElementById('pinVerifyBtn').addEventListener('click', ()=>{
  const val = document.getElementById('attPin').value.trim();
  const emp = employees.find(e=> e.pin && e.pin === val);
  if(!emp){ const pi = document.getElementById('attPin'); pi.classList.remove('shake'); void pi.offsetWidth; pi.classList.add('shake'); setTimeout(()=>pi.classList.remove('shake'), 500); toast(T('scanner.pinIncorrect'), 'err'); return; }
  document.getElementById('attPin').value='';
  logAttendance(emp, T('methods.pin'));
});

/* FINGERPRINT VERIFICATION */
const fpBox = document.getElementById('fpBox');
const fpHint = document.getElementById('fpHint');
const fpSub = document.getElementById('fpSub');
const fpRetryBtn = document.getElementById('fpRetryBtn');
let fpBusy = false;

function resetFpState(){
  fpBusy = false;
  fpBox.classList.remove('waiting','success','fail');
  fpHint.textContent = T('scanner.fp.hint');
  fpSub.textContent = T('scanner.fp.sub');
  fpRetryBtn.style.display = 'none';
}

async function startFpSensor(){
  if(fpBusy) return;
  if(!employees.length){ toast(T('scanner.fp.noRecords'), 'err'); return; }
  const candidates = employees.filter(e => e.fpTag && e.fpEnrolled && e.fpCredentialId);
  if(!candidates.length){
    fpBox.classList.add('fail');
    fpHint.textContent = T('scanner.fp.noneEnrolled');
    fpSub.textContent = T('scanner.fp.enrollFirst');
    fpRetryBtn.style.display = 'none';
    return;
  }

  fpBusy = true;
  fpBox.classList.remove('success','fail');
  fpBox.classList.add('waiting');
  fpHint.textContent = T('scanner.fp.waiting');
  fpSub.textContent = T('scanner.fp.followPrompt');
  fpRetryBtn.style.display = 'none';

  if(!navigator.credentials || !window.PublicKeyCredential || !window.isSecureContext){
    fpBox.classList.remove('waiting');
    fpBox.classList.add('fail');
    fpHint.textContent = T('scanner.fp.unavailable');
    fpSub.textContent = T('scanner.fp.unavailableSub');
    fpRetryBtn.style.display = 'block';
    fpBusy = false;
    return;
  }

  try{
    const challenge = new Uint8Array(32); crypto.getRandomValues(challenge);
    const allowCredentials = candidates.map(c => ({
      type: 'public-key',
      id: b64ToBuf(c.fpCredentialId)
    }));

    const assertion = await navigator.credentials.get({
      publicKey: {
        challenge,
        allowCredentials,
        userVerification: 'required',
        timeout: 30000
      }
    });

    const rawIdB64 = bufToB64(assertion.rawId);
    const matched = candidates.find(c => c.fpCredentialId === rawIdB64);

    if(matched){
      fpBox.classList.remove('waiting');
      fpBox.classList.add('success');
      fpHint.textContent = T('scanner.fp.verified', {name: matched.name});
      fpSub.textContent = T('scanner.fp.matched');
      fpRetryBtn.style.display = 'none';
      logAttendance(matched, T('methods.fingerprint'));
    } else {
      fpBox.classList.remove('waiting');
      fpBox.classList.add('fail');
      fpHint.textContent = T('scanner.fp.notRecognized');
      fpSub.textContent = T('scanner.fp.notMatched');
      fpRetryBtn.style.display = 'block';
    }
  }catch(err){
    fpBox.classList.remove('waiting');
    fpBox.classList.add('fail');
    fpHint.textContent = T('scanner.fp.cancelled');
    fpSub.textContent = T('scanner.fp.tryAgainSub');
    fpRetryBtn.style.display = 'block';
  }finally{
    fpBusy = false;
  }
}

fpRetryBtn.addEventListener('click', ()=>{ resetFpState(); startFpSensor(); });

/* ATTENDANCE LOGIC & METHOD COMBINATION */
let lastLoggedTime = 0;
let lastLoggedEmpId = '';

function logAttendance(emp, method){
  const now = new Date();
  if(emp.id === lastLoggedEmpId && (now.getTime() - lastLoggedTime) < 5000){
    return;
  }
  lastLoggedEmpId = emp.id;
  lastLoggedTime = now.getTime();

  const dStr = todayStr();
  const tStr = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  
  let action = scanMode;
  let activeLog = logs.find(l => l.empId === emp.id && l.date === dStr && !l.exit);

  if(scanMode === 'AUTO'){
    if(activeLog){
      action = 'OUT';
    } else {
      action = 'IN';
    }
  }

  /* STATE-BASED DUPLICATE VALIDATION (dynamic IN/OUT state check) */
  const currentState = activeLog ? 'IN' : 'OUT';

  if(action === 'IN' && currentState === 'IN'){
    toast(T('attendance.alreadyIn', {name: emp.name}), 'err');
    return;
  }

  if(action === 'OUT' && currentState === 'OUT'){
    toast(T('attendance.alreadyOut', {name: emp.name}), 'err');
    return;
  }

  if(action === 'IN'){
    if(activeLog){
      activeLog.entrance = tStr;
      activeLog.inMethod = method;
      activeLog.method = method;
      activeLog.lastUpdated = now.getTime();
      /* move to end so ordering reflects most recent activity, not original push order */
      logs.splice(logs.indexOf(activeLog), 1);
      logs.push(activeLog);
    } else {
      logs.push({
        empId: emp.id,
        name: emp.name,
        date: dStr,
        entrance: tStr,
        exit: null,
        inMethod: method,
        outMethod: null,
        method: method,
        lastUpdated: now.getTime()
      });
    }
  } else {
    if(activeLog){
      activeLog.exit = tStr;
      activeLog.outMethod = method;
      activeLog.method = (activeLog.inMethod ? activeLog.inMethod : T('methods.manual')) + ' + ' + method;
      activeLog.lastUpdated = now.getTime();
      /* move to end so ordering reflects most recent activity (the checkout), not the earlier check-in time */
      logs.splice(logs.indexOf(activeLog), 1);
      logs.push(activeLog);
    } else {
      logs.push({
        empId: emp.id,
        name: emp.name,
        date: dStr,
        entrance: null,
        exit: tStr,
        inMethod: null,
        outMethod: method,
        method: method,
        lastUpdated: now.getTime()
      });
    }
  }

  saveLogs(logs[logs.length-1]);

  document.getElementById('succName').textContent = emp.name;
  document.getElementById('succMeta').textContent = T('attendance.checked', {action: T('status.' + action), time: tStr});
  const overlay = document.getElementById('successOverlay');
  overlay.classList.add('show');
  setTimeout(()=>overlay.classList.remove('show'), 2000);

  document.getElementById('lastLogEmpty').style.display = 'none';
  const lContent = document.getElementById('lastLogContent');
  lContent.style.display = 'block';
  const displayMethod = activeLog ? activeLog.method : method;
  document.getElementById('llName').textContent = emp.name;
  document.getElementById('llMeta').textContent = T('attendance.lastMeta', {action: T('status.' + action), time: tStr, method: displayMethod});

  toast(T('attendance.recorded', {name: emp.name, action: T('status.' + action)}), 'ok');

  if(currentPage === 'dashboard') renderDashboard();
  if(currentPage === 'reports') renderReports();

  /* Auto-stop the camera immediately after a successful check-in/check-out */
  if(scanStream){ stopScannerCamera(); }
}

/* REPORTS & AUDIT LOGS */
function renderReports(){
  const searchQ = (document.getElementById('reportSearch').value || '').trim().replace(/\s+/g,' ').toLowerCase();
  let list = [...logs];

  const dateFilter = activeFilter;
  const today = todayStr();
  
  if(dateFilter === 'today'){
    list = list.filter(l => l.date === today);
  } else if(dateFilter === 'yesterday'){
    const y = new Date(); y.setDate(y.getDate() - 1);
    const yStr = y.toISOString().slice(0,10);
    list = list.filter(l => l.date === yStr);
  } else if(dateFilter === 'month'){
    const monthPrefix = today.slice(0,7);
    list = list.filter(l => l.date.startsWith(monthPrefix));
  } else if(dateFilter === 'custom'){
    const customVal = document.getElementById('customDate').value;
    if(customVal) list = list.filter(l => l.date === customVal);
  }

  if(searchQ){
    list = list.filter(l => l.name.toLowerCase().includes(searchQ) || l.empId.toLowerCase().includes(searchQ));
  }

  list.sort((a,b)=>(b.lastUpdated||0)-(a.lastUpdated||0));

  const tbody = document.getElementById('reportBody');
  const emptyEl = document.getElementById('reportEmpty');

  if(!list.length){
    tbody.innerHTML = '';
    emptyEl.style.display = 'block';
    return;
  }
  emptyEl.style.display = 'none';

  tbody.innerHTML = list.map(l => {
    const status = l.exit ? 'OUT' : 'IN';
    const statusClass = status === 'IN' ? 'in' : 'out';
    const combinedMethod = l.method || (l.inMethod && l.outMethod ? `${l.inMethod} + ${l.outMethod}` : (l.inMethod || l.outMethod || T('methods.manual')));
    return `
      <tr>
        <td>${escapeHtml(l.empId)}</td>
        <td><b>${escapeHtml(l.name)}</b></td>
        <td>${l.date}</td>
        <td>${l.entrance || '—'}</td>
        <td>${l.exit || '—'}</td>
        <td><span class="badge ${statusClass}">${T('status.' + status)}</span></td>
        <td><small style="color:var(--accent);font-weight:600;">${escapeHtml(combinedMethod)}</small></td>
      </tr>
    `;
  }).join('');
}

document.querySelectorAll('.filters button').forEach(b=>{
  b.addEventListener('click', ()=>{
    document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    activeFilter = b.dataset.f;
    const cInput = document.getElementById('customDate');
    cInput.style.display = activeFilter === 'custom' ? 'inline-block' : 'none';
    refreshReports();
  });
});

document.getElementById('customDate').addEventListener('change', refreshReports);
document.getElementById('reportSearch').addEventListener('input', renderReports);

/* CSV EXPORT */
document.getElementById('exportCsvBtn').addEventListener('click', async ()=>{
  try{ putLogs((await fdb.ref('attendance_logs').once('value')).val() || {}); }catch(e){ toast(T('toast.firebaseError', {error: e.message}), 'err'); }
  if(!logs.length){ toast(T('reports.exportNone'), 'err'); return; }
  const rows = [...logs].sort((a,b)=> (b.date||'').localeCompare(a.date||'') || (b.lastUpdated||0)-(a.lastUpdated||0)).map(l => [
    l.empId, l.name, l.date, l.entrance || '', l.exit || '', l.exit ? 'OUT' : 'IN',
    l.method || (l.inMethod && l.outMethod ? `${l.inMethod} + ${l.outMethod}` : (l.inMethod || l.outMethod || T('methods.manual')))]);
  const head = T('reports.exportHeaders');
  const save = (blob, ext) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = T('files.attendanceReport', {date: todayStr(), ext: ext}); a.click(); };
  if(typeof ExcelJS === 'undefined'){
    save(new Blob([[head, ...rows].map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n')], {type:'text/csv;charset=utf-8;'}), 'csv');
    toast(T('reports.csvFallback'), 'info'); return;
  }
  const wb = new ExcelJS.Workbook(); wb.creator = T('reports.workbookCreator');
  const ws = wb.addWorksheet(T('reports.worksheetName'), { views: [{ state: 'frozen', ySplit: 1 }] });
  const edge = { style: 'thin', color: { argb: 'FFCBD5E1' } }, box = { top: edge, left: edge, bottom: edge, right: edge };
  ws.columns = [14, 24, 13, 13, 13, 11, 30].map((w, i) => ({ header: head[i], width: w }));
  rows.forEach(r => ws.addRow(r));
  const hr = ws.getRow(1); hr.height = 26;
  hr.eachCell(c => { c.font = { bold: true, color: { argb: 'FFFFFFFF' }, size: 11 }; c.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF1E40AF' } };
    c.alignment = { vertical: 'middle', horizontal: 'center' }; c.border = { top: { style: 'thin', color: { argb: 'FF1E3A8A' } }, bottom: { style: 'medium', color: { argb: 'FF1E3A8A' } }, left: edge, right: edge }; });
  ws.eachRow((row, n) => {
    if(n === 1) return;
    row.height = 20;
    row.eachCell({ includeEmpty: true }, (c, col) => {
      c.border = box; c.alignment = { vertical: 'middle', horizontal: col === 2 || col === 7 ? 'left' : 'center', indent: col === 2 || col === 7 ? 1 : 0 };
      c.font = { size: 10.5, bold: col === 2, color: { argb: 'FF0F172A' } };
      if(n % 2 === 0) c.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFF1F5F9' } };
    });
    const s = row.getCell(6), isIn = s.value === 'IN';
    s.font = { bold: true, size: 10.5, color: { argb: isIn ? 'FF92400E' : 'FF166534' } };
    s.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: isIn ? 'FFFEF3C7' : 'FFDCFCE7' } };
  });
  ws.autoFilter = { from: 'A1', to: 'G1' };
  ws.pageSetup = { orientation: 'landscape', fitToPage: true, fitToWidth: 1, fitToHeight: 0 };
  save(new Blob([await wb.xlsx.writeBuffer()], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }), 'xlsx');
  toast(T('reports.excelDownloaded'), 'ok');
});

/* INITIALIZATION */
loadFaceModels(true);
renderDashboard();


/* ===== PWA: manifest, service worker, install prompt, FCM push ===== */
const VAPID_KEY = 'PASTE_YOUR_FIREBASE_WEB_PUSH_VAPID_KEY_HERE';   /* Firebase Console > Project settings > Cloud Messaging > Web Push certificates */
(function pwaInit(){
  const head = document.head;
  [['link', {rel:'manifest', href:'/manifest.json'}],
   ['link', {rel:'apple-touch-icon', href:'/icon-192.png'}],
   ['meta', {name:'theme-color', content:'#030712'}],
   ['meta', {name:'mobile-web-app-capable', content:'yes'}],
   ['meta', {name:'apple-mobile-web-app-capable', content:'yes'}]
  ].forEach(([tag, attrs]) => { if(head.querySelector(tag + '[' + Object.keys(attrs)[0] + '="' + Object.values(attrs)[0] + '"]')) return;
    const el = document.createElement(tag); Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v)); head.appendChild(el); });

  let swReg = null;
  if('serviceWorker' in navigator){
    window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').then(r => { swReg = r; r.update().catch(()=>{}); }).catch(e => console.warn('SW register failed', e)));
  }

  /* Install button (Android/Chrome): shows a small floating "Install" chip */
  let deferred = null;
  window.addEventListener('beforeinstallprompt', e => {
    e.preventDefault(); deferred = e;
    if(document.getElementById('pwaInstall')) return;
    const b = document.createElement('button'); b.id = 'pwaInstall'; b.textContent = 'Install App';
    b.style.cssText = 'position:fixed;right:14px;bottom:calc(14px + env(safe-area-inset-bottom,0px));z-index:9999;padding:10px 16px;border:0;border-radius:999px;background:linear-gradient(135deg,#0ea5e9,#6366f1);color:#fff;font-weight:700;box-shadow:0 8px 24px rgba(14,165,233,.4)';
    b.onclick = async () => { b.remove(); deferred.prompt(); await deferred.userChoice.catch(()=>{}); deferred = null; };
    document.body.appendChild(b);
  });
  window.addEventListener('appinstalled', () => { const b = document.getElementById('pwaInstall'); if(b) b.remove(); });

  /* FCM */
  const loadScript = src => new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
  let msgInit = false;
  window.enablePush = async function(){
    try{
      if(!('Notification' in window) || !('serviceWorker' in navigator)) return;
      if(VAPID_KEY.startsWith('PASTE_')){ console.warn('FCM: set VAPID_KEY in app.js'); return; }
      if(Notification.permission === 'denied') return;
      if(Notification.permission === 'default' && (await Notification.requestPermission()) !== 'granted') return;
      if(!window.firebase.messaging) await loadScript('https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js');
      const reg = swReg || await navigator.serviceWorker.ready;
      const messaging = firebase.messaging();
      const token = await messaging.getToken({ vapidKey: VAPID_KEY, serviceWorkerRegistration: reg });
      if(!token) return;
      if(localStorage.getItem('fcm_token') !== token){
        await fdb.ref('fcm_tokens/' + token.replace(/[.#$\[\]\/]/g, '_')).set({ token, ua: navigator.userAgent.slice(0, 120), updatedAt: TS() });
        localStorage.setItem('fcm_token', token);
      }
      if(!msgInit){
        msgInit = true;
        messaging.onMessage(p => {                                  /* app open: in-app toast instead of system notification */
          const n = p.notification || p.data || {};
          toast((n.title ? n.title + ': ' : '') + (n.body || ''), 'ok');
        });
      }
    }catch(e){ console.warn('FCM setup failed', e); }
  };
})();
