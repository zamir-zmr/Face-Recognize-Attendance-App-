/* Service Worker: offline cache + FCM background push. Must be served from site root (/sw.js). */
importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js');

firebase.initializeApp({
  apiKey: "AIzaSyA0rirztMO13FyXcKYz1aEB1ERYH-HQUbA",
  authDomain: "sweethouse-e3e49.firebaseapp.com",
  databaseURL: "https://sweethouse-e3e49-default-rtdb.firebaseio.com",
  projectId: "sweethouse-e3e49",
  storageBucket: "sweethouse-e3e49.firebasestorage.app",
  messagingSenderId: "579322038108",
  appId: "1:579322038108:web:166b7fdb103080f56d6399"
});
const messaging = firebase.messaging();

const VERSION = 'v4';
const SHELL = `shell-${VERSION}`, RUNTIME = `runtime-${VERSION}`;
const APP_SHELL = ['/', '/app.js', '/bridge.js', '/i18n.js', '/text.json', '/manifest.json',
                   '/icon-192.png', '/icon-512.png', '/finger1.png', '/finger2.png', '/finger3.png', '/finger4.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(SHELL).then(c => Promise.allSettled(APP_SHELL.map(u => c.add(u)))).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => ![SHELL, RUNTIME].includes(k)).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET') return;
  if (url.pathname.startsWith('/api/')) return;                                   // live backend data: never cache
  if (/(firebaseio\.com|firebasestorage|googleapis\.com\/(identitytoolkit|v1|v0)|fcm\.googleapis)/.test(url.host + url.pathname)) return;

  if (req.mode === 'navigate') {                                                  // pages: network first, offline fallback
    e.respondWith(fetch(req).then(r => { const c = r.clone(); caches.open(SHELL).then(x => x.put('/', c)); return r; })
      .catch(() => caches.match('/')));
    return;
  }
  if (url.origin === location.origin || /gstatic\.com|jsdelivr\.net|cdnjs\.cloudflare\.com/.test(url.host)) {
    e.respondWith(caches.match(req).then(hit => {                                 // stale-while-revalidate
      const net = fetch(req).then(r => { if (r && (r.ok || r.type === 'opaque')) { const c = r.clone(); caches.open(RUNTIME).then(x => x.put(req, c)); } return r; }).catch(() => hit);
      return hit || net;
    }));
  }
});

/* ---- FCM background messages (data-only payloads; notification payloads are shown by the browser itself) ---- */
messaging.onBackgroundMessage(p => {
  if (p.notification) return;
  const d = p.data || {};
  return self.registration.showNotification(d.title || 'Attendance', {
    body: d.body || '', icon: '/icon-192.png', badge: '/icon-192.png', data: { url: d.url || '/' }, tag: d.tag || 'attendance'
  });
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  const target = (e.notification.data && e.notification.data.url) || '/';
  e.waitUntil(clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
    for (const c of list) if ('focus' in c) return c.focus();
    return clients.openWindow(target);
  }));
});
