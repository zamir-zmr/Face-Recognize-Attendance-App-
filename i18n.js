/* i18n.js - loads every piece of UI text from text.json, fills the static HTML, then starts the app scripts. */
(function () {
  'use strict';
  window.TEXT = {};

  /* T('section.key', {var: value}) -> text from text.json, with {var} placeholders replaced. Arrays/objects are returned as-is. */
  window.T = function (path, vars) {
    var node = window.TEXT, parts = String(path).split('.');
    for (var i = 0; i < parts.length; i++) {
      if (node == null || typeof node !== 'object') { node = undefined; break; }
      node = node[parts[i]];
    }
    if (node === undefined) { console.warn('[i18n] missing text key:', path); return path; }
    if (typeof node !== 'string') return node;
    if (!vars) return node;
    return node.replace(/\{(\w+)\}/g, function (m, k) { return Object.prototype.hasOwnProperty.call(vars, k) ? vars[k] : m; });
  };

  /* Fill every element marked with data-i18n* attributes. Safe to call again for dynamically added markup. */
  window.applyText = function (root) {
    root = root || document;
    root.querySelectorAll('[data-i18n]').forEach(function (el) { el.textContent = T(el.getAttribute('data-i18n')); });
    root.querySelectorAll('[data-i18n-html]').forEach(function (el) { el.innerHTML = T(el.getAttribute('data-i18n-html')); });
    root.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) { el.setAttribute('placeholder', T(el.getAttribute('data-i18n-placeholder'))); });
    root.querySelectorAll('[data-i18n-title]').forEach(function (el) { el.setAttribute('title', T(el.getAttribute('data-i18n-title'))); });
    root.querySelectorAll('[data-i18n-alt]').forEach(function (el) { el.setAttribute('alt', T(el.getAttribute('data-i18n-alt'))); });
    if (root === document && TEXT.meta && TEXT.meta.pageTitle) document.title = TEXT.meta.pageTitle;
  };

  /* Load scripts one after another (order matters: the backend bridge extends app.js). */
  function loadScripts(list, done) {
    var i = 0;
    (function next() {
      if (i >= list.length) { if (done) done(); return; }
      var s = document.createElement('script');
      s.src = list[i++];
      s.onload = next;
      s.onerror = function () { console.error('[i18n] failed to load script', s.src); next(); };
      document.body.appendChild(s);
    })();
  }

  function fail(err) {
    console.error('[i18n] could not load text.json', err);
    document.body.innerHTML = '<div style="font:16px system-ui,sans-serif;padding:24px;color:#f8fafc;background:#0f172a;min-height:100vh">' +
      'Unable to load text.json. Serve this folder over HTTP(S) and make sure text.json is next to index.html.</div>';
  }

  fetch('text.json', { cache: 'no-cache' })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function (json) {
      window.TEXT = json;
      applyText(document);
      loadScripts(['app.js', 'bridge.js']);
    })
    .catch(fail);
})();
