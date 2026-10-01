/* jevs · brain docs: the shared runtime of every page (menu, languages, section links).
   docs/site.js is generated from src/site.template.js by src/build_site.py: edit the template, not site.js.

   A page joins the site with two things (see src/build_site.py for the details):
     <meta name="jevs-page" content='{"order": 4, "label": {"fr": "…", "en": "…", "de": "…"}, "desc": {…}}'>
     <script>window.I18N = {en: [...], de: [...]};</script><script src="site.js"></script>   (end of <body>)
   Elements carrying data-i18n="N" are written in French; EN[N] / DE[N] replace their innerHTML. */
(function () {
  'use strict';
  var PAGES = [
  {
    "order": 2,
    "label": {
      "fr": "Le fonctionnement",
      "en": "How it works",
      "de": "Funktionsweise"
    },
    "desc": {
      "fr": "Les trois niveaux, le tick, la pensée, la chimie, la mémoire",
      "en": "The three levels, the tick, thinking, the chemistry, memory",
      "de": "Die drei Ebenen, der Tick, das Denken, die Chemie, das Gedächtnis"
    },
    "file": "brain-schemas.html"
  },
  {
    "order": 3,
    "label": {
      "fr": "L’atlas",
      "en": "The atlas",
      "de": "Der Atlas"
    },
    "desc": {
      "fr": "Le génome, la chimie câblée, les hormones (lus dans le template)",
      "en": "The genome, the wired chemistry, the hormones (read from the template)",
      "de": "Das Genom, die verdrahtete Chemie, die Hormone (aus dem Template gelesen)"
    },
    "file": "brain-atlas.html"
  }
];
  var LANGS = ['fr', 'en', 'de'];
  var UI = {
    fr: { prev: 'Page précédente', next: 'Page suivante', lang: 'Langue', pages: 'Pages du bundle' },
    en: { prev: 'Previous page', next: 'Next page', lang: 'Language', pages: 'Bundle pages' },
    de: { prev: 'Vorherige Seite', next: 'Nächste Seite', lang: 'Sprache', pages: 'Seiten des Bundles' }
  };
  var KEY = 'brain-schemas-lang';
  var I18N = window.I18N || {};
  var current = 'fr';

  // -- the registry: built by build_site.py, completed by this page's own meta (a page not yet built in)
  var here = decodeURIComponent(location.pathname.split('/').pop() || '');
  // served without the extension (e.g. /brain-atlas): the registry knows it as brain-atlas.html
  if (here && !/\.html?$/i.test(here)) here += '.html';
  var meta = document.querySelector('meta[name="jevs-page"]');
  if (meta && !PAGES.some(function (p) { return p.file === here; })) {
    try { var m = JSON.parse(meta.getAttribute('content')); m.file = here; PAGES.push(m); } catch (e) {}
  }
  PAGES.sort(function (a, b) { return ((a.order || 99) - (b.order || 99)) || (a.file < b.file ? -1 : 1); });
  var idx = -1;
  PAGES.forEach(function (p, i) { if (p.file === here) idx = i; });

  function pick(o) { return o ? (o[current] || o.en || o.fr || '') : ''; }
  function num(i) { return (i < 9 ? '0' : '') + (i + 1); }
  function withLang(href) { return href.replace(/^([^?#]*)(\?[^#]*)?/, '$1?lang=' + current); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  // -- styles (self-contained, so a new page needs no extra CSS)
  var MONO = "'Geist Mono', ui-monospace, Menlo, monospace";
  var css = [
    '.site-nav{position:sticky;top:0;z-index:50;background:rgba(245,245,245,0.94);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);border-bottom:1px solid rgba(45,49,66,0.10);font-family:"Geist",system-ui,sans-serif}',
    '.site-nav-inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:1.25rem;padding-top:0.55rem;padding-bottom:0.55rem;box-sizing:content-box}',
    '.site-nav a{text-decoration:none;border:0}',
    '.site-brand{font-family:' + MONO + ';font-size:0.66rem;font-weight:500;letter-spacing:0.18em;text-transform:uppercase;color:#4f5d75;white-space:nowrap}',
    '.site-brand:hover{color:#2d3142}',
    '.site-pages{list-style:none;display:flex;gap:0.2rem;margin:0;padding:0;overflow-x:auto;flex:1;scrollbar-width:none}',
    '.site-pages::-webkit-scrollbar{display:none}',
    '.site-pages a{display:inline-flex;align-items:baseline;gap:0.4rem;padding:0.35rem 0.6rem;color:#4f5d75;font-size:0.84rem;white-space:nowrap;border-bottom:2px solid transparent}',
    '.site-pages a .n{font-family:' + MONO + ';font-size:0.62rem;letter-spacing:0.06em;color:#7a8399}',
    '.site-pages a:hover{color:#2d3142}',
    '.site-pages a[aria-current="page"]{color:#2d3142;font-weight:600;border-bottom-color:#eb6c36}',
    '.site-pages a[aria-current="page"] .n{color:#eb6c36}',
    '.site-lang{display:inline-flex;border:1px solid rgba(45,49,66,0.12);border-radius:999px;padding:2px;background:#ffffff;flex-shrink:0}',
    '.site-lang button{font-family:' + MONO + ';font-size:0.66rem;font-weight:500;letter-spacing:0.12em;color:#4f5d75;background:none;border:0;border-radius:999px;padding:0.3rem 0.65rem;cursor:pointer}',
    '.site-lang button:hover{color:#2d3142}',
    '.site-lang button[aria-pressed="true"]{background:#2d3142;color:#f5f5f5}',
    '.site-nav a:focus-visible,.site-lang button:focus-visible,.site-pager a:focus-visible{outline:2px solid #eb6c36;outline-offset:1px}',
    '.site-pager{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:3.5rem}',
    '.site-pager a{display:block;background:#ffffff;border:1px solid rgba(45,49,66,0.12);border-radius:6px;padding:1rem 1.25rem;text-decoration:none;color:#2d3142}',
    '.site-pager a:hover{border-color:#eb6c36}',
    '.site-pager .k{display:block;font-family:' + MONO + ';font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:#7a8399}',
    '.site-pager .t{display:block;font-family:"Instrument Serif","Times New Roman",serif;font-size:1.35rem;margin-top:0.2rem}',
    '.site-pager .d{display:block;font-size:0.8rem;line-height:1.45;color:#4f5d75;margin-top:0.2rem}',
    '.site-pager .next{grid-column:2;text-align:right}',
    'section[id]{scroll-margin-top:4.5rem}',
    '.section-link{display:block;color:inherit;text-decoration:none;border-bottom:0}',
    '.section-link h2::after{content:" #";color:#eb6c36;opacity:0;transition:opacity 0.15s}',
    '.section-link:hover h2::after,.section-link:focus-visible h2::after{opacity:1}',
    '@media (max-width:640px){.site-brand{display:none}.site-pager{grid-template-columns:1fr}.site-pager .next{grid-column:1}}'
  ].join('\n');
  document.head.appendChild(el('style', '', css));

  // -- the menu
  var nav = el('nav', 'site-nav');
  var inner = el('div', 'site-nav-inner');
  var brand = el('a', 'site-brand', 'jevs · brain');
  brand.setAttribute('href', PAGES.length ? PAGES[0].file : '#');
  var list = el('ol', 'site-pages');
  var links = PAGES.map(function (p, i) {
    var li = el('li');
    var a = el('a');
    a.setAttribute('href', p.file);
    if (i === idx) a.setAttribute('aria-current', 'page');
    a.appendChild(el('span', 'n', num(i)));
    a.appendChild(el('span', 'l'));
    li.appendChild(a);
    list.appendChild(li);
    return a;
  });
  var sw = el('div', 'site-lang');
  sw.setAttribute('role', 'group');
  var buttons = LANGS.map(function (l) {
    var b = el('button', '', l.toUpperCase());
    b.type = 'button';
    b.lang = l;
    b.addEventListener('click', function () { setLang(l, true); });
    sw.appendChild(b);
    return b;
  });
  inner.appendChild(brand);
  inner.appendChild(list);
  inner.appendChild(sw);
  nav.appendChild(inner);
  document.body.insertBefore(nav, document.body.firstChild);
  // the bar spans the window whatever the page's body padding
  function fit() {
    var cs = getComputedStyle(document.body);
    nav.style.margin = '-' + cs.paddingTop + ' -' + cs.paddingRight + ' ' + cs.paddingTop + ' -' + cs.paddingLeft;
    inner.style.paddingLeft = cs.paddingLeft;
    inner.style.paddingRight = cs.paddingRight;
  }
  fit();
  window.addEventListener('resize', fit);

  // -- previous / next, above the footer
  var pager = null;
  if (idx >= 0 && PAGES.length > 1) {
    pager = el('nav', 'site-pager');
    [[idx - 1, 'prev'], [idx + 1, 'next']].forEach(function (x) {
      var p = PAGES[x[0]];
      if (!p) return;
      var a = el('a', x[1]);
      a.setAttribute('href', p.file);
      a.appendChild(el('span', 'k'));
      a.appendChild(el('span', 't'));
      a.appendChild(el('span', 'd'));
      a._page = p;
      a._kind = x[1];
      pager.appendChild(a);
    });
    var box = document.querySelector('.container') || document.body;
    var footer = box.querySelector('.footer');
    box.insertBefore(pager, footer || null);
  }

  function renderChrome() {
    var ui = UI[current];
    list.setAttribute('aria-label', ui.pages);
    sw.setAttribute('aria-label', ui.lang);
    links.forEach(function (a, i) {
      a.querySelector('.l').textContent = pick(PAGES[i].label);
      a.title = pick(PAGES[i].desc);
    });
    buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.lang === current)); });
    if (pager) {
      Array.prototype.forEach.call(pager.children, function (a) {
        a.querySelector('.k').textContent = (a._kind === 'prev' ? '← ' : '') + ui[a._kind] + (a._kind === 'next' ? ' →' : '');
        a.querySelector('.t').textContent = pick(a._page.label);
        a.querySelector('.d').textContent = pick(a._page.desc);
      });
    }
  }

  // -- translation
  var nodes = document.querySelectorAll('[data-i18n]');
  var sections = document.querySelectorAll('section[id]');
  function slug(sec) { return current === 'fr' ? sec.id : (sec.getAttribute('data-' + current) || sec.id); }
  function sectionUrl(sec) { return '?lang=' + current + '#' + slug(sec); }
  function setLang(lang, remember) {
    current = lang;
    var dict = I18N[lang];
    nodes.forEach(function (n) {
      if (n.__fr === undefined) n.__fr = n.innerHTML;
      var v = dict && dict[+n.getAttribute('data-i18n')];
      n.innerHTML = lang === 'fr' || v == null ? n.__fr : v;
    });
    document.documentElement.lang = lang;
    // every link to a sibling page carries the language
    document.querySelectorAll('a[href]').forEach(function (a) {
      var base = a.getAttribute('data-base-href');
      if (base === null) {
        base = a.getAttribute('href');
        if (/^[a-z][a-z0-9+.-]*:|^\/\/|^#/i.test(base) || !/\.html?([?#]|$)/i.test(base)) return;
        a.setAttribute('data-base-href', base);
      }
      a.setAttribute('href', withLang(base));
    });
    sections.forEach(function (sec) { var a = sec.querySelector('.section-link'); if (a) a.href = sectionUrl(sec); });
    renderChrome();
    if (remember) { try { localStorage.setItem(KEY, lang); } catch (e) {} }
  }

  // -- section titles become links to their own section
  sections.forEach(function (sec) {
    var h2 = sec.querySelector('h2');
    if (!h2 || h2.closest('.section-link')) return;
    var a = el('a', 'section-link');
    h2.parentNode.insertBefore(a, h2);
    a.appendChild(h2);
    a.addEventListener('click', function (e) {
      e.preventDefault();
      history.pushState(null, '', sectionUrl(sec));
      sec.scrollIntoView({ behavior: 'smooth' });
    });
  });
  // #hormones, #the-hormones, #die-hormone... all reach the same section
  function key(s) {
    try { s = decodeURIComponent(s); } catch (e) {}
    return s.normalize('NFD').toLowerCase().replace(/[^a-z0-9]/g, '').replace(/^(the|les|le|la|der|die|das)/, '');
  }
  function findSection(hash) {
    var k = key(hash.replace(/^#/, ''));
    if (!k) return null;
    for (var i = 0; i < sections.length; i++) {
      var sec = sections[i];
      if (k === key(sec.id) || k === key(sec.getAttribute('data-en') || '') || k === key(sec.getAttribute('data-de') || '')) return sec;
    }
    return null;
  }
  function goToHash() { var sec = findSection(location.hash); if (sec) sec.scrollIntoView(); }

  var lang = null;
  try { lang = new URLSearchParams(location.search).get('lang'); } catch (e) {}
  if (LANGS.indexOf(lang) < 0) { try { lang = localStorage.getItem(KEY); } catch (e) {} }
  if (LANGS.indexOf(lang) < 0) {
    var nl = (navigator.language || '').slice(0, 2).toLowerCase();
    lang = LANGS.indexOf(nl) >= 0 ? nl : 'en';
  }
  setLang(lang, false);
  // after translation reflows the page, land on the section asked in the URL
  goToHash();
  window.addEventListener('hashchange', goToHash);
})();
