"use strict";
// Presentation only: source messages and control/configuration values remain Chinese.
// Explicit text/attribute bindings let language changes preserve forms and terminals.
const ConsoleI18n = (() => {
  const key = 'sonic-console-language', bindings = new WeakMap();
  let language = 'zh-CN';
  try { if (localStorage.getItem(key) === 'ja') language = 'ja'; } catch {}
  const escape = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const patterns = Object.entries(ConsoleJapanese).filter(([source]) => /\{\d+\}/.test(source)).sort((a,b) => b[0].replace(/\{\d+\}/g,'').length - a[0].replace(/\{\d+\}/g,'').length).map(([source, target]) => {
    const parts = source.split(/(\{\d+\})/), indexes = [];
    const expression = parts.map(part => {
      if (/^\{\d+\}$/.test(part)) { indexes.push(Number(part.slice(1,-1))); return '([\\s\\S]*?)'; }
      return escape(part);
    }).join('');
    return {regex:new RegExp('^'+expression+'$'), indexes, target};
  });
  // Explicit formatting keeps user-provided names and paths literal. Nest format()
  // only for arguments that are themselves interface labels.
  function format(source, ...values) { return {source, values}; }
  function t(value, depth=0) {
    if (value && typeof value === 'object' && 'source' in value) {
      const translated = t(value.source, depth);
      return value.values.length ? translated.replace(/\{(\d+)\}/g, (_,index) => {
        const argument = value.values[index];
        return typeof argument === 'object' ? t(argument, depth+1) : String(argument ?? '');
      }) : translated;
    }
    const source = String(value ?? '');
    if (language !== 'ja' || !source || depth > 6) return source;
    if (Object.hasOwn(ConsoleJapanese, source)) return ConsoleJapanese[source];
    for (const {regex,indexes,target} of patterns) {
      const match = source.match(regex);
      if (!match) continue;
      const args = {}; indexes.forEach((index,i) => { args[index] = match[i+1]; });
      return target.replace(/\{(\d+)\}/g, (_,index) => t(args[index], depth+1));
    }
    // Backend notes concatenate complete sentences according to the selected options.
    const parts = source.split(/(?<=。)/u);
    if (parts.length > 1) return parts.map(part => {
      const trimmed = part.trim();
      return Object.hasOwn(ConsoleJapanese, trimmed) ? part.replace(trimmed, ConsoleJapanese[trimmed]) : part;
    }).join('');
    return source;
  }
  function remember(element, field, source) {
    let fields = bindings.get(element);
    if (!fields) { fields = new Map(); bindings.set(element, fields); }
    fields.set(field, source);
  }
  function text(element, source) {
    remember(element, 'text', source);
    element.textContent = t(source);
    return source; // Preserve assignment-expression semantics in the calling code.
  }
  function raw(element, value) {
    bindings.get(element)?.delete('text');
    element.textContent = value;
  }
  function attr(element, name, source) {
    remember(element, name, source);
    element.setAttribute(name, t(source));
    return source;
  }
  function setLanguage(value) {
    language = value === 'ja' ? 'ja' : 'zh-CN';
    try { localStorage.setItem(key, language); } catch {}
    document.documentElement.lang = language;
    const walker = document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let element;
    while ((element = walker.nextNode())) {
      for (const [field,source] of bindings.get(element) || []) {
        // A button may start empty and later receive icons/labels. Its children
        // own their bindings; replacing the parent would remove live controls.
        if (field === 'text') {
          if (!element.childElementCount) element.textContent = t(source);
        }
        else element.setAttribute(field, t(source));
      }
    }
    document.getElementById('language-select').value = language;
  }
  // Only the initial, static HTML is scanned. Dynamic UI uses text()/attr().
  const walker = document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
  let element;
  while ((element = walker.nextNode())) {
    if (element.nodeType === Node.TEXT_NODE && /[\u3400-\u9fff]/u.test(element.textContent)
        && !element.parentElement?.closest('script,style,pre,code')) {
      text(element, element.textContent);
    } else if (element.nodeType === Node.ELEMENT_NODE) {
      for (const name of ['title','aria-label','placeholder']) {
        if (element.hasAttribute(name)) attr(element, name, element.getAttribute(name));
      }
    }
  }
  document.getElementById('language-select').onchange = event => setLanguage(event.target.value);
  setLanguage(language);
  return {t, text, raw, attr, format, setLanguage, get language() { return language; }};
})();
