'use strict';

// ── Layout constants (mirror svggen.go) ───────────────────────────────────────
const L = {
  W: 960, MARGIN: 30, BOX_GAP: 15, ARROW: 20,
  GROUP_GAP: 10, NORM_H: 34, NORM_W: 300,
  GLOBAL_H: 40, GLOBAL_W: 300,
  LOGITS_H: 32, LOGITS_W: 200,
};

// Simplified chip heights for editor view (not the full block SVG fragments)
const CHIP_H = {
  attention:            100,
  full_attention_gated: 110,
  mla_attention:        120,
  gated_delta_net:       90,
  swiglu:                56,
  moe_with_shared:      110,
};
const CHIP_H_DEFAULT = 90;

const BUILDERS = {
  block: ['attention', 'full_attention_gated', 'mla_attention', 'gated_delta_net'],
  ffn:   ['swiglu', 'moe_with_shared'],
};

const BUILDER_COLOR = {
  attention:            '#1565c0',
  full_attention_gated: '#1565c0',
  mla_attention:        '#1565c0',
  gated_delta_net:      '#2e7d32',
  swiglu:               '#e65100',
  moe_with_shared:      '#e65100',
};

const BUILDER_FILL = {
  attention:            'url(#attnGrad)',
  full_attention_gated: 'url(#attnGrad)',
  mla_attention:        'url(#attnGrad)',
  gated_delta_net:      'url(#ssmGrad)',
  swiglu:               'url(#ffnGrad)',
  moe_with_shared:      'url(#ffnGrad)',
};

const BUILDER_LABEL = {
  attention:            'Attention',
  full_attention_gated: 'Full Attention (Gated)',
  mla_attention:        'MLA Attention',
  gated_delta_net:      'Gated Delta-Net',
  swiglu:               'SwiGLU FFN',
  moe_with_shared:      'MoE + Shared Expert',
};

const archExt = ".arch.toml";

// ── App state ─────────────────────────────────────────────────────────────────
let model = null;         // parsed ArchDef (Go JSON: PascalCase field names)
let displayOrder = [];    // block names in display order (cosmetic only)
let undoStack = [];
let redoStack = [];
let selected = null;      // {type: 'arch'|'block'|'routing'|'ffn'|'add-block', name?}
let dragState = null;

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
let svgEl, svgContainer, inspectorBody, statusMsg, tomlEditor, tomlStatus;

// ── API ───────────────────────────────────────────────────────────────────────
async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
  return r;
}

async function apiPost(path, body) {
  return fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function listArchs() {
  return (await apiGet('/api/arch')).json();
}

async function getDef(name) {
  return (await apiGet(`/arch/${name}${archExt}`)).text();
}

async function parseTOML(toml) {
  const r = await apiPost('/api/parse', { toml });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || 'parse error');
  return data;
}

async function serializeModel(m) {
  const r = await apiPost('/api/serialize', m);
  if (!r.ok) throw new Error(await r.text());
  return r.text();
}

async function validateTOML(toml) {
  const r = await apiPost('/api/validate', { toml });
  return r.json();
}

// ── Undo / Redo ───────────────────────────────────────────────────────────────
function pushUndo() {
  undoStack.push(JSON.stringify({ model, displayOrder }));
  if (undoStack.length > 60) undoStack.shift();
  redoStack = [];
  syncUndoButtons();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.stringify({ model, displayOrder }));
  const snap = JSON.parse(undoStack.pop());
  model = snap.model;
  displayOrder = snap.displayOrder;
  syncUndoButtons();
  renderAll();
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.stringify({ model, displayOrder }));
  const snap = JSON.parse(redoStack.pop());
  model = snap.model;
  displayOrder = snap.displayOrder;
  syncUndoButtons();
  renderAll();
}

function syncUndoButtons() {
  $('btn-undo').disabled = undoStack.length === 0;
  $('btn-redo').disabled = redoStack.length === 0;
}

// ── SVG helpers ───────────────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function chipH(builder) {
  return CHIP_H[builder] ?? CHIP_H_DEFAULT;
}

function layerGroupH(blockBuilder, ffnBuilder) {
  return 28 + L.BOX_GAP + L.NORM_H + L.ARROW +
    chipH(blockBuilder) + L.BOX_GAP + L.NORM_H + L.ARROW +
    chipH(ffnBuilder) + L.BOX_GAP + 10;
}

// Returns [{name, role}] in display order.
function getRoutingPaths(m) {
  const r = (m.Layers && m.Layers.Routing) || {};
  const paths = [];
  for (const name of displayOrder) {
    if (name === r.IfTrue)                    paths.push({ name, role: 'if_true' });
    else if (name === r.IfFalse && name !== r.IfTrue) paths.push({ name, role: 'if_false' });
  }
  // Append routing-referenced blocks not yet in displayOrder
  if (r.IfTrue  && !paths.find(p => p.name === r.IfTrue))  paths.push({ name: r.IfTrue,  role: 'if_true'  });
  if (r.IfFalse && !paths.find(p => p.name === r.IfFalse)) paths.push({ name: r.IfFalse, role: 'if_false' });
  return paths;
}

// ── SVG generation ────────────────────────────────────────────────────────────
function svgDefs() {
  return `<defs>
  <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <path d="M0,0 L8,3 L0,6" fill="#555"/>
  </marker>
  <linearGradient id="ssmGrad"    x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#e8f5e9"/><stop offset="100%" stop-color="#c8e6c9"/>
  </linearGradient>
  <linearGradient id="attnGrad"   x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#e3f2fd"/><stop offset="100%" stop-color="#bbdefb"/>
  </linearGradient>
  <linearGradient id="ffnGrad"    x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#fff3e0"/><stop offset="100%" stop-color="#ffe0b2"/>
  </linearGradient>
  <linearGradient id="normGrad"   x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#f3e5f5"/><stop offset="100%" stop-color="#e1bee7"/>
  </linearGradient>
  <linearGradient id="globalGrad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#fce4ec"/><stop offset="100%" stop-color="#f8bbd0"/>
  </linearGradient>
  <filter id="shadow" x="-2%" y="-2%" width="104%" height="104%">
    <feDropShadow dx="1" dy="1" stdDeviation="2" flood-opacity="0.12"/>
  </filter>
</defs>`;
}

function svgArrow(cx, y) {
  return `<line x1="${cx}" y1="${y}" x2="${cx}" y2="${y + L.ARROW}" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)"/>`;
}

function svgNorm(x, y, label) {
  const w = L.NORM_W, h = L.NORM_H;
  return `<g transform="translate(${x},${y})">
  <rect width="${w}" height="${h}" rx="5" fill="url(#normGrad)" stroke="#7b1fa2" stroke-width="1" filter="url(#shadow)"/>
  <text x="${w/2}" y="14" text-anchor="middle" font-weight="600" fill="#6a1b9a">RMSNorm</text>
  <text x="${w/2}" y="27" text-anchor="middle" font-size="10" fill="#777">${esc(label)}</text>
</g>`;
}

function svgGlobalBox(x, y, title, subtitle, action, isSel) {
  const w = L.GLOBAL_W, h = L.GLOBAL_H;
  const stroke = isSel ? '#b71c1c' : '#c62828';
  const sw = isSel ? '2' : '1.2';
  return `<g transform="translate(${x},${y})" data-click="${esc(action)}" style="cursor:pointer">
  <rect width="${w}" height="${h}" rx="6" fill="url(#globalGrad)" stroke="${stroke}" stroke-width="${sw}" filter="url(#shadow)"/>
  <text x="${w/2}" y="17" text-anchor="middle" font-weight="600" fill="${stroke}">${esc(title)}</text>
  <text x="${w/2}" y="32" text-anchor="middle" font-size="10" fill="#777">${esc(subtitle)}</text>
</g>`;
}

function svgBlockChip(x, y, w, h, builder, blockName, isSel) {
  const color = BUILDER_COLOR[builder] ?? '#555';
  const fill  = BUILDER_FILL[builder]  ?? '#eee';
  const label = BUILDER_LABEL[builder] ?? builder;
  const sw = isSel ? '2.5' : '1.2';
  return `<g transform="translate(${x},${y})" data-click="block:${esc(blockName)}" style="cursor:pointer">
  <rect width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${color}" stroke-width="${sw}" filter="url(#shadow)"/>
  <text x="${w/2}" y="22" text-anchor="middle" font-weight="700" font-size="13" fill="${color}">${esc(label)}</text>
  <text x="${w/2}" y="40" text-anchor="middle" font-size="11" fill="#555">${esc(blockName)}</text>
</g>`;
}

function svgFFNChip(x, y, w, h, builder, isSel) {
  const color = BUILDER_COLOR[builder] ?? '#e65100';
  const fill  = BUILDER_FILL[builder]  ?? 'url(#ffnGrad)';
  const label = BUILDER_LABEL[builder] ?? builder;
  const sw = isSel ? '2.5' : '1.2';
  return `<g transform="translate(${x},${y})" data-click="ffn" style="cursor:pointer">
  <rect width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${color}" stroke-width="${sw}" filter="url(#shadow)"/>
  <text x="${w/2}" y="22" text-anchor="middle" font-weight="700" font-size="13" fill="${color}">${esc(label)}</text>
</g>`;
}

function svgDragHandle(x, y) {
  return [0, 5, 10].map(dy =>
    `<line x1="${x+3}" y1="${y+dy}" x2="${x+17}" y2="${y+dy}" stroke="#aaa" stroke-width="2.5" stroke-linecap="round"/>`
  ).join('');
}

function svgSmallBtn(x, y, label, action) {
  return `<g data-click="${esc(action)}" style="cursor:pointer" transform="translate(${x},${y})">
  <rect width="22" height="20" rx="3" fill="#f0f0f0" stroke="#ccc" stroke-width="0.8"/>
  <text x="11" y="14" text-anchor="middle" font-size="13" fill="#444">${label}</text>
</g>`;
}

function svgLegend(m, cx) {
  const items = [
    { label: 'Global',  fill: 'url(#globalGrad)', stroke: '#c62828' },
    { label: 'RMSNorm', fill: 'url(#normGrad)',   stroke: '#7b1fa2' },
  ];
  if (m.Blocks) {
    for (const blk of Object.values(m.Blocks)) {
      if (!items.find(i => i.label === blk.Builder)) {
        items.push({ label: blk.Builder, fill: BUILDER_FILL[blk.Builder] ?? '#eee', stroke: BUILDER_COLOR[blk.Builder] ?? '#555' });
      }
    }
  }
  if (m.FFN) items.push({ label: m.FFN.Builder, fill: 'url(#ffnGrad)', stroke: '#e65100' });

  const totalW = items.reduce((s, it) => s + 24 + it.label.length * 7 + 16, 0);
  let lx = 0;
  const rows = items.map(it => {
    const r = `<rect x="${lx}" width="18" height="14" rx="3" fill="${it.fill}" stroke="${it.stroke}" stroke-width="0.8"/>
    <text x="${lx + 24}" y="12" font-size="11" fill="#555">${esc(it.label)}</text>`;
    lx += 24 + it.label.length * 7 + 16;
    return r;
  });
  return `<g transform="translate(${cx - totalW / 2},52)">${rows.join('')}</g>`;
}

function generateSVG(m) {
  const cx    = L.W / 2;
  const normX = cx - L.NORM_W / 2;
  const chipW = 500;
  const chipX = cx - chipW / 2;
  const paths = getRoutingPaths(m);
  const ffnBuilder = (m.FFN && m.FFN.Builder) || 'swiglu';
  const canSwap = paths.length >= 2;

  // Compute total height
  let totalH = 50 + 30 + 30 + L.GLOBAL_H + L.ARROW;
  for (let i = 0; i < paths.length; i++) {
    if (i > 0) totalH += L.GROUP_GAP;
    const blkDef = m.Blocks && m.Blocks[paths[i].name];
    totalH += layerGroupH(blkDef ? blkDef.Builder : 'full_attention_gated', ffnBuilder);
  }
  totalH += 30 + L.ARROW + L.NORM_H + L.ARROW + L.GLOBAL_H + L.ARROW + L.LOGITS_H + 70 + 20;

  const p = [];
  p.push(svgDefs());
  p.push(`<rect width="${L.W}" height="${totalH}" fill="#fafafa" rx="8"/>`);

  // Title — click to open arch inspector
  const archName = (m.Architecture && m.Architecture.Name) || '(unnamed)';
  const selArch = selected && selected.type === 'arch';
  p.push(`<text x="${cx}" y="32" text-anchor="middle" font-size="20" font-weight="bold" fill="${selArch ? '#b71c1c' : '#333'}" data-click="arch" style="cursor:pointer">${esc(archName)} Architecture</text>`);

  p.push(svgLegend(m, cx));

  p.push(`<text x="${cx}" y="84" text-anchor="middle" font-size="10" fill="#999"><tspan font-family="monospace">@{name}</tspan> = engine builtin,\u2002<tspan font-family="monospace">\${name}</tspan> = GGUF-resolved param</text>`);

  let cur = 110;

  // Token embedding
  const embLabel = (m.Architecture && m.Architecture.TiedEmbeddings)
    ? 'token_embd.weight (tied to output)'
    : ((m.Weights && m.Weights.Global && m.Weights.Global['token_embd']) || 'token_embd.weight');
  p.push(svgGlobalBox(cx - L.GLOBAL_W / 2, cur, 'Token Embedding', embLabel, 'arch', selArch));
  cur += L.GLOBAL_H;
  p.push(svgArrow(cx, cur));
  cur += L.ARROW;

  // Layer block groups
  for (let gi = 0; gi < paths.length; gi++) {
    const pt = paths[gi];
    if (gi > 0) cur += L.GROUP_GAP;

    const blkDef = m.Blocks && m.Blocks[pt.name];
    const blkBuilder = blkDef ? blkDef.Builder : 'full_attention_gated';
    const blkH = chipH(blkBuilder);
    const ffnH = chipH(ffnBuilder);
    const gH = layerGroupH(blkBuilder, ffnBuilder);
    const gW = L.W - 2 * L.MARGIN;
    const color = BUILDER_COLOR[blkBuilder] ?? '#555';
    const isSel = selected && selected.type === 'block' && selected.name === pt.name;

    // Dashed group border
    p.push(`<rect x="${L.MARGIN}" y="${cur}" width="${gW}" height="${gH}" rx="8"
      fill="${isSel ? 'rgba(25,100,200,0.03)' : 'none'}"
      stroke="${color}" stroke-width="${isSel ? '2.2' : '1.5'}" stroke-dasharray="6,3" opacity="0.8"
      data-click="block:${esc(pt.name)}" style="cursor:pointer"/>`);

    // Group label
    const roleNote = pt.role === 'if_true' ? 'rule\u00a0=\u00a0true' : 'rule\u00a0=\u00a0false';
    p.push(`<text x="${L.MARGIN + 40}" y="${cur + 20}" font-size="11" font-weight="600" fill="${color}" data-click="block:${esc(pt.name)}" style="cursor:pointer">${esc(pt.name)} \u2014 when ${roleNote}</text>`);

    // Drag handle
    p.push(`<g data-drag="${esc(pt.name)}" style="cursor:grab">`);
    p.push(svgDragHandle(L.MARGIN + 12, cur + 9));
    p.push(`</g>`);

    // Swap buttons
    if (canSwap) {
      const bx = L.W - L.MARGIN - 56;
      if (gi > 0)                 p.push(svgSmallBtn(bx,      cur + 4, '▲', `swap-up:${pt.name}`));
      if (gi < paths.length - 1) p.push(svgSmallBtn(bx + 26, cur + 4, '▼', `swap-down:${pt.name}`));
    }

    cur += 28;

    // Pre-attention norm
    cur += L.BOX_GAP;
    const attnNorm = (m.Layers && m.Layers.CommonWeights && m.Layers.CommonWeights['attn_norm']) || 'attn_norm.weight';
    p.push(svgNorm(normX, cur, attnNorm));
    cur += L.NORM_H;
    p.push(svgArrow(cx, cur));
    cur += L.ARROW;

    // Block chip
    p.push(svgBlockChip(chipX, cur, chipW, blkH, blkBuilder, pt.name, isSel));
    p.push(`<text x="${chipX + chipW + 14}" y="${cur + blkH / 2 + 4}" font-size="11" fill="#555" font-weight="600">+ residual</text>`);
    cur += blkH + L.BOX_GAP;

    // Post-attention norm
    const postNorm = (m.Layers && m.Layers.CommonWeights && m.Layers.CommonWeights['ffn_norm']) || 'ffn_norm.weight';
    p.push(svgNorm(normX, cur, postNorm));
    cur += L.NORM_H;
    p.push(svgArrow(cx, cur));
    cur += L.ARROW;

    // FFN chip
    const isSelFFN = selected && selected.type === 'ffn';
    p.push(svgFFNChip(chipX, cur, chipW, ffnH, ffnBuilder, isSelFFN));
    p.push(`<text x="${chipX + chipW + 14}" y="${cur + ffnH / 2 + 4}" font-size="11" fill="#555" font-weight="600">+ residual</text>`);
    cur += ffnH + L.BOX_GAP + 10;
  }

  // Add-block button
  p.push(`<g data-click="add-block" style="cursor:pointer" transform="translate(${cx - 50},${cur + 2})">
  <rect width="100" height="22" rx="4" fill="#e8f0fe" stroke="#4a7cf7" stroke-width="1"/>
  <text x="50" y="15" text-anchor="middle" font-size="12" fill="#1a4fd4">+ Add Block</text>
</g>`);
  cur += 30;

  p.push(svgArrow(cx, cur));
  cur += L.ARROW;

  // Final norm
  const outputNorm = (m.Weights && m.Weights.Global && m.Weights.Global['output_norm']) || 'output_norm.weight';
  p.push(svgNorm(normX, cur, outputNorm));
  cur += L.NORM_H;
  p.push(svgArrow(cx, cur));
  cur += L.ARROW;

  // LM Head
  const lmLabel = (m.Architecture && m.Architecture.TiedEmbeddings)
    ? 'tied: reuses token_embd.weight'
    : ((m.Weights && m.Weights.Global && m.Weights.Global['output']) || 'output.weight');
  p.push(svgGlobalBox(cx - L.GLOBAL_W / 2, cur, 'LM Head', lmLabel, 'arch', false));
  cur += L.GLOBAL_H;
  p.push(svgArrow(cx, cur));
  cur += L.ARROW;

  // Logits
  p.push(`<g transform="translate(${cx - L.LOGITS_W / 2},${cur})">
  <rect width="${L.LOGITS_W}" height="${L.LOGITS_H}" rx="6" fill="#eee" stroke="#9e9e9e" stroke-width="1"/>
  <text x="${L.LOGITS_W / 2}" y="20" text-anchor="middle" font-weight="600" fill="#424242">Logits [n_vocab]</text>
</g>`);
  cur += L.LOGITS_H + 28;

  // Footer: routing rule (clickable)
  const rule    = (m.Layers && m.Layers.Routing && m.Layers.Routing.Rule)    || '';
  const ifTrue  = (m.Layers && m.Layers.Routing && m.Layers.Routing.IfTrue)  || '';
  const ifFalse = (m.Layers && m.Layers.Routing && m.Layers.Routing.IfFalse) || '';
  const rSel    = selected && selected.type === 'routing';
  p.push(`<text x="${L.MARGIN}" y="${cur}" font-size="11" fill="${rSel ? '#1a4fd4' : '#666'}" data-click="routing" style="cursor:pointer">
  <tspan font-weight="600">Routing rule:</tspan>
  <tspan font-family="monospace" fill="${rSel ? '#1a4fd4' : '#333'}"> ${esc(rule)}</tspan>
  <tspan fill="${rSel ? '#1a4fd4' : '#666'}"> \u2192 true:\u00a0${esc(ifTrue)} / false:\u00a0${esc(ifFalse)}</tspan>
</text>`);

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${L.W} ${totalH}" width="${L.W}" height="${totalH}" font-family="system-ui,-apple-system,sans-serif" font-size="13">
${p.join('\n')}
</svg>`;
}

// ── Render ────────────────────────────────────────────────────────────────────
function renderAll() {
  if (!model) {
    svgContainer.innerHTML = '';
    svgEl = null;
    $('empty-state').classList.add('visible');
    $('btn-save').disabled = true;
    renderInspector();
    return;
  }
  $('empty-state').classList.remove('visible');
  $('btn-save').disabled = false;

  svgContainer.innerHTML = generateSVG(model);
  svgEl = svgContainer.querySelector('svg');
  attachSVGEvents();
  renderInspector();
}

// ── SVG event handling ────────────────────────────────────────────────────────
function attachSVGEvents() {
  if (!svgEl) return;
  svgEl.addEventListener('click', onSVGClick);
  svgEl.querySelectorAll('[data-drag]').forEach(el => {
    el.addEventListener('mousedown', onDragStart);
  });
}

function onSVGClick(e) {
  const target = e.target.closest('[data-click]');
  if (!target) return;
  const action = target.dataset.click;

  if      (action === 'arch')              selected = { type: 'arch' };
  else if (action === 'routing')           selected = { type: 'routing' };
  else if (action === 'ffn')               selected = { type: 'ffn' };
  else if (action === 'add-block')         selected = { type: 'add-block' };
  else if (action.startsWith('block:'))    selected = { type: 'block', name: action.slice(6) };
  else if (action.startsWith('swap-up:'))  { swapBlockUp(action.slice(8));   return; }
  else if (action.startsWith('swap-down:')){ swapBlockDown(action.slice(10)); return; }
  renderAll();
}

// Drag to reorder block groups
function onDragStart(e) {
  const blockName = e.currentTarget.dataset.drag;
  if (!blockName || !svgEl) return;
  const groups = [];
  svgEl.querySelectorAll('[data-drag]').forEach(el => {
    const rect = el.getBoundingClientRect();
    groups.push({ name: el.dataset.drag, midY: rect.top + rect.height / 2 });
  });
  dragState = { blockName, startY: e.clientY, groups };
  document.addEventListener('mousemove', onDragMove);
  document.addEventListener('mouseup',   onDragEnd);
  e.preventDefault();
}

function onDragMove() {
  if (svgEl) svgEl.style.cursor = 'grabbing';
}

function onDragEnd(e) {
  if (!dragState) return;
  document.removeEventListener('mousemove', onDragMove);
  document.removeEventListener('mouseup',   onDragEnd);
  if (svgEl) svgEl.style.cursor = '';

  const dy    = e.clientY - dragState.startY;
  const mine  = dragState.groups.find(g => g.name === dragState.blockName);
  const other = dragState.groups.find(g => g.name !== dragState.blockName);

  if (mine && other) {
    const newMidY   = mine.midY + dy;
    const origBelow = mine.midY > other.midY;
    const nowBelow  = newMidY  > other.midY;
    if (origBelow !== nowBelow) {
      pushUndo();
      const a = displayOrder.indexOf(dragState.blockName);
      const b = displayOrder.findIndex(n => n !== dragState.blockName);
      if (a !== -1 && b !== -1) [displayOrder[a], displayOrder[b]] = [displayOrder[b], displayOrder[a]];
    }
  }
  dragState = null;
  renderAll();
}

// ── Inspector ─────────────────────────────────────────────────────────────────
function renderInspector() {
  if (!model || !selected) {
    inspectorBody.innerHTML = '<p class="inspector-hint">Click an element in the diagram to inspect and edit it.</p>';
    return;
  }
  switch (selected.type) {
    case 'arch':      renderArchInspector();               break;
    case 'block':     renderBlockInspector(selected.name); break;
    case 'routing':   renderRoutingInspector();            break;
    case 'ffn':       renderFFNInspector();                break;
    case 'add-block': renderAddBlockInspector();           break;
    default:
      inspectorBody.innerHTML = '<p class="inspector-hint">Select an element.</p>';
  }
}

function fld(label, inputHTML) {
  return `<div class="field-group"><label>${esc(label)}</label>${inputHTML}</div>`;
}

function kvTableHTML(sectionLabel, obj, prefix) {
  const entries = Object.entries(obj || {});
  const rows = entries.map(([k, v]) => `<tr>
    <td><input type="text" value="${esc(k)}" data-kv-key="${esc(prefix)}:${esc(k)}" class="kv-key"></td>
    <td><input type="text" value="${esc(String(v))}" data-kv-val="${esc(prefix)}:${esc(k)}" class="kv-val"></td>
    <td><span data-kv-del="${esc(prefix)}:${esc(k)}" style="cursor:pointer;color:#c62828;font-size:16px;line-height:1;padding:0 3px" title="Delete">\xd7</span></td>
  </tr>`).join('');
  return `<div class="inspector-section">${esc(sectionLabel)}</div>
<table class="kv-table">
  <thead><tr><th>Key</th><th>Value</th><th></th></tr></thead>
  <tbody>${rows}</tbody>
</table>
<button class="btn-small" data-kv-add="${esc(prefix)}" style="margin-top:4px">+ Add entry</button>`;
}

function renderArchInspector() {
  const a = model.Architecture || {};
  const l = model.Layers || {};
  inspectorBody.innerHTML =
    `<div class="inspector-section">Architecture</div>` +
    fld('Name', `<input type="text" id="arch-name" value="${esc(a.Name || '')}">`) +
    `<div class="field-group"><label><input type="checkbox" id="arch-tied"${a.TiedEmbeddings ? ' checked' : ''}> Tied Embeddings (LM head reuses token_embd)</label></div>` +
    `<div class="inspector-section">Layers</div>` +
    fld('Layer Count Expression', `<input type="text" id="layers-count" value="${esc(l.Count || '')}">`) +
    fld('Layer Prefix',           `<input type="text" id="layers-prefix" value="${esc(l.Prefix || '')}">`) +
    kvTableHTML('Common Weights', l.CommonWeights, 'common_weights') +
    kvTableHTML('Global Weights', (model.Weights && model.Weights.Global) || {}, 'global_weights');

  on('arch-name',    'change', e => mutate(() => { model.Architecture.Name = e.target.value; }));
  on('arch-tied',    'change', e => mutate(() => { model.Architecture.TiedEmbeddings = e.target.checked; }));
  on('layers-count', 'change', e => mutate(() => { model.Layers.Count = e.target.value; }));
  on('layers-prefix','change', e => mutate(() => { model.Layers.Prefix = e.target.value; }));
  attachKVHandlers(inspectorBody);
}

function renderRoutingInspector() {
  const r = (model.Layers && model.Layers.Routing) || {};
  const names = model.Blocks ? Object.keys(model.Blocks) : [];
  const opts  = sel => names.map(n => `<option value="${esc(n)}"${n === sel ? ' selected' : ''}>${esc(n)}</option>`).join('');
  inspectorBody.innerHTML =
    `<div class="inspector-section">Routing</div>` +
    fld('Rule Expression',    `<textarea id="routing-rule" rows="3">${esc(r.Rule || '')}</textarea>`) +
    fld('If True \u2192 Block',  `<select id="routing-iftrue">${opts(r.IfTrue)}</select>`) +
    fld('If False \u2192 Block', `<select id="routing-iffalse">${opts(r.IfFalse)}</select>`);

  on('routing-rule',    'change', e => mutate(() => { model.Layers.Routing.Rule    = e.target.value; }));
  on('routing-iftrue',  'change', e => mutate(() => { model.Layers.Routing.IfTrue  = e.target.value; }));
  on('routing-iffalse', 'change', e => mutate(() => { model.Layers.Routing.IfFalse = e.target.value; }));
}

function renderBlockInspector(blockName) {
  const blk = model.Blocks && model.Blocks[blockName];
  if (!blk) {
    inspectorBody.innerHTML = `<p class="inspector-hint">Block not found: ${esc(blockName)}</p>`;
    return;
  }
  const builderOpts = BUILDERS.block
    .map(b => `<option value="${b}"${b === blk.Builder ? ' selected' : ''}>${b}</option>`).join('');
  inspectorBody.innerHTML =
    `<div class="inspector-section">Block: ${esc(blockName)}</div>` +
    fld('Name',    `<input type="text" id="blk-name" value="${esc(blockName)}">`) +
    fld('Builder', `<select id="blk-builder">${builderOpts}</select>`) +
    kvTableHTML('Weights', blk.Weights, `bw:${blockName}`) +
    kvTableHTML('Config',  blk.Config,  `bc:${blockName}`) +
    `<div class="btn-row"><button class="btn-small btn-danger" id="btn-del-block">Delete Block</button></div>`;

  on('blk-name', 'change', e => {
    const n = e.target.value.trim();
    if (n && n !== blockName) mutate(() => renameBlock(blockName, n));
  });
  on('blk-builder', 'change', e => mutate(() => { model.Blocks[blockName].Builder = e.target.value; }));
  on('btn-del-block', 'click', () => {
    if (!confirm(`Delete block "${blockName}"?`)) return;
    mutate(() => deleteBlock(blockName));
  });
  attachKVHandlers(inspectorBody);
}

function renderFFNInspector() {
  const ffn = model.FFN || { Builder: 'swiglu', Weights: {}, Config: {} };
  const builderOpts = BUILDERS.ffn
    .map(b => `<option value="${b}"${b === ffn.Builder ? ' selected' : ''}>${b}</option>`).join('');
  inspectorBody.innerHTML =
    `<div class="inspector-section">FFN</div>` +
    fld('Builder', `<select id="ffn-builder">${builderOpts}</select>`) +
    kvTableHTML('Weights', ffn.Weights, 'fw') +
    kvTableHTML('Config',  ffn.Config,  'fc');

  on('ffn-builder', 'change', e => mutate(() => { model.FFN.Builder = e.target.value; }));
  attachKVHandlers(inspectorBody);
}

function renderAddBlockInspector() {
  const builderOpts = BUILDERS.block.map(b => `<option value="${b}">${b}</option>`).join('');
  inspectorBody.innerHTML =
    `<div class="inspector-section">Add Block</div>` +
    fld('Block Name', `<input type="text" id="new-blk-name" placeholder="e.g. my_attention">`) +
    fld('Builder',    `<select id="new-blk-builder">${builderOpts}</select>`) +
    `<div class="btn-row"><button class="btn-small" id="btn-do-add">Add Block</button></div>`;

  on('btn-do-add', 'click', () => {
    const name    = $('new-blk-name').value.trim();
    const builder = $('new-blk-builder').value;
    if (!name) { setStatus('Block name required', true); return; }
    if (model.Blocks && model.Blocks[name]) { setStatus(`"${name}" already exists`, true); return; }
    mutate(() => addBlock(name, builder));
    selected = { type: 'block', name };
    renderAll();
  });
}

function on(id, event, handler) {
  const el = $(id);
  if (el) el.addEventListener(event, handler);
}

// ── KV table handlers ─────────────────────────────────────────────────────────
function attachKVHandlers(container) {
  container.querySelectorAll('[data-kv-add]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = prompt('New key name:');
      if (!key) return;
      mutate(() => setKV(btn.dataset.kvAdd, key, ''));
      renderAll();
    });
  });
  container.querySelectorAll('[data-kv-del]').forEach(el => {
    el.addEventListener('click', () => {
      const [prefix, key] = splitKV(el.dataset.kvDel);
      mutate(() => delKV(prefix, key));
      renderAll();
    });
  });
  container.querySelectorAll('.kv-key').forEach(el => {
    const [prefix, oldKey] = splitKV(el.dataset.kvKey);
    el.addEventListener('change', () => {
      const newKey = el.value.trim();
      if (!newKey || newKey === oldKey) return;
      mutate(() => renameKV(prefix, oldKey, newKey));
      renderAll();
    });
  });
  container.querySelectorAll('.kv-val').forEach(el => {
    const [prefix, key] = splitKV(el.dataset.kvVal);
    el.addEventListener('change', () => mutate(() => setKV(prefix, key, el.value)));
  });
}

function splitKV(encoded) {
  const i = encoded.indexOf(':');
  return [encoded.slice(0, i), encoded.slice(i + 1)];
}

function kvTarget(prefix) {
  if (prefix === 'common_weights') return model.Layers && model.Layers.CommonWeights;
  if (prefix === 'global_weights') return model.Weights && model.Weights.Global;
  if (prefix === 'fw') return model.FFN && model.FFN.Weights;
  if (prefix === 'fc') return model.FFN && model.FFN.Config;
  if (prefix.startsWith('bw:')) { const b = model.Blocks[prefix.slice(3)]; return b && b.Weights; }
  if (prefix.startsWith('bc:')) { const b = model.Blocks[prefix.slice(3)]; return b && b.Config; }
  return null;
}

function setKV(prefix, key, value) {
  const t = kvTarget(prefix);
  if (t) t[key] = value;
}

function delKV(prefix, key) {
  const t = kvTarget(prefix);
  if (t) delete t[key];
}

function renameKV(prefix, oldKey, newKey) {
  const t = kvTarget(prefix);
  if (!t) return;
  const val = t[oldKey];
  delete t[oldKey];
  t[newKey] = val;
}

// ── Block mutations ───────────────────────────────────────────────────────────
function mutate(fn) {
  pushUndo();
  fn();
  renderAll();
}

function renameBlock(oldName, newName) {
  if (!model.Blocks) return;
  model.Blocks[newName] = model.Blocks[oldName];
  delete model.Blocks[oldName];
  const r = model.Layers && model.Layers.Routing;
  if (r) {
    if (r.IfTrue  === oldName) r.IfTrue  = newName;
    if (r.IfFalse === oldName) r.IfFalse = newName;
  }
  const i = displayOrder.indexOf(oldName);
  if (i !== -1) displayOrder[i] = newName;
  if (selected && selected.name === oldName) selected.name = newName;
}

function deleteBlock(blockName) {
  if (!model.Blocks) return;
  delete model.Blocks[blockName];
  const i = displayOrder.indexOf(blockName);
  if (i !== -1) displayOrder.splice(i, 1);
  const r = model.Layers && model.Layers.Routing;
  if (r) {
    if (r.IfTrue  === blockName) r.IfTrue  = '';
    if (r.IfFalse === blockName) r.IfFalse = '';
  }
  selected = null;
}

function addBlock(name, builder) {
  if (!model.Blocks) model.Blocks = {};
  model.Blocks[name] = { Builder: builder, Weights: {}, Config: {}, Cache: {} };
  displayOrder.push(name);
}

function swapBlockUp(name) {
  const i = displayOrder.indexOf(name);
  if (i > 0) {
    pushUndo();
    [displayOrder[i - 1], displayOrder[i]] = [displayOrder[i], displayOrder[i - 1]];
    renderAll();
  }
}

function swapBlockDown(name) {
  const i = displayOrder.indexOf(name);
  if (i !== -1 && i < displayOrder.length - 1) {
    pushUndo();
    [displayOrder[i], displayOrder[i + 1]] = [displayOrder[i + 1], displayOrder[i]];
    renderAll();
  }
}

// ── Load / Save ───────────────────────────────────────────────────────────────
async function loadFromTOML(tomlText, filename) {
  try {
    const parsed = await parseTOML(tomlText);
    model = parsed;
    displayOrder = [];
    const r = model.Layers && model.Layers.Routing;
    if (r) {
      if (r.IfTrue)                              displayOrder.push(r.IfTrue);
      if (r.IfFalse && r.IfFalse !== r.IfTrue)  displayOrder.push(r.IfFalse);
    }
    undoStack = [];
    redoStack = [];
    selected  = null;
    syncUndoButtons();
    if (filename) $('filename-label').textContent = filename;
    setStatus('Loaded', false);
    renderAll();
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
}

async function loadBuiltinDef(name) {
  if (!name) return;
  try {
    await loadFromTOML(await getDef(name), name + archExt);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
}

async function saveTOML() {
  if (!model) return;
  try {
    const toml = await serializeModel(model);
    const blob = new Blob([toml], { type: 'text/plain' });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = ((model.Architecture && model.Architecture.Name) || 'arch') + archTomlExt;
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus('Downloaded', false);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
}

// ── TOML tab ──────────────────────────────────────────────────────────────────
let activeTab = 'visual';

async function switchTab(tab) {
  if (tab === activeTab) return;
  if (tab === 'toml') {
    if (model) {
      try {
        tomlEditor.value   = await serializeModel(model);
        tomlStatus.textContent = '';
      } catch (err) {
        tomlEditor.value   = '';
        tomlStatus.textContent = String(err.message || err);
      }
    } else {
      tomlEditor.value = '';
    }
    $('main-area').style.display = 'none';
    $('toml-area').style.display = 'flex';
    $('tab-visual').classList.remove('active');
    $('tab-toml').classList.add('active');
    activeTab = 'toml';
  } else {
    const txt = tomlEditor.value.trim();
    if (txt) {
      const result = await validateTOML(txt);
      if (!result.valid) {
        tomlStatus.textContent = (result.errors || ['Invalid TOML']).join('\n');
        tomlStatus.style.color = '#c62828';
        return; // keep TOML tab open
      }
      await loadFromTOML(txt, null);
    }
    $('toml-area').style.display = 'none';
    $('main-area').style.display = 'flex';
    $('tab-visual').classList.add('active');
    $('tab-toml').classList.remove('active');
    activeTab = 'visual';
  }
}

let tomlTimer = null;
function onTomlInput() {
  clearTimeout(tomlTimer);
  tomlTimer = setTimeout(async () => {
    const txt = tomlEditor.value.trim();
    if (!txt) { tomlStatus.textContent = ''; return; }
    const result = await validateTOML(txt);
    tomlStatus.textContent = result.valid ? '\u2713 Valid' : (result.errors || []).join('\n');
    tomlStatus.style.color = result.valid ? '#2e7d32' : '#c62828';
  }, 700);
}

// ── Status bar ────────────────────────────────────────────────────────────────
let statusTimer = null;
function setStatus(msg, isErr) {
  statusMsg.textContent = msg;
  statusMsg.className   = 'status-msg' + (isErr ? ' error' : ' ok');
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => {
    statusMsg.textContent = '';
    statusMsg.className   = 'status-msg';
  }, 3500);
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  svgContainer  = $('svg-container');
  inspectorBody = $('inspector-body');
  statusMsg     = $('status-msg');
  tomlEditor    = $('toml-editor');
  tomlStatus    = $('toml-status');

  // Populate built-in def list
  try {
    const names = await listArchs();
    const sel   = $('def-select');
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value       = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
  } catch (e) {
    console.warn('Could not list defs:', e);
  }

  // Toolbar
  $('def-select').addEventListener('change', e => { if (e.target.value) loadBuiltinDef(e.target.value); });
  $('btn-load-file').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => loadFromTOML(ev.target.result, file.name);
    reader.readAsText(file);
    e.target.value = ''; // allow reloading same file
  });
  $('btn-save').addEventListener('click', saveTOML);
  $('btn-undo').addEventListener('click', undo);
  $('btn-redo').addEventListener('click', redo);
  $('tab-visual').addEventListener('click', () => switchTab('visual'));
  $('tab-toml').addEventListener('click',   () => switchTab('toml'));
  tomlEditor.addEventListener('input', onTomlInput);

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'z') { e.preventDefault(); undo(); }
    if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'z'))) { e.preventDefault(); redo(); }
    if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveTOML(); }
  });

  renderAll();
});
