// 21_figure5.js — 원고 그림 5(판정 설계 절차). PowerPoint 도형으로 그려 편집할 수 있다.
// 실행: npm install pptxgenjs && node code/21_figure5.js
const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.defineLayout({ name: 'FIG5', width: 6.6, height: 6.1 });
pres.layout = 'FIG5';
pres.title = 'Figure 5. Designing an expert-panel adoption decision';
const s = pres.addSlide();
s.background = { color: 'FFFFFF' };

const INK = '1F2328', MUTED = '5B6168', LINE = '6B7178', TINT = 'F3F5F7', BLUE = '2A78D6';
const F = 'Arial';
const sub = (t) => ({ text: t, options: { subscript: true } });
const sup = (t) => ({ text: t, options: { superscript: true } });
const T = (t, o = {}) => ({ text: t, options: o });
const sig = (k) => [T('σ'), sub(k), sup('2')];

// ---------- helpers
function box(x, y, w, h, runs, o = {}) {
  s.addText(runs, {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x, y, w, h,
    fill: { color: o.fill || TINT }, line: { color: o.lineColor || LINE, width: o.lw || 0.75, dashType: o.dash || 'solid' },
    fontFace: F, fontSize: o.size || 9.5, color: INK, align: o.align || 'left', valign: 'middle',
    margin: o.margin || [4, 7, 4, 7], paraSpaceAfter: 1, isTextBox: true, objectName: o.name,
  });
}
function step(n, x, y, w, h, runs, name) {
  box(x, y, w, h, [T(n + '  ', { bold: true, color: BLUE, fontSize: 10.5 }), ...runs], { name });
}
function arrow(x1, y1, x2, y2, dash) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.0001, h: Math.abs(y2 - y1) || 0.0001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color: LINE, width: 1, endArrowType: 'triangle', dashType: dash || 'solid' },
  });
}

// ---------- geometry
const X = 0.2, W = 4.2, HALF = 2.04, GAP = W - 2 * HALF, RX = 4.62, RW = 1.8;

// Step 1
step('1', X, 0.2, W, 0.46, [T('Define the decision the panel scores will support', { bold: true })], 'step1');
box(X, 0.92, HALF, 0.78, [T('Comparative decision', { bold: true, breakLine: true }), T('Choose among candidates', { breakLine: true }), T('(which output is better?)', { color: MUTED })], { name: 'comparative' });
box(X + HALF + GAP, 0.92, HALF, 0.78, [T('Threshold decision', { bold: true, breakLine: true }), T('Go/no-go on one output', { breakLine: true }), T('(does it meet a standard?)', { color: MUTED })], { dash: 'dash', name: 'threshold' });
arrow(X + HALF / 2, 0.66, X + HALF / 2, 0.92);
arrow(X + HALF + GAP + HALF / 2, 0.66, X + HALF + GAP + HALF / 2, 0.92);

// alternative (untested) route
box(RX, 0.92, RW, 0.78, [T('Alternative: rate a reference output alongside and compare against it ', { color: MUTED }), T('(untested; Sec. 7.2)', { color: MUTED, italic: true })], { fill: 'FFFFFF', dash: 'sysDash', lineColor: '9AA0A6', size: 8.5, name: 'alternative' });
arrow(X + W, 1.31, RX, 1.31, 'sysDash');

// Step 2
step('2', X, 1.98, W, 0.5, [T('Set the smallest difference that matters, ', { bold: true }), T('Δ', { bold: true, italic: true }), T(' (scale points)', { bold: true })], 'step2');
arrow(X + HALF / 2, 1.7, X + HALF / 2, 1.98);
arrow(X + HALF + GAP + HALF / 2, 1.7, X + HALF + GAP + HALF / 2, 1.98);

// Step 3
step('3', X, 2.76, W, 0.74, [T('Run a pilot evaluation and estimate rater severity variance ', { bold: true }), ...sig('r').map(r => ({ ...r, options: { ...r.options, bold: true } })), T(' and residual variance ', { bold: true }), ...sig('e').map(r => ({ ...r, options: { ...r.options, bold: true } })), T('', { breakLine: true }), T('Every rater rates every candidate (crossed design)', { color: MUTED, fontSize: 8.5 })], 'step3');
arrow(X + W / 2, 2.48, X + W / 2, 2.76);

// Step 4
step('4', X, 3.78, W, 0.4, [T('Compute the panel size ', { bold: true }), T('k', { bold: true, italic: true }), T(' for target accuracy ', { bold: true }), T('p', { bold: true, italic: true })], 'step4');
arrow(X + W / 2, 3.5, X + W / 2, 3.78);
box(X, 4.26, HALF, 0.62, [T('Comparative', { bold: true, breakLine: true }), T('k', { italic: true }), T(' = 2'), ...sig('e'), T(' '), T('z', { italic: true }), sup('2'), T(' / '), T('Δ', { italic: true }), sup('2')], { fill: 'FFFFFF', align: 'center', name: 'k_comparative' });
box(X + HALF + GAP, 4.26, HALF, 0.62, [T('Threshold', { bold: true, breakLine: true }), T('k', { italic: true }), T(' = ('), ...sig('r'), T(' + '), ...sig('e'), T(') '), T('z', { italic: true }), sup('2'), T(' / '), T('Δ', { italic: true }), sup('2')], { fill: 'FFFFFF', dash: 'dash', align: 'center', name: 'k_threshold' });
s.addText([T('z', { italic: true }), T(': standard normal quantile of '), T('p', { italic: true }), T(' (1.28 for '), T('p', { italic: true }), T(' = .90).', { breakLine: true }), T('Round up, and report '), T('k', { italic: true }), T(' as a range because pilot estimates are uncertain.')],
  { x: X, y: 4.9, w: W, h: 0.34, fontFace: F, fontSize: 8.5, color: MUTED, margin: 0, isTextBox: true, objectName: 'k_note' });

// Step 5
step('5', X, 5.5, W, 0.5, [T('Record the decision type, panel size, unit of comparison, ICC form, and results across panel splits', { bold: true })], 'step5');
arrow(X + W / 2, 5.28, X + W / 2, 5.5);

// Example table (this study)
s.addText([T('Example: this study', { bold: true, breakLine: true }), T('pooled estimates, ', { color: MUTED }), T('p', { color: MUTED, italic: true }), T(' = .90', { color: MUTED, breakLine: true }), T('(Table 12)', { color: MUTED })],
  { x: RX, y: 3.62, w: RW, h: 0.58, fontFace: F, fontSize: 8.5, color: INK, margin: 0, isTextBox: true, objectName: 'example_title' });
const hdr = { bold: true, fill: { color: TINT }, color: INK };
s.addTable([
  [{ text: 'Δ', options: { ...hdr, italic: true } }, { text: 'Comp.', options: hdr }, { text: 'Thresh.', options: hdr }],
  ['0.50', '3', '8'], ['0.25', '10', '32'], ['0.11', '49', '163'],
], { x: RX, y: 4.26, w: RW, colW: [0.5, 0.6, 0.7], rowH: 0.2, fontFace: F, fontSize: 8.5, color: INK, align: 'center', valign: 'middle', margin: 0.02, border: { type: 'solid', color: 'C9CDD2', pt: 0.5 }, objectName: 'example_table' });

pres.writeFile({ fileName: process.argv[2] || require('path').join(__dirname, '..', 'outputs', 'figures', 'fig5_decision_procedure.pptx') }).then(f => console.log('written', f));
