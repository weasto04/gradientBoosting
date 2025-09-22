// Minimal gradient boosting and residual visualization (two canvases), vanilla JS

// Coordinate system & data
const canvasLeft = document.getElementById('plotLeft');
const canvasRight = document.getElementById('plotRight');
const canvasBL = document.getElementById('plotBottomLeft');
const canvasBR = document.getElementById('plotBottomRight');
const canvasR3L = document.getElementById('plotRow3Left');
const canvasR3R = document.getElementById('plotRow3Right');
const ctxL = canvasLeft.getContext('2d');
const ctxR = canvasRight.getContext('2d');
const ctxBL = canvasBL.getContext('2d');
const ctxBR = canvasBR.getContext('2d');
const ctxR3L = canvasR3L.getContext('2d');
const ctxR3R = canvasR3R.getContext('2d');
const W = canvasLeft.width;
const H = canvasLeft.height;

// World bounds (x in [-2.5, 2.5], y in [-1.5, 6]) to fit a parabola y = 1 + x^2 nicely
const X_MIN = -2.5, X_MAX = 2.5;
const Y_MIN = -1.5, Y_MAX = 6;

// UI elements
const regenBtn = document.getElementById('regen');
const togglePoints = document.getElementById('togglePoints');

// Settings
const N_POINTS = 120; // dataset size
const NOISE_STD = 0.35; // Gaussian noise
const SHOW_GT_MS = 900; // show green ground truth curve for ms after regeneration

// Model settings
const N_TREES = 30; // still used if we wanted full boosting; left now uses one tree of depth 2
const LEARNING_RATE = 0.1; // shrinkage
const MAX_DEPTH = 2; // shallow trees: exactly 2 for both base and residual fits
const MIN_SAMPLES_SPLIT = 6; // small so we can fit small partitions

// Data containers
let dataX = [];
let dataY = [];
let baseTree = null; // depth-2 regression tree fit on data
let residuals1 = []; // y - y_hat_base
let residualTree1 = null; // depth-2 regression tree fit on residuals1
let combinedPred = null; // x => baseTree(x) + residualTree1(x)
let residuals2 = []; // y - combinedPred(x)
let residualTree2 = null; // depth-2 regression tree fit on residuals2
let finalPred = null; // x => base + r1 + r2
let residualsFinal = []; // y - finalPred(x)
let residualTreeFinal = null; // depth-2 regression tree fit on final residuals (for viz)
let showGroundTruthUntil = 0;
let hoverXLeft = null;
let hoverXRight = null;
let hoverXBL = null;
let hoverXBR = null;
let hoverXR3L = null;
let hoverXR3R = null;

// Utilities
function randBetween(a, b) { return a + Math.random() * (b - a); }
function gaussNoise(std) {
  // Box–Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
}

// Ground-truth function (green parabola)
function fTrue(x) { return 1 + x * x; }

// Canvas transform helpers
function xToPx(x) { return (x - X_MIN) / (X_MAX - X_MIN) * W; }
function yToPx(y) { return H - (y - Y_MIN) / (Y_MAX - Y_MIN) * H; }
function pxToX(px) { return X_MIN + (px / W) * (X_MAX - X_MIN); }

// Simple axis drawing
// Nice ticks util (Wilkinson-like nice numbers)
function niceNum(range, round) {
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / Math.pow(10, exponent);
  let niceFraction;
  if (round) {
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;
  } else {
    if (fraction <= 1) niceFraction = 1;
    else if (fraction <= 2) niceFraction = 2;
    else if (fraction <= 5) niceFraction = 5;
    else niceFraction = 10;
  }
  return niceFraction * Math.pow(10, exponent);
}

function niceTicks(min, max, maxTicks = 6) {
  const range = niceNum(max - min, false);
  const step = niceNum(range / (maxTicks - 1), true);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let t = niceMin; t <= niceMax + 1e-12; t += step) {
    // avoid -0
    const v = Math.abs(t) < 1e-12 ? 0 : t;
    ticks.push(v);
  }
  return ticks;
}

function drawAxes(ctx) {
  ctx.save();
  ctx.strokeStyle = '#3a3f4b';
  ctx.lineWidth = 1;
  // main axes lines at 0 if inside range
  if (Y_MIN < 0 && Y_MAX > 0) {
    const y0 = yToPx(0);
    ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(W, y0); ctx.stroke();
  }
  if (X_MIN < 0 && X_MAX > 0) {
    const x0 = xToPx(0);
    ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, H); ctx.stroke();
  }

  // ticks and labels
  const xTicks = niceTicks(X_MIN, X_MAX, 7);
  const yTicks = niceTicks(Y_MIN, Y_MAX, 7);
  ctx.font = '11px system-ui';
  ctx.fillStyle = '#9aa0a6';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  const bottomY = H - 4;
  for (const xv of xTicks) {
    const px = xToPx(xv);
    // tick at bottom
    ctx.beginPath(); ctx.moveTo(px, H); ctx.lineTo(px, H - 6); ctx.stroke();
    // label
    const txt = Math.abs(xv) < 1e-9 ? '0' : Math.abs(xv) >= 10 ? xv.toFixed(0) : xv.toFixed(1);
    ctx.fillText(txt, px, bottomY);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const yv of yTicks) {
    const py = yToPx(yv);
    // tick at left
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(6, py); ctx.stroke();
    const txt = Math.abs(yv) < 1e-9 ? '0' : Math.abs(yv) >= 10 ? yv.toFixed(0) : yv.toFixed(1);
    ctx.fillText(txt, 28, py); // draw a bit in from the left
  }
  ctx.restore();
}

// Data generation
function generateData() {
  dataX = new Array(N_POINTS);
  dataY = new Array(N_POINTS);
  for (let i = 0; i < N_POINTS; i++) {
    const x = randBetween(X_MIN, X_MAX);
    const y = fTrue(x) + gaussNoise(NOISE_STD);
    dataX[i] = x;
    dataY[i] = y;
  }
}

// Tiny regression tree (CART) with binary splits on x only
class TreeNode {
  constructor() {
    this.isLeaf = true;
    this.threshold = null; // split threshold on x
    this.left = null;
    this.right = null;
    this.value = 0; // prediction at leaf
  }
  predict(x) {
    if (this.isLeaf) return this.value;
    if (x <= this.threshold) return this.left.predict(x);
    else return this.right.predict(x);
  }
}

function mean(arr) {
  if (arr.length === 0) return 0;
  let s = 0; for (const v of arr) s += v; return s / arr.length;
}

function variance(arr, m) {
  if (arr.length === 0) return 0;
  let s = 0; for (const v of arr) { const d = v - m; s += d * d; } return s / arr.length;
}

function buildRegressionTree(X, y, depth = 0) {
  const node = new TreeNode();
  const n = X.length;
  const yMean = mean(y);
  node.value = yMean;

  if (depth >= MAX_DEPTH) return node;
  if (n < MIN_SAMPLES_SPLIT) return node;

  // find best split on x by trying candidate thresholds from sorted unique x
  const pairs = X.map((xi, i) => [xi, y[i]]).sort((a, b) => a[0] - b[0]);
  const xs = pairs.map(p => p[0]);
  const ys = pairs.map(p => p[1]);

  // Prefix sums for left means/variances efficiently
  let bestGain = 0, bestThr = null, bestIdx = -1;
  const prefixSum = new Array(n).fill(0);
  const prefixSum2 = new Array(n).fill(0);
  let s = 0, s2 = 0;
  for (let i = 0; i < n; i++) {
    s += ys[i]; s2 += ys[i] * ys[i];
    prefixSum[i] = s; prefixSum2[i] = s2;
  }

  const totalSum = s, totalSum2 = s2;
  const totalVar = totalSum2 / n - (totalSum / n) ** 2;

  for (let i = 1; i < n; i++) {
    if (xs[i] === xs[i - 1]) continue; // skip identical thresholds
    const leftN = i, rightN = n - i;
    if (leftN < Math.max(2, MIN_SAMPLES_SPLIT / 2) || rightN < Math.max(2, MIN_SAMPLES_SPLIT / 2)) continue;
    const leftSum = prefixSum[i - 1];
    const leftSum2 = prefixSum2[i - 1];
    const rightSum = totalSum - leftSum;
    const rightSum2 = totalSum2 - leftSum2;
    const leftMean = leftSum / leftN;
    const rightMean = rightSum / rightN;
    const leftVar = leftSum2 / leftN - leftMean * leftMean;
    const rightVar = rightSum2 / rightN - rightMean * rightMean;
    // Reduction in MSE
    const gain = totalVar - (leftVar * leftN + rightVar * rightN) / n;
    if (gain > bestGain) {
      bestGain = gain;
      bestIdx = i;
      bestThr = (xs[i - 1] + xs[i]) / 2;
    }
  }

  if (bestThr == null) return node;

  // split data
  const leftX = [], leftY = [], rightX = [], rightY = [];
  for (let i = 0; i < n; i++) {
    const xi = xs[i], yi = ys[i];
    if (xi <= bestThr) { leftX.push(xi); leftY.push(yi); }
    else { rightX.push(xi); rightY.push(yi); }
  }

  node.isLeaf = false;
  node.threshold = bestThr;
  node.left = buildRegressionTree(leftX, leftY, depth + 1);
  node.right = buildRegressionTree(rightX, rightY, depth + 1);
  return node;
}

// Gradient Boosting Regressor with mean init + shrinked residual trees
class GradientBoostingRegressor {
  constructor(nTrees, learningRate) {
    this.nTrees = nTrees;
    this.learningRate = learningRate;
    this.trees = [];
    this.initValue = 0;
  }
  fit(X, y) {
    const n = X.length;
    this.trees = [];
    this.initValue = mean(y);
    let preds = new Array(n).fill(this.initValue);

    for (let t = 0; t < this.nTrees; t++) {
      // residuals
      const residuals = y.map((yi, i) => yi - preds[i]);
      const tree = buildRegressionTree(X, residuals, 0);
      this.trees.push(tree);
      // update predictions
      for (let i = 0; i < n; i++) {
        preds[i] += this.learningRate * tree.predict(X[i]);
      }
    }
  }
  predictOne(x) {
    let yhat = this.initValue;
    for (const tree of this.trees) yhat += this.learningRate * tree.predict(x);
    return yhat;
  }
}

// Rendering
function clear(ctx) { ctx.clearRect(0, 0, W, H); }

function drawPoints(ctx) {
  if (!togglePoints.checked) return;
  ctx.save();
  ctx.fillStyle = '#e6e6e6';
  for (let i = 0; i < dataX.length; i++) {
    const px = xToPx(dataX[i]);
    const py = yToPx(dataY[i]);
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawCurve(ctx, fn, color, width = 2) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  const steps = 400;
  for (let i = 0; i <= steps; i++) {
    const x = X_MIN + (i / steps) * (X_MAX - X_MIN);
    const y = fn(x);
    const px = xToPx(x); const py = yToPx(y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();
}

function drawHoverProbeLeft() {
  if (hoverXLeft == null || baseTree == null) return;
  const px = xToPx(hoverXLeft);
  const yPred = baseTree.predict(hoverXLeft);
  const py = yToPx(yPred);
  ctxL.save();
  ctxL.strokeStyle = '#556';
  ctxL.setLineDash([4, 4]);
  ctxL.beginPath(); ctxL.moveTo(px, 0); ctxL.lineTo(px, H); ctxL.stroke();
  ctxL.setLineDash([]);
  ctxL.fillStyle = '#9b59b6';
  ctxL.beginPath(); ctxL.arc(px, py, 4, 0, Math.PI * 2); ctxL.fill();
  const label = `x=${hoverXLeft.toFixed(2)}, y_pred=${yPred.toFixed(2)}`;
  ctxL.font = '12px system-ui';
  const tw = ctxL.measureText(label).width + 10;
  const th = 18;
  const bx = Math.min(W - tw - 6, Math.max(6, px + 8));
  const by = Math.max(6 + th, Math.min(H - 6, py - 8));
  ctxL.fillStyle = 'rgba(20,22,30,0.9)';
  ctxL.fillRect(bx, by - th, tw, th);
  ctxL.strokeStyle = '#444';
  ctxL.strokeRect(bx, by - th, tw, th);
  ctxL.fillStyle = '#ddd';
  ctxL.fillText(label, bx + 5, by - 5);
  ctxL.restore();
}

function drawHoverProbeRight() {
  if (hoverXRight == null || residualTree1 == null) return;
  const px = xToPx(hoverXRight);
  const rPred = residualTree1.predict(hoverXRight);
  const py = yToPx(rPred);
  ctxR.save();
  ctxR.strokeStyle = '#556';
  ctxR.setLineDash([4, 4]);
  ctxR.beginPath(); ctxR.moveTo(px, 0); ctxR.lineTo(px, H); ctxR.stroke();
  ctxR.setLineDash([]);
  ctxR.fillStyle = '#9b59b6';
  ctxR.beginPath(); ctxR.arc(px, py, 4, 0, Math.PI * 2); ctxR.fill();
  const label = `x=${hoverXRight.toFixed(2)}, r_pred=${rPred.toFixed(2)}`;
  ctxR.font = '12px system-ui';
  const tw = ctxR.measureText(label).width + 10;
  const th = 18;
  const bx = Math.min(W - tw - 6, Math.max(6, px + 8));
  const by = Math.max(6 + th, Math.min(H - 6, py - 8));
  ctxR.fillStyle = 'rgba(20,22,30,0.9)';
  ctxR.fillRect(bx, by - th, tw, th);
  ctxR.strokeStyle = '#444';
  ctxR.strokeRect(bx, by - th, tw, th);
  ctxR.fillStyle = '#ddd';
  ctxR.fillText(label, bx + 5, by - 5);
  ctxR.restore();
}

function drawHoverProbeBL() {
  if (hoverXBL == null || combinedPred == null) return;
  const px = xToPx(hoverXBL);
  const yPred = combinedPred(hoverXBL);
  const py = yToPx(yPred);
  ctxBL.save();
  ctxBL.strokeStyle = '#556';
  ctxBL.setLineDash([4, 4]);
  ctxBL.beginPath(); ctxBL.moveTo(px, 0); ctxBL.lineTo(px, H); ctxBL.stroke();
  ctxBL.setLineDash([]);
  ctxBL.fillStyle = '#9b59b6';
  ctxBL.beginPath(); ctxBL.arc(px, py, 4, 0, Math.PI * 2); ctxBL.fill();
  const label = `x=${hoverXBL.toFixed(2)}, y_pred=${yPred.toFixed(2)}`;
  ctxBL.font = '12px system-ui';
  const tw = ctxBL.measureText(label).width + 10;
  const th = 18;
  const bx = Math.min(W - tw - 6, Math.max(6, px + 8));
  const by = Math.max(6 + th, Math.min(H - 6, py - 8));
  ctxBL.fillStyle = 'rgba(20,22,30,0.9)';
  ctxBL.fillRect(bx, by - th, tw, th);
  ctxBL.strokeStyle = '#444';
  ctxBL.strokeRect(bx, by - th, tw, th);
  ctxBL.fillStyle = '#ddd';
  ctxBL.fillText(label, bx + 5, by - 5);
  ctxBL.restore();
}

function drawHoverProbeBR() {
  if (hoverXBR == null || residualTree2 == null) return;
  const px = xToPx(hoverXBR);
  const rPred = residualTree2.predict(hoverXBR);
  const py = yToPx(rPred);
  ctxBR.save();
  ctxBR.strokeStyle = '#556';
  ctxBR.setLineDash([4, 4]);
  ctxBR.beginPath(); ctxBR.moveTo(px, 0); ctxBR.lineTo(px, H); ctxBR.stroke();
  ctxBR.setLineDash([]);
  ctxBR.fillStyle = '#9b59b6';
  ctxBR.beginPath(); ctxBR.arc(px, py, 4, 0, Math.PI * 2); ctxBR.fill();
  const label = `x=${hoverXBR.toFixed(2)}, r2_pred=${rPred.toFixed(2)}`;
  ctxBR.font = '12px system-ui';
  const tw = ctxBR.measureText(label).width + 10;
  const th = 18;
  const bx = Math.min(W - tw - 6, Math.max(6, px + 8));
  const by = Math.max(6 + th, Math.min(H - 6, py - 8));
  ctxBR.fillStyle = 'rgba(20,22,30,0.9)';
  ctxBR.fillRect(bx, by - th, tw, th);
  ctxBR.strokeStyle = '#444';
  ctxBR.strokeRect(bx, by - th, tw, th);
  ctxBR.fillStyle = '#ddd';
  ctxBR.fillText(label, bx + 5, by - 5);
  ctxBR.restore();
}

function drawHoverProbeR3L() {
  if (hoverXR3L == null || finalPred == null) return;
  const px = xToPx(hoverXR3L);
  const yPred = finalPred(hoverXR3L);
  const py = yToPx(yPred);
  ctxR3L.save();
  ctxR3L.strokeStyle = '#556';
  ctxR3L.setLineDash([4, 4]);
  ctxR3L.beginPath(); ctxR3L.moveTo(px, 0); ctxR3L.lineTo(px, H); ctxR3L.stroke();
  ctxR3L.setLineDash([]);
  ctxR3L.fillStyle = '#9b59b6';
  ctxR3L.beginPath(); ctxR3L.arc(px, py, 4, 0, Math.PI * 2); ctxR3L.fill();
  const label = `x=${hoverXR3L.toFixed(2)}, y_pred=${yPred.toFixed(2)}`;
  ctxR3L.font = '12px system-ui';
  const tw = ctxR3L.measureText(label).width + 10;
  const th = 18;
  const bx = Math.min(W - tw - 6, Math.max(6, px + 8));
  const by = Math.max(6 + th, Math.min(H - 6, py - 8));
  ctxR3L.fillStyle = 'rgba(20,22,30,0.9)';
  ctxR3L.fillRect(bx, by - th, tw, th);
  ctxR3L.strokeStyle = '#444';
  ctxR3L.strokeRect(bx, by - th, tw, th);
  ctxR3L.fillStyle = '#ddd';
  ctxR3L.fillText(label, bx + 5, by - 5);
  ctxR3L.restore();
}

function drawHoverProbeR3R() {
  if (hoverXR3R == null || residualTreeFinal == null) return;
  const px = xToPx(hoverXR3R);
  const rPred = residualTreeFinal.predict(hoverXR3R);
  const py = yToPx(rPred);
  ctxR3R.save();
  ctxR3R.strokeStyle = '#556';
  ctxR3R.setLineDash([4, 4]);
  ctxR3R.beginPath(); ctxR3R.moveTo(px, 0); ctxR3R.lineTo(px, H); ctxR3R.stroke();
  ctxR3R.setLineDash([]);
  ctxR3R.fillStyle = '#9b59b6';
  ctxR3R.beginPath(); ctxR3R.arc(px, py, 4, 0, Math.PI * 2); ctxR3R.fill();
  const label = `x=${hoverXR3R.toFixed(2)}, r3_pred=${rPred.toFixed(2)}`;
  ctxR3R.font = '12px system-ui';
  const tw = ctxR3R.measureText(label).width + 10;
  const th = 18;
  const bx = Math.min(W - tw - 6, Math.max(6, px + 8));
  const by = Math.max(6 + th, Math.min(H - 6, py - 8));
  ctxR3R.fillStyle = 'rgba(20,22,30,0.9)';
  ctxR3R.fillRect(bx, by - th, tw, th);
  ctxR3R.strokeStyle = '#444';
  ctxR3R.strokeRect(bx, by - th, tw, th);
  ctxR3R.fillStyle = '#ddd';
  ctxR3R.fillText(label, bx + 5, by - 5);
  ctxR3R.restore();
}

function render() {
  // Left panel: ground truth briefly, base data points, base tree prediction
  clear(ctxL);
  drawAxes(ctxL);
  const now = performance.now();
  if (now < showGroundTruthUntil) drawCurve(ctxL, fTrue, '#2ecc71', 2); // green
  if (baseTree) drawCurve(ctxL, x => baseTree.predict(x), '#9b59b6', 2.5); // purple
  drawPoints(ctxL);
  drawHoverProbeLeft();

  // Right panel: residuals as red points; residual tree prediction as purple curve
  clear(ctxR);
  drawAxes(ctxR);
  if (residuals1.length > 0) {
    ctxR.save();
    ctxR.fillStyle = '#e74c3c';
    for (let i = 0; i < dataX.length; i++) {
      const px = xToPx(dataX[i]);
      const py = yToPx(residuals1[i]);
      ctxR.beginPath(); ctxR.arc(px, py, 3, 0, Math.PI * 2); ctxR.fill();
    }
    ctxR.restore();
  }
  if (residualTree1) drawCurve(ctxR, x => residualTree1.predict(x), '#9b59b6', 2.5);
  drawHoverProbeRight();

  // Bottom-left: combined prediction (base + residual1) with original points
  clear(ctxBL);
  drawAxes(ctxBL);
  if (combinedPred) drawCurve(ctxBL, x => combinedPred(x), '#9b59b6', 2.5);
  drawPoints(ctxBL);
  drawHoverProbeBL();

  // Bottom-right: residuals of combined and residual2 fit
  clear(ctxBR);
  drawAxes(ctxBR);
  if (residuals2.length > 0) {
    ctxBR.save();
    ctxBR.fillStyle = '#e74c3c';
    for (let i = 0; i < dataX.length; i++) {
      const px = xToPx(dataX[i]);
      const py = yToPx(residuals2[i]);
      ctxBR.beginPath(); ctxBR.arc(px, py, 3, 0, Math.PI * 2); ctxBR.fill();
    }
    ctxBR.restore();
  }
  if (residualTree2) drawCurve(ctxBR, x => residualTree2.predict(x), '#9b59b6', 2.5);
  drawHoverProbeBR();

  // Row 3 left: final prediction (sum of row 2 purple curves) with original points
  clear(ctxR3L);
  drawAxes(ctxR3L);
  if (finalPred) drawCurve(ctxR3L, x => finalPred(x), '#9b59b6', 2.5);
  drawPoints(ctxR3L);
  drawHoverProbeR3L();

  // Row 3 right: residuals of final prediction and their depth-2 fit
  clear(ctxR3R);
  drawAxes(ctxR3R);
  if (residualsFinal.length > 0) {
    ctxR3R.save();
    ctxR3R.fillStyle = '#e74c3c';
    for (let i = 0; i < dataX.length; i++) {
      const px = xToPx(dataX[i]);
      const py = yToPx(residualsFinal[i]);
      ctxR3R.beginPath(); ctxR3R.arc(px, py, 3, 0, Math.PI * 2); ctxR3R.fill();
    }
    ctxR3R.restore();
  }
  if (residualTreeFinal) drawCurve(ctxR3R, x => residualTreeFinal.predict(x), '#9b59b6', 2.5);
  drawHoverProbeR3R();

  requestAnimationFrame(render);
}

// Pipeline: generate -> fit -> show parabola briefly
function regenerateAndFit() {
  generateData();
  // Base fit: a single depth-2 regression tree on (X, y)
  baseTree = buildRegressionTree(dataX, dataY, 0);
  // Compute residuals: r = y - y_hat_base
  residuals1 = dataY.map((yi, i) => yi - baseTree.predict(dataX[i]));
  // Residual fit: a single depth-2 regression tree on (X, residuals1)
  residualTree1 = buildRegressionTree(dataX, residuals1, 0);
  // Combined prediction f1 + r1
  combinedPred = (x) => baseTree.predict(x) + residualTree1.predict(x);
  // Second residuals: r2 = y - (f1 + r1)
  residuals2 = dataY.map((yi, i) => yi - combinedPred(dataX[i]));
  // Second residual fit
  residualTree2 = buildRegressionTree(dataX, residuals2, 0);
  // Final prediction = base + r1 + r2
  finalPred = (x) => baseTree.predict(x) + residualTree1.predict(x) + residualTree2.predict(x);
  // Residuals of final prediction
  residualsFinal = dataY.map((yi, i) => yi - finalPred(dataX[i]));
  // Depth-2 fit to final residuals (for visualization)
  residualTreeFinal = buildRegressionTree(dataX, residualsFinal, 0);
  showGroundTruthUntil = performance.now() + SHOW_GT_MS;
}

// Events
regenBtn.addEventListener('click', regenerateAndFit);
canvasLeft.addEventListener('mousemove', (e) => {
  const rect = canvasLeft.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvasLeft.width / rect.width);
  hoverXLeft = pxToX(px);
});
canvasLeft.addEventListener('mouseleave', () => { hoverXLeft = null; });
canvasRight.addEventListener('mousemove', (e) => {
  const rect = canvasRight.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvasRight.width / rect.width);
  hoverXRight = pxToX(px);
});
canvasRight.addEventListener('mouseleave', () => { hoverXRight = null; });
canvasBL.addEventListener('mousemove', (e) => {
  const rect = canvasBL.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvasBL.width / rect.width);
  hoverXBL = pxToX(px);
});
canvasBL.addEventListener('mouseleave', () => { hoverXBL = null; });
canvasBR.addEventListener('mousemove', (e) => {
  const rect = canvasBR.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvasBR.width / rect.width);
  hoverXBR = pxToX(px);
});
canvasBR.addEventListener('mouseleave', () => { hoverXBR = null; });
canvasR3L.addEventListener('mousemove', (e) => {
  const rect = canvasR3L.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvasR3L.width / rect.width);
  hoverXR3L = pxToX(px);
});
canvasR3L.addEventListener('mouseleave', () => { hoverXR3L = null; });
canvasR3R.addEventListener('mousemove', (e) => {
  const rect = canvasR3R.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvasR3R.width / rect.width);
  hoverXR3R = pxToX(px);
});
canvasR3R.addEventListener('mouseleave', () => { hoverXR3R = null; });

// Boot
regenerateAndFit();
render();
