import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as math from 'mathjs';
import { Play, Pause, RotateCcw, Brain } from 'lucide-react';

const PRESETS = [
  { name: 'sin(2x)', expr: 'sin(2*x)' },
  { name: 'sin(10x)', expr: 'sin(10*x)' },
  { name: 'x²', expr: 'x^2' },
  { name: '|x|', expr: 'abs(x)' },
  { name: 'gaussian', expr: 'exp(-x^2)*2' },
  { name: 'sin·cos', expr: 'sin(x)*cos(2*x)' },
];
const ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'linear'];
const OPTIMIZERS = ['adam', 'sgd', 'rmsprop'];

const evalFn = (expr, x) => { try { return math.evaluate(expr, { x }); } catch { return 0; } };

const NetworkViz = ({ layers, weights, onAddNeuron, onRemoveNeuron, onAddLayer, onRemoveLayer }) => {
  const W = 420, H = 260, startX = 40;
  const sizes = [1, ...layers, 1];
  const spacing = (W - 80) / (sizes.length + 1);
  const [tooltip, setTooltip] = useState(null);
  
  const nodes = [];
  sizes.forEach((sz, li) => {
    const x = startX + spacing * (li + 1);
    const ns = Math.min(28, (H - 80) / Math.max(sz, 1));
    const startY = (H - 50 - ns * (sz - 1)) / 2 + 15;
    for (let i = 0; i < sz; i++) nodes.push({ x, y: startY + ns * i, layer: li });
  });
  
  const conns = [];
  let ni = 0;
  for (let l = 0; l < sizes.length - 1; l++) {
    const cs = ni, cSz = sizes[l], nSz = sizes[l + 1], ns = cs + cSz;
    for (let i = 0; i < cSz; i++) {
      for (let j = 0; j < nSz; j++) {
        let w = 0; try { w = weights[l][i][j] || 0; } catch {}
        conns.push({ from: nodes[cs + i], to: nodes[ns + j], w });
      }
    }
    ni += cSz;
  }
  const maxW = Math.max(0.5, ...conns.map(c => Math.abs(c.w)));
  
  return (
    <svg width={W} height={H} className="bg-slate-900 rounded-lg">
      {conns.map((c, i) => {
        const n = Math.abs(c.w) / maxW;
        const color = c.w > 0 ? `rgba(59,130,246,${0.15+n*0.85})` : `rgba(239,68,68,${0.15+n*0.85})`;
        return (
          <line key={i} x1={c.from.x} y1={c.from.y} x2={c.to.x} y2={c.to.y}
            stroke={color} strokeWidth={0.5 + n * 3.5} style={{cursor:'pointer'}}
            onMouseEnter={e => setTooltip({x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, w: c.w})}
            onMouseLeave={() => setTooltip(null)} />
        );
      })}
      {nodes.map((nd, i) => (
        <circle key={i} cx={nd.x} cy={nd.y} r={10} fill="#1e293b"
          stroke={nd.layer === 0 ? '#22c55e' : nd.layer === sizes.length - 1 ? '#f97316' : '#60a5fa'} strokeWidth={2.5} />
      ))}
      <text x={startX + spacing} y={H - 8} textAnchor="middle" fill="#94a3b8" fontSize={10}>Input (x)</text>
      <text x={startX + spacing * sizes.length} y={H - 8} textAnchor="middle" fill="#94a3b8" fontSize={10}>Output f(x)</text>
      {layers.map((num, i) => {
        const x = startX + spacing * (i + 2);
        return (
          <g key={i}>
            <text x={x} y={14} textAnchor="middle" fill="#64748b" fontSize={9}>{num}</text>
            <rect x={x-18} y={H-30} width={16} height={16} rx={3} fill="#475569" style={{cursor:'pointer'}} onClick={() => onRemoveNeuron(i)} />
            <text x={x-10} y={H-18} textAnchor="middle" fill="#e2e8f0" fontSize={12} style={{pointerEvents:'none'}}>−</text>
            <rect x={x+2} y={H-30} width={16} height={16} rx={3} fill="#475569" style={{cursor:'pointer'}} onClick={() => onAddNeuron(i)} />
            <text x={x+10} y={H-18} textAnchor="middle" fill="#e2e8f0" fontSize={12} style={{pointerEvents:'none'}}>+</text>
          </g>
        );
      })}
      <rect x={W-72} y={6} width={65} height={20} rx={4} fill="#475569" style={{cursor:'pointer'}} onClick={onAddLayer} />
      <text x={W-39} y={20} textAnchor="middle" fill="#e2e8f0" fontSize={10} style={{pointerEvents:'none'}}>+ Layer</text>
      <rect x={W-72} y={30} width={65} height={20} rx={4} fill="#475569" style={{cursor:'pointer'}} onClick={onRemoveLayer} />
      <text x={W-39} y={44} textAnchor="middle" fill="#e2e8f0" fontSize={10} style={{pointerEvents:'none'}}>− Layer</text>
      {tooltip && (
        <g transform={`translate(${Math.min(Math.max(tooltip.x, 30), W - 50)}, ${Math.min(tooltip.y - 25, H - 25)})`}>
          <rect x={-25} y={0} width={50} height={18} rx={3} fill="#1e293b" stroke="#475569" />
          <text x={0} y={13} textAnchor="middle" fill="#e2e8f0" fontSize={10}>{tooltip.w.toFixed(4)}</text>
        </g>
      )}
    </svg>
  );
};

const DataPlot = ({ train, test, preds }) => {
  const W = 500, H = 452, pad = { t: 25, r: 20, b: 80, l: 45 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
  if (!preds.length) return <svg width={W} height={H} className="bg-slate-900 rounded-lg" />;
  const allY = [...train.map(d => d.y), ...test.map(d => d.y), ...preds.map(d => d.t), ...preds.map(d => d.p)];
  const minY = Math.min(...allY), maxY = Math.max(...allY), yR = maxY - minY || 1;
  const sx = x => pad.l + ((x + 5) / 10) * pW;
  const sy = y => pad.t + pH - ((y - minY) / yR) * pH;
  let tp = '', pp = '';
  preds.forEach((d, i) => { tp += `${i ? 'L' : 'M'}${sx(d.x)} ${sy(d.t)} `; pp += `${i ? 'L' : 'M'}${sx(d.x)} ${sy(d.p)} `; });
  
  return (
    <svg width={W} height={H} className="bg-slate-900 rounded-lg">
      <text x={W/2} y={18} textAnchor="middle" fill="#94a3b8" fontSize={12} fontWeight="500">Function Approximation</text>
      <line x1={pad.l} y1={pad.t + pH} x2={pad.l + pW} y2={pad.t + pH} stroke="#475569" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + pH} stroke="#475569" />
      {[-5, -2.5, 0, 2.5, 5].map(t => <text key={t} x={sx(t)} y={H - pad.b + 15} textAnchor="middle" fill="#64748b" fontSize={10}>{t}</text>)}
      <text x={W/2} y={H - 45} textAnchor="middle" fill="#94a3b8" fontSize={10}>x</text>
      {[minY, (minY+maxY)/2, maxY].map((t,i) => <text key={i} x={pad.l - 8} y={sy(t) + 4} textAnchor="end" fill="#64748b" fontSize={9}>{t.toFixed(1)}</text>)}
      <text x={15} y={pad.t + pH/2} textAnchor="middle" fill="#94a3b8" fontSize={10} transform={`rotate(-90, 15, ${pad.t + pH/2})`}>f(x)</text>
      {train.map((d, i) => <circle key={`tr${i}`} cx={sx(d.x)} cy={sy(d.y)} r={4} fill="#f97316" opacity={0.7} />)}
      {test.map((d, i) => <circle key={`te${i}`} cx={sx(d.x)} cy={sy(d.y)} r={4} fill="#22c55e" opacity={0.7} />)}
      <path d={tp} fill="none" stroke="#3b82f6" strokeWidth={2.5} />
      <path d={pp} fill="none" stroke="#a855f7" strokeWidth={2.5} />
      <g transform={`translate(${W/2 - 140}, ${H - 22})`}>
        <circle cx={0} cy={0} r={4} fill="#f97316" /><text x={10} y={4} fill="#94a3b8" fontSize={10}>Train</text>
        <circle cx={65} cy={0} r={4} fill="#22c55e" /><text x={75} y={4} fill="#94a3b8" fontSize={10}>Test</text>
        <line x1={120} y1={0} x2={135} y2={0} stroke="#3b82f6" strokeWidth={2.5} /><text x={140} y={4} fill="#94a3b8" fontSize={10}>True</text>
        <line x1={185} y1={0} x2={200} y2={0} stroke="#a855f7" strokeWidth={2.5} /><text x={205} y={4} fill="#94a3b8" fontSize={10}>Predicted</text>
      </g>
    </svg>
  );
};

const LossPlot = ({ data, lockedScale }) => {
  const W = 420, H = 180, pad = { t: 25, r: 15, b: 30, l: 55 };
  const pW = W - pad.l - pad.r, pH = H - pad.t - pad.b;
  
  if (!data.length) return (
    <svg width={W} height={H} className="bg-slate-900 rounded-lg">
      <text x={W/2} y={H/2} textAnchor="middle" fill="#64748b" fontSize={11}>Start training to see loss curve</text>
    </svg>
  );
  
  const maxE = data[data.length - 1].e || 1;
  const allL = data.flatMap(d => [d.tr, d.te]).filter(v => v != null && isFinite(v) && v > 0);
  if (!allL.length) return <svg width={W} height={H} className="bg-slate-900 rounded-lg" />;
  
  let maxL, minL;
  if (lockedScale) {
    maxL = lockedScale.max;
    minL = lockedScale.min;
  } else {
    maxL = Math.max(...allL);
    minL = Math.min(...allL, maxL * 0.001);
  }
  
  const logMax = Math.log10(Math.max(maxL, 1e-10));
  const logMin = Math.log10(Math.max(minL, 1e-10));
  const logRange = logMax - logMin || 1;
  
  // Always start x-axis at 0
  const sx = e => pad.l + (e / maxE) * pW;
  const sy = l => {
    if (!l || l <= 0) return pad.t + pH;
    const logL = Math.log10(l);
    const clamped = Math.max(logMin, Math.min(logMax, logL));
    return pad.t + pH - ((clamped - logMin) / logRange) * pH;
  };
  
  let trP = '', teP = '';
  data.forEach(d => {
    if (d.tr != null && isFinite(d.tr) && d.tr > 0) trP += `${trP ? 'L' : 'M'}${sx(d.e)} ${sy(d.tr)} `;
    if (d.te != null && isFinite(d.te) && d.te > 0) teP += `${teP ? 'L' : 'M'}${sx(d.e)} ${sy(d.te)} `;
  });
  
  const midE = Math.round(maxE / 2);
  
  return (
    <svg width={W} height={H} className="bg-slate-900 rounded-lg">
      <text x={W/2} y={14} textAnchor="middle" fill="#94a3b8" fontSize={11}>Loss (log scale)</text>
      <line x1={pad.l} y1={pad.t + pH} x2={pad.l + pW} y2={pad.t + pH} stroke="#475569" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + pH} stroke="#475569" />
      <text x={pad.l} y={H - 10} textAnchor="middle" fill="#64748b" fontSize={9}>0</text>
      {midE > 0 && midE < maxE && <text x={sx(midE)} y={H - 10} textAnchor="middle" fill="#64748b" fontSize={9}>{midE}</text>}
      <text x={pad.l + pW} y={H - 10} textAnchor="middle" fill="#64748b" fontSize={9}>{maxE}</text>
      <text x={pad.l - 5} y={pad.t + 4} textAnchor="end" fill="#64748b" fontSize={8}>{maxL.toExponential(1)}</text>
      <text x={pad.l - 5} y={pad.t + pH + 4} textAnchor="end" fill="#64748b" fontSize={8}>{minL.toExponential(1)}</text>
      <path d={trP} fill="none" stroke="#f97316" strokeWidth={2} />
      <path d={teP} fill="none" stroke="#22c55e" strokeWidth={2} />
      <g transform={`translate(${W - 95}, ${pad.t + 5})`}>
        <line x1={0} y1={6} x2={12} y2={6} stroke="#f97316" strokeWidth={2} /><text x={16} y={10} fill="#e2e8f0" fontSize={9}>Train</text>
        <line x1={50} y1={6} x2={62} y2={6} stroke="#22c55e" strokeWidth={2} /><text x={66} y={10} fill="#e2e8f0" fontSize={9}>Test</text>
      </g>
    </svg>
  );
};

export default function App() {
  const [funcExpr, setFuncExpr] = useState('sin(2*x)');
  const [lastCustom, setLastCustom] = useState('x^3');
  const [layers, setLayers] = useState([4, 4]);
  const [activation, setActivation] = useState('tanh');
  const [optimizer, setOptimizer] = useState('adam');
  const [lr, setLr] = useState(0.005);
  const [noise, setNoise] = useState(0);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [numPoints, setNumPoints] = useState(200);
  const [batchSize, setBatchSize] = useState(32);
  const [useBias, setUseBias] = useState(true);
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [trLoss, setTrLoss] = useState(null);
  const [teLoss, setTeLoss] = useState(null);
  const [lossHist, setLossHist] = useState([]);
  const [lockedScale, setLockedScale] = useState(null);
  const [preds, setPreds] = useState([]);
  const [trainData, setTrainData] = useState([]);
  const [testData, setTestData] = useState([]);
  const [weights, setWeights] = useState([]);

  const modelRef = useRef(null);
  const dataRef = useRef(null);
  const runningRef = useRef(false);
  const mountedRef = useRef(true);
  const sessionRef = useRef(0);
  const epochRef = useRef(0);
  const configRef = useRef({ funcExpr, layers, activation, optimizer, lr, noise, trainRatio, numPoints, batchSize, useBias });

  useEffect(() => { configRef.current = { funcExpr, layers, activation, optimizer, lr, noise, trainRatio, numPoints, batchSize, useBias }; });

  useEffect(() => { mountedRef.current = true; initAll(); return () => { mountedRef.current = false; cleanup(); }; }, []);

  useEffect(() => {
    const isPreset = PRESETS.some(p => p.expr === funcExpr);
    if (!isPreset && funcExpr.trim()) setLastCustom(funcExpr);
  }, [funcExpr]);

  useEffect(() => { reset(); }, [funcExpr, layers, activation, optimizer, noise, trainRatio, numPoints, batchSize, useBias]);

  const cleanup = () => {
    runningRef.current = false;
    if (dataRef.current) { ['trX', 'trY', 'teX', 'teY'].forEach(k => { try { dataRef.current[k]?.dispose(); } catch {} }); dataRef.current = null; }
    if (modelRef.current) { try { modelRef.current.dispose(); } catch {} modelRef.current = null; }
  };

  const getOptimizer = (opt, rate) => {
    switch(opt) { case 'sgd': return tf.train.sgd(rate); case 'rmsprop': return tf.train.rmsprop(rate); default: return tf.train.adam(rate); }
  };

  const initAll = () => {
    cleanup();
    sessionRef.current++;
    epochRef.current = 0;
    setLockedScale(null);
    const sid = sessionRef.current;
    const cfg = configRef.current;

    let minY = Infinity, maxY = -Infinity;
    for (let x = -5; x <= 5; x += 0.1) { const y = evalFn(cfg.funcExpr, x); if (isFinite(y)) { minY = Math.min(minY, y); maxY = Math.max(maxY, y); } }
    const yScale = Math.max(Math.abs(minY), Math.abs(maxY), 1);

    const step = 10 / cfg.numPoints;
    const pts = [];
    for (let x = -5; x <= 5; x += step) {
      const trueY = evalFn(cfg.funcExpr, x);
      const noisyY = cfg.noise > 0 ? trueY + (Math.random() - 0.5) * 2 * cfg.noise * yScale : trueY;
      pts.push({ x: Math.round(x * 1000) / 1000, y: noisyY });
    }
    
    const shuffled = [...pts];
    for (let i = shuffled.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; }

    const split = cfg.trainRatio >= 1 ? shuffled.length : Math.floor(shuffled.length * cfg.trainRatio);
    const tr = shuffled.slice(0, split), te = cfg.trainRatio >= 1 ? [] : shuffled.slice(split);
    
    setTrainData(tr); setTestData(te);

    dataRef.current = {
      trX: tf.tensor2d(tr.map(p => [p.x / 5])), trY: tf.tensor2d(tr.map(p => [p.y / yScale])),
      teX: te.length ? tf.tensor2d(te.map(p => [p.x / 5])) : null, teY: te.length ? tf.tensor2d(te.map(p => [p.y / yScale])) : null, yScale
    };

    const m = tf.sequential();
    m.add(tf.layers.dense({ units: cfg.layers[0], inputShape: [1], activation: cfg.activation, kernelInitializer: tf.initializers.glorotNormal({ seed: Date.now() % 1000 }), useBias: cfg.useBias }));
    for (let i = 1; i < cfg.layers.length; i++) m.add(tf.layers.dense({ units: cfg.layers[i], activation: cfg.activation, kernelInitializer: tf.initializers.glorotNormal({ seed: (Date.now() + i) % 1000 }), useBias: cfg.useBias }));
    m.add(tf.layers.dense({ units: 1, activation: 'linear', kernelInitializer: tf.initializers.glorotNormal({ seed: (Date.now() + 99) % 1000 }), useBias: cfg.useBias }));
    m.compile({ optimizer: getOptimizer(cfg.optimizer, cfg.lr), loss: 'meanSquaredError' });
    modelRef.current = m;

    setEpoch(0); setLossHist([]); setTrLoss(null); setTeLoss(null);
    if (sid === sessionRef.current && mountedRef.current) { updatePreds(); updateWeights(); }
  };

  const updatePreds = () => {
    if (!modelRef.current || !dataRef.current) return;
    const s = dataRef.current.yScale, cfg = configRef.current;
    const xs = []; for (let x = -5; x <= 5; x += 0.05) xs.push(x);
    const inp = tf.tensor2d(xs.map(x => [x / 5]));
    const out = modelRef.current.predict(inp).arraySync();
    inp.dispose();
    setPreds(xs.map((x, i) => ({ x: Math.round(x * 100) / 100, t: evalFn(cfg.funcExpr, x), p: out[i][0] * s })));
  };

  const updateWeights = () => {
    if (!modelRef.current) return;
    setWeights(modelRef.current.layers.map(l => { const w = l.getWeights(); return w[0] ? w[0].arraySync() : []; }));
  };

  useEffect(() => {
    runningRef.current = running;
    if (!running) return;
    const sid = sessionRef.current;

    const step = async () => {
      if (!runningRef.current || !modelRef.current || !dataRef.current || sid !== sessionRef.current) return;
      try {
        const cfg = configRef.current;
        const res = await modelRef.current.fit(dataRef.current.trX, dataRef.current.trY, { epochs: 10, batchSize: cfg.batchSize, shuffle: true, verbose: 0 });
        if (!runningRef.current || sid !== sessionRef.current || !mountedRef.current) return;

        const tl = res.history.loss[res.history.loss.length - 1];
        let vl = null;
        if (dataRef.current.teX) { const ev = modelRef.current.evaluate(dataRef.current.teX, dataRef.current.teY); vl = ev.dataSync()[0]; ev.dispose(); }

        epochRef.current += 10;
        const currentEpoch = epochRef.current;

        setTrLoss(tl); setTeLoss(vl); setEpoch(currentEpoch);
        setLossHist(h => {
          const newHist = [...h, { e: currentEpoch, tr: tl, te: vl }];
          // Lock scale after 20 epochs
          if (currentEpoch === 20 && !lockedScale) {
            const allL = newHist.flatMap(d => [d.tr, d.te]).filter(v => v != null && isFinite(v) && v > 0);
            if (allL.length) {
              setLockedScale({ max: Math.max(...allL) * 1.5, min: Math.min(...allL) * 0.1 });
            }
          }
          return newHist; // Keep all data, don't slice
        });
        updatePreds(); updateWeights();

        if (runningRef.current && sid === sessionRef.current) setTimeout(step, 30);
      } catch (e) { console.log('Training error:', e); runningRef.current = false; setRunning(false); }
    };
    step();
  }, [running, lockedScale]);

  const reset = () => { setRunning(false); runningRef.current = false; setTimeout(initAll, 50); };
  const updateLr = v => { setLr(v); if (modelRef.current) modelRef.current.compile({ optimizer: getOptimizer(configRef.current.optimizer, v), loss: 'meanSquaredError' }); };

  const isCustom = !PRESETS.some(p => p.expr === funcExpr);

  return (
    <div className="min-h-screen bg-slate-950 text-white text-sm flex flex-col">
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-700 px-4 py-3 flex items-center gap-3">
        <Brain className="text-blue-400" size={28} />
        <div>
          <h1 className="text-lg font-bold text-white">Neural Network Playground</h1>
          <p className="text-xs text-slate-400">Visualize how neural networks learn to approximate functions</p>
        </div>
      </div>

      <div className="bg-slate-900 border-b border-slate-700 px-4 py-2 flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <button onClick={() => setRunning(r => !r)} className={`p-2 rounded-lg ${running ? 'bg-amber-600' : 'bg-blue-600'}`}>
            {running ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <button onClick={reset} className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600"><RotateCcw size={16} /></button>
          <span className="text-slate-400">Epoch: <span className="text-white font-mono">{epoch}</span></span>
        </div>
        <div className="h-5 w-px bg-slate-700" />
        <span className="text-orange-400 text-xs">Train: {trLoss?.toFixed(5) ?? '—'}</span>
        <span className="text-green-400 text-xs">Test: {teLoss?.toFixed(5) ?? '—'}</span>
        <div className="h-5 w-px bg-slate-700" />
        <span className="text-slate-400 text-xs">LR:</span>
        <input type="range" min={-4} max={-1} step={0.1} value={Math.log10(lr)} onChange={e => updateLr(Math.pow(10, +e.target.value))} className="w-16 accent-blue-500" />
        <span className="font-mono text-xs w-12">{lr.toFixed(4)}</span>
        <span className="text-slate-400 text-xs">Activation:</span>
        <select value={activation} onChange={e => setActivation(e.target.value)} className="bg-slate-700 rounded px-1 py-0.5 text-xs">
          {ACTIVATIONS.map(a => <option key={a} value={a}>{a}</option>)}
        </select>
        <span className="text-slate-400 text-xs">Optimizer:</span>
        <select value={optimizer} onChange={e => setOptimizer(e.target.value)} className="bg-slate-700 rounded px-1 py-0.5 text-xs">
          {OPTIMIZERS.map(o => <option key={o} value={o}>{o.toUpperCase()}</option>)}
        </select>
      </div>

      <div className="flex flex-1">
        <div className="w-44 bg-slate-900 border-r border-slate-700 p-2 space-y-2 text-xs">
          <div>
            <label className="text-slate-400 block mb-1">Function f(x)</label>
            <input type="text" value={funcExpr} onChange={e => setFuncExpr(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs font-mono" />
            <div className="flex flex-wrap gap-1 mt-1">
              {PRESETS.map(p => (
                <button key={p.name} onClick={() => setFuncExpr(p.expr)}
                  className={`px-1.5 py-0.5 rounded text-xs ${funcExpr === p.expr ? 'bg-blue-600' : 'bg-slate-700 hover:bg-slate-600'}`}>{p.name}</button>
              ))}
              <button onClick={() => setFuncExpr(lastCustom)}
                className={`px-1.5 py-0.5 rounded text-xs ${isCustom ? 'bg-blue-600' : 'bg-slate-700 hover:bg-slate-600'}`}>custom</button>
            </div>
          </div>
          <div><label className="text-slate-400">Points: {numPoints}</label><input type="range" min={50} max={500} step={10} value={numPoints} onChange={e => setNumPoints(+e.target.value)} className="w-full accent-blue-500" /></div>
          <div><label className="text-slate-400">Noise: {noise.toFixed(2)}</label><input type="range" min={0} max={1} step={0.05} value={noise} onChange={e => setNoise(+e.target.value)} className="w-full accent-blue-500" /></div>
          <div><label className="text-slate-400">Train: {(trainRatio * 100).toFixed(0)}%</label><input type="range" min={0.5} max={1} step={0.05} value={trainRatio} onChange={e => setTrainRatio(+e.target.value)} className="w-full accent-blue-500" /></div>
          <div><label className="text-slate-400">Batch: {batchSize}</label><input type="range" min={1} max={128} step={1} value={batchSize} onChange={e => setBatchSize(+e.target.value)} className="w-full accent-blue-500" /></div>
          <div className="flex items-center gap-2"><input type="checkbox" id="bias" checked={useBias} onChange={e => setUseBias(e.target.checked)} className="accent-blue-500" /><label htmlFor="bias" className="text-slate-400">Include bias terms</label></div>
          <div className="pt-2 border-t border-slate-700">
            <div className="text-slate-400 mb-1">Weight Legend</div>
            <div className="flex items-center gap-1 mb-0.5"><span className="w-3 h-0.5 bg-blue-500 rounded" /> Positive</div>
            <div className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-500 rounded" /> Negative</div>
            <div className="text-slate-500 mt-1">Thickness = magnitude</div>
          </div>
        </div>

        <div className="flex-1 p-3 flex gap-3">
          <div className="flex flex-col gap-3">
            <NetworkViz layers={layers} weights={weights}
              onAddNeuron={i => { if (layers[i] < 8) { const nl = [...layers]; nl[i]++; setLayers(nl); } }}
              onRemoveNeuron={i => { if (layers[i] > 1) { const nl = [...layers]; nl[i]--; setLayers(nl); } }}
              onAddLayer={() => { if (layers.length < 6) setLayers([...layers, 2]); }}
              onRemoveLayer={() => { if (layers.length > 1) setLayers(layers.slice(0, -1)); }} />
            <LossPlot data={lossHist} lockedScale={lockedScale} />
          </div>
          <DataPlot train={trainData} test={testData} preds={preds} />
        </div>
      </div>

      <div className="bg-slate-900 border-t border-slate-700 px-4 py-2 flex justify-end">
        <span className="text-xs text-slate-500">
          by <a href="https://egemenokte.com" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">Egemen Okte</a>
          <span className="mx-2">·</span>
          Inspired by <a href="https://playground.tensorflow.org/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">TensorFlow Playground</a>
        </span>
      </div>
    </div>
  );
}
