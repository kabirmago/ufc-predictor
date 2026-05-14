const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend/dist')));

let MODEL = null;
let FIGHTERS = null;
let FIGHTERS_WITH_DATA = null;

function loadData() {
  const dataPath = path.join(__dirname, '../data/dashboard_v2.json');
  const raw = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  MODEL = raw.js_model;
  FIGHTERS = raw.fighter_stats;
  // Only expose fighters with real slpm data in autocomplete
  FIGHTERS_WITH_DATA = Object.keys(FIGHTERS).filter(name => FIGHTERS[name].slpm > 0);
  console.log(`Loaded ${Object.keys(FIGHTERS).length} fighters (${FIGHTERS_WITH_DATA.length} with real data)`);
  return raw;
}

const DATA = loadData();

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function getStats(name) {
  return FIGHTERS[name] || { slpm: 0, str_acc: 0, sapm: 0, str_def: 0, td_avg: 0, td_acc: 0, td_def: 0, sub_avg: 0 };
}

function computeFeatures(s1, s2) {
  const slpm_diff    = s1.slpm    - s2.slpm;
  const str_acc_diff = s1.str_acc - s2.str_acc;
  const sapm_diff    = s1.sapm    - s2.sapm;
  const str_def_diff = s1.str_def - s2.str_def;
  const td_diff      = s1.td_avg  - s2.td_avg;
  const td_acc_diff  = s1.td_acc  - s2.td_acc;
  const td_def_diff  = s1.td_def  - s2.td_def;
  const sub_diff     = s1.sub_avg - s2.sub_avg;
  const net_strike   = slpm_diff * (s1.str_acc / 100) - sapm_diff * (1 - s1.str_def / 100);
  const net_grapple  = td_diff * (s1.td_acc / 100) + td_def_diff / 100;
  const style_mm     = (s1.slpm > 4 && s1.td_avg < 1 && s2.td_avg > 2 && s2.slpm < 3.5) ? 1 : 0;
  const def_ratio    = s2.str_def ? s1.str_def / s2.str_def : 1.0;
  return { slpm_diff, str_acc_diff, sapm_diff, str_def_diff, td_diff, td_acc_diff, td_def_diff, sub_diff, net_strike_edge: net_strike, net_grapple_edge: net_grapple, style_mismatch: style_mm, def_ratio };
}

function runModel(feats) {
  let logit = MODEL.intercept;
  for (const f of MODEL.feature_order) {
    logit += (MODEL.coefficients[f] || 0) * (feats[f] || 0);
  }
  return sigmoid(logit);
}

app.get('/api/fighters', (req, res) => {
  const q = (req.query.q || '').toLowerCase();
  // Only search fighters with real slpm data
  const matches = q.length < 2
    ? FIGHTERS_WITH_DATA.slice(0, 20)
    : FIGHTERS_WITH_DATA.filter(n => n.toLowerCase().includes(q)).slice(0, 12);
  res.json(matches);
});

app.post('/api/predict', (req, res) => {
  const { f1, f2 } = req.body;
  if (!f1 || !f2) return res.status(400).json({ error: 'Both fighter names required' });

  const s1 = getStats(f1);
  const s2 = getStats(f2);
  const feats = computeFeatures(s1, s2);
  const f1prob = runModel(feats);
  const f2prob = 1 - f1prob;
  const winner = f1prob > 0.5 ? f1 : f2;
  const conf   = Math.max(f1prob, f2prob) > 0.70 ? 'high' : Math.max(f1prob, f2prob) > 0.58 ? 'medium' : 'low';

  const coeffs = MODEL.coefficients;
  const shap = MODEL.feature_order
    .map(f => ({ feature: f, contrib: (coeffs[f] || 0) * (feats[f] || 0) }))
    .filter(x => Math.abs(x.contrib) > 0.001)
    .sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));

  res.json({
    f1_name: f1, f2_name: f2,
    f1_win_prob: Math.round(f1prob * 1000) / 1000,
    f2_win_prob: Math.round(f2prob * 1000) / 1000,
    predicted_winner: winner,
    confidence: conf,
    shap_breakdown: shap,
    stats: [
      { label: 'Sig. strikes/min', f1: s1.slpm,    f2: s2.slpm,    unit: '' },
      { label: 'Strike accuracy',  f1: s1.str_acc,  f2: s2.str_acc, unit: '%' },
      { label: 'Strike defense',   f1: s1.str_def,  f2: s2.str_def, unit: '%' },
      { label: 'Takedown avg',     f1: s1.td_avg,   f2: s2.td_avg,  unit: '/15m' },
      { label: 'Takedown defense', f1: s1.td_def,   f2: s2.td_def,  unit: '%' },
      { label: 'Submission avg',   f1: s1.sub_avg,  f2: s2.sub_avg, unit: '/15m' },
    ].filter(r => r.f1 !== 0 || r.f2 !== 0),
    global_shap: DATA.global_shap,
    reliability: DATA.reliability,
    model_accuracy: DATA.model_accuracy,
    n_fights_trained: DATA.n_fights_trained,
  });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/dist/index.html'));
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`UFC Predictor running on port ${PORT}`));
