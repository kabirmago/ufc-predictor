const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

let MODEL = null;
let FIGHTERS = null;
let FIGHTERS_WITH_DATA = null;
let ELO = null;

function loadData() {
  const dataPath = path.join(__dirname, '../data/dashboard_v2.json');
  const raw = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  MODEL = raw.js_model;
  FIGHTERS = raw.fighter_stats;
  ELO = raw.elo_ratings || {};
  FIGHTERS_WITH_DATA = Object.keys(FIGHTERS).filter(name => FIGHTERS[name].slpm > 0);
  console.log(`Loaded ${Object.keys(FIGHTERS).length} fighters (${FIGHTERS_WITH_DATA.length} with stats), ${Object.keys(ELO).length} ELO ratings`);
  return raw;
}

const DATA = loadData();

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function getElo(name) { return ELO[name] || 1500; }

function eloWinProb(ra, rb) { return 1 / (1 + Math.pow(10, (rb - ra) / 400)); }

function getStats(name) {
  return FIGHTERS[name] || { slpm:0, str_acc:0, sapm:0, str_def:0, td_avg:0, td_acc:0, td_def:0, sub_avg:0 };
}

const WC_MAP = {
  'Strawweight':0,'Flyweight':1,'Bantamweight':2,'Featherweight':3,
  'Lightweight':4,'Welterweight':5,'Middleweight':6,
  'Light Heavyweight':7,'Heavyweight':8,'Super Heavyweight':9
};

function computeFeatures(s1, s2, elo_f1, elo_f2, wc) {
  const ne = (s1.slpm*(s1.str_acc/100) - s1.sapm*(1-s1.str_def/100)) -
             (s2.slpm*(s2.str_acc/100) - s2.sapm*(1-s2.str_def/100));
  const ng = (s1.td_avg*(s1.td_acc/100+.01)*(s1.td_def/100+.01)) -
             (s2.td_avg*(s2.td_acc/100+.01)*(s2.td_def/100+.01));
  const dr = (s1.str_def+s1.td_def+1)/(s2.str_def+s2.td_def+1);
  const ar = Math.max(s1.sapm,0.1)/Math.max(s2.sapm,0.1);
  const sc = Math.abs((s1.striker_score||0)-(s2.striker_score||0)) +
             Math.abs((s1.wrestler_score||0)-(s2.wrestler_score||0));
  const gd = (s1.td_avg*(s1.td_acc/100+.01)*(s1.td_def/100+.01)) -
             (s2.td_avg*(s2.td_acc/100+.01)*(s2.td_def/100+.01));
  const elo_diff = elo_f1 - elo_f2;
  const elo_win_prob = eloWinProb(elo_f1, elo_f2);

  return [
    s1.slpm-s2.slpm, s1.str_acc-s2.str_acc,
    s1.sapm-s2.sapm, s1.str_def-s2.str_def,
    s1.td_avg-s2.td_avg, s1.td_acc-s2.td_acc,
    s1.td_def-s2.td_def, s1.sub_avg-s2.sub_avg,
    ne, ng, dr,
    (s1.striker_score||0)-(s2.striker_score||0),
    (s1.wrestler_score||0)-(s2.wrestler_score||0),
    (s1.ko_rate||0)-(s2.ko_rate||0),
    (s1.sub_rate||0)-(s2.sub_rate||0),
    ar, gd, sc,
    elo_diff, elo_win_prob, elo_f1,
    WC_MAP[wc] !== undefined ? WC_MAP[wc] : 4,
  ];
}

function runSurrogate(feats) {
  let logit = MODEL.intercept;
  for (let i = 0; i < MODEL.feature_order.length; i++) {
    const f = MODEL.feature_order[i];
    logit += (MODEL.coefficients[f] || 0) * (feats[i] || 0);
  }
  return sigmoid(logit);
}

// API
app.get('/api/fighters', (req, res) => {
  const q = (req.query.q || '').toLowerCase();
  const matches = q.length < 2
    ? FIGHTERS_WITH_DATA.slice(0, 20)
    : FIGHTERS_WITH_DATA.filter(n => n.toLowerCase().includes(q)).slice(0, 12);
  res.json(matches);
});

app.post('/api/predict', (req, res) => {
  const { f1, f2, weight_class } = req.body;
  if (!f1 || !f2) return res.status(400).json({ error: 'Both fighter names required' });

  const s1 = getStats(f1);
  const s2 = getStats(f2);
  const elo_f1 = getElo(f1);
  const elo_f2 = getElo(f2);
  const wc = weight_class || '';

  const feats = computeFeatures(s1, s2, elo_f1, elo_f2, wc);

  // Algo probability from JS surrogate
  const algo_prob = runSurrogate(feats);

  // ELO probability
  const elo_prob = eloWinProb(elo_f1, elo_f2);

  // Blended: 60% algo + 40% ELO
  // ELO anchors extreme predictions to prevent overconfidence
  const f1prob = 0.6 * algo_prob + 0.4 * elo_prob;
  const f2prob = 1 - f1prob;

  const winner = f1prob > 0.5 ? f1 : f2;
  const conf = Math.max(f1prob, f2prob) > 0.68 ? 'high'
             : Math.max(f1prob, f2prob) > 0.57 ? 'medium' : 'low';

  // SHAP from surrogate coefficients
  const coeffs = MODEL.coefficients;
  const featureNames = MODEL.feature_order;
  const shap = featureNames
    .map((f, i) => ({ feature: f, contrib: (coeffs[f] || 0) * (feats[i] || 0) }))
    .filter(x => Math.abs(x.contrib) > 0.001)
    .sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib))
    .slice(0, 8);

  res.json({
    f1_name: f1, f2_name: f2,
    f1_win_prob: Math.round(f1prob * 1000) / 1000,
    f2_win_prob: Math.round(f2prob * 1000) / 1000,
    predicted_winner: winner,
    confidence: conf,
    algo_prob: Math.round(algo_prob * 1000) / 1000,
    elo_prob: Math.round(elo_prob * 1000) / 1000,
    elo_f1: Math.round(elo_f1),
    elo_f2: Math.round(elo_f2),
    shap_breakdown: shap,
    stats: [
      { label: 'Sig. strikes/min', f1: s1.slpm,    f2: s2.slpm,    unit: '' },
      { label: 'Strike accuracy',  f1: s1.str_acc,  f2: s2.str_acc, unit: '%' },
      { label: 'Strike defense',   f1: s1.str_def,  f2: s2.str_def, unit: '%' },
      { label: 'Takedown avg',     f1: s1.td_avg,   f2: s2.td_avg,  unit: '/15m' },
      { label: 'Takedown defense', f1: s1.td_def,   f2: s2.td_def,  unit: '%' },
      { label: 'Submission avg',   f1: s1.sub_avg,  f2: s2.sub_avg, unit: '/15m' },
    ].filter(r => r.f1 !== 0 || r.f2 !== 0),
    model_accuracy: DATA.model_accuracy,
    n_fights_trained: DATA.n_fights_trained,
  });
});

// Old landing at /landing
app.get('/landing', (req, res) => {
  res.sendFile(path.join(__dirname, '../landing.html'));
});

// React app — all routes
app.use(express.static(path.join(__dirname, '../frontend/dist')));
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/dist/index.html'));
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`UFC Predictor running on port ${PORT}`));
