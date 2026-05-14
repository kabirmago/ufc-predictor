import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip } from 'chart.js';
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip);

const FEAT_LABELS = {
  slpm_diff: 'Strike output', str_acc_diff: 'Strike accuracy', sapm_diff: 'Strikes absorbed',
  str_def_diff: 'Strike defense', td_diff: 'Takedown diff', td_acc_diff: 'Takedown accuracy',
  td_def_diff: 'Takedown defense', sub_diff: 'Submission threat',
  net_strike_edge: 'Net strike edge', net_grapple_edge: 'Net grapple edge',
  style_mismatch: 'Style mismatch', def_ratio: 'Defense ratio'
};

const ease = [0.22, 1, 0.36, 1];
const fade = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0 } };

function FighterInput({ label, color, value, onChange, onSelect }) {
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const search = useCallback(async (q) => {
    onChange(q);
    if (q.length < 2) { setResults([]); setOpen(false); return; }
    const res = await fetch(`/api/fighters?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    setResults(data);
    setOpen(data.length > 0);
  }, [onChange]);

  const pick = (name) => { onSelect(name); onChange(name); setOpen(false); setResults([]); };

  return (
    <div ref={ref} style={{ position: 'relative', flex: 1 }}>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
        {label}
        <span style={{ padding: '1px 8px', borderRadius: 20, fontSize: 10, fontWeight: 600, background: color === 'blue' ? 'rgba(55,138,221,0.15)' : 'rgba(216,90,48,0.15)', color: color === 'blue' ? 'var(--blue)' : 'var(--red)' }}>
          {color === 'blue' ? 'BLUE' : 'RED'}
        </span>
      </div>
      <input
        value={value}
        onChange={e => search(e.target.value)}
        onFocus={() => value.length >= 2 && setOpen(results.length > 0)}
        placeholder="Type fighter name..."
        style={{
          width: '100%', padding: '12px 16px',
          background: 'var(--surface2)', border: `1px solid ${open ? (color === 'blue' ? 'var(--blue)' : 'var(--red)') : 'var(--border)'}`,
          borderRadius: 10, color: 'var(--fg)', fontSize: 14,
          fontFamily: 'Space Grotesk, sans-serif', outline: 'none',
          transition: 'border-color 0.2s'
        }}
      />
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            style={{
              position: 'absolute', top: '100%', left: 0, right: 0, marginTop: 4,
              background: 'var(--surface2)', border: '1px solid var(--border)',
              borderRadius: 10, zIndex: 50, overflow: 'hidden'
            }}
          >
            {results.map(name => (
              <div
                key={name}
                onMouseDown={() => pick(name)}
                style={{
                  padding: '10px 16px', fontSize: 13, cursor: 'pointer',
                  borderBottom: '1px solid var(--border)', transition: 'background 0.1s'
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--surface)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                {name}
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ProBBar: f1 is always BLUE (left), f2 is always RED (right)
// prob = f1 win probability (0..1)
function ProbBar({ f1, f2, prob }) {
  const pct = Math.round(prob * 100);
  const f1Last = f1.split(' ').slice(-1)[0];
  const f2Last = f2.split(' ').slice(-1)[0];
  return (
    <div style={{ margin: '24px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 13 }}>
        <span style={{ color: 'var(--blue)', fontWeight: 600 }}>{f1Last}</span>
        <span style={{ color: 'var(--muted)', fontSize: 11 }}>win probability</span>
        <span style={{ color: 'var(--red)', fontWeight: 600 }}>{f2Last}</span>
      </div>
      <div style={{ height: 22, background: 'var(--surface2)', borderRadius: 11, overflow: 'hidden', display: 'flex' }}>
        <motion.div
          initial={{ width: 0 }} animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease }}
          style={{ background: 'var(--blue)', height: '100%' }}
        />
        <motion.div
          initial={{ width: 0 }} animate={{ width: `${100 - pct}%` }}
          transition={{ duration: 0.9, ease }}
          style={{ background: 'var(--red)', height: '100%' }}
        />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 13, fontWeight: 600 }}>
        <span style={{ color: 'var(--blue)' }}>{pct}%</span>
        <span style={{ color: 'var(--red)' }}>{100 - pct}%</span>
      </div>
    </div>
  );
}

// ShapBar: positive contrib = favors F1 (blue), negative = favors F2 (red)
// This matches the model: SHAP values are computed from F1's perspective
function ShapBar({ item, f1name, f2name }) {
  const favorsF1 = item.contrib > 0;
  const label = FEAT_LABELS[item.feature] || item.feature;
  const favoredName = favorsF1 ? f1name.split(' ').slice(-1)[0] : f2name.split(' ').slice(-1)[0];
  const barColor = favorsF1 ? 'var(--blue)' : 'var(--red)';
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: 'var(--fg)' }}>{label}</span>
        <span style={{ color: 'var(--muted)', fontFamily: 'Fira Code, monospace', fontSize: 11 }}>
          {item.contrib > 0 ? '+' : ''}{item.contrib.toFixed(3)} · {favoredName}
        </span>
      </div>
      <div style={{ height: 6, background: 'var(--surface2)', borderRadius: 3, overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(Math.abs(item.contrib) * 80, 100)}%` }}
          transition={{ duration: 0.7, ease }}
          style={{ height: '100%', background: barColor, borderRadius: 3 }}
        />
      </div>
    </div>
  );
}

function ReliabilityChart({ data }) {
  const chartData = {
    labels: data.map(p => `${Math.round(p.predicted * 100)}%`),
    datasets: [
      { label: 'Model', data: data.map(p => p.actual), borderColor: '#378ADD', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#378ADD', tension: 0.2 },
      { label: 'Perfect', data: data.map(p => p.predicted), borderColor: 'rgba(255,255,255,0.2)', borderWidth: 1, borderDash: [4, 4], pointRadius: 0, tension: 0 }
    ]
  };
  return (
    <div style={{ height: 180 }}>
      <Line data={chartData} options={{
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888', font: { size: 10 } } },
          y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888', font: { size: 10 }, callback: v => `${Math.round(v * 100)}%` } }
        }
      }} />
    </div>
  );
}

export default function App() {
  const [f1, setF1] = useState('');
  const [f2, setF2] = useState('');
  const [f1val, setF1val] = useState('');
  const [f2val, setF2val] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const predict = async () => {
    if (!f1 || !f2) { setError('Select both fighters'); return; }
    if (f1 === f2) { setError('Select two different fighters'); return; }
    setError(''); setLoading(true); setResult(null);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ f1, f2 })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const container = { hidden: {}, show: { transition: { staggerChildren: 0.08 } } };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <nav style={{
        padding: '20px 40px', display: 'flex', alignItems: 'center',
        borderBottom: '1px solid var(--border)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 28, height: 28, background: 'var(--accent)', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontSize: 14, color: '#000', fontWeight: 700 }}>⬡</span>
          </div>
          <span style={{ fontWeight: 600, fontSize: 15, letterSpacing: '-0.01em' }}>UFC Predictor</span>
        </div>
      </nav>

      <main style={{ maxWidth: 760, margin: '0 auto', padding: '60px 24px' }}>
        <motion.div initial="hidden" animate="show" variants={container}>
          <motion.div variants={fade} transition={{ duration: 0.7, ease }}>
            <p style={{ color: 'var(--accent)', fontSize: 11, letterSpacing: '0.15em', marginBottom: 12 }}>ML-POWERED · XGBOOST · 68.4% CV ACCURACY</p>
            <h1 style={{ fontSize: 'clamp(2.4rem, 6vw, 4rem)', lineHeight: 1.05, fontWeight: 700, marginBottom: 16, letterSpacing: '-0.02em' }}>
              Who wins<br />the fight?
            </h1>
            <p style={{ color: 'var(--muted)', fontSize: 15, maxWidth: 420, marginBottom: 48, lineHeight: 1.7 }}>
              Search any UFC fighter. The model analyzes striking, grappling, and defense differentials trained on 11,280 fights.
            </p>
          </motion.div>

          <motion.div variants={fade} transition={{ duration: 0.7, ease, delay: 0.1 }}
            style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16, padding: 28, marginBottom: 20 }}
          >
            <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
              <FighterInput label="Fighter 1" color="blue" value={f1val} onChange={v => { setF1val(v); }} onSelect={v => setF1(v)} />
              <div style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: 12 }}>
                <span style={{ color: 'var(--muted)', fontSize: 13, fontWeight: 600 }}>VS</span>
              </div>
              <FighterInput label="Fighter 2" color="red" value={f2val} onChange={v => { setF2val(v); }} onSelect={v => setF2(v)} />
            </div>
            {error && <p style={{ color: 'var(--red)', fontSize: 13, marginBottom: 16 }}>{error}</p>}
            <button
              onClick={predict} disabled={loading}
              style={{
                width: '100%', padding: '14px 24px',
                background: loading ? 'var(--surface2)' : 'var(--accent)',
                color: loading ? 'var(--muted)' : '#000',
                border: 'none', borderRadius: 10, fontSize: 14, fontWeight: 700,
                fontFamily: 'Space Grotesk, sans-serif', cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s', letterSpacing: '0.01em'
              }}
            >
              {loading ? 'Analyzing...' : 'Predict winner →'}
            </button>
          </motion.div>
        </motion.div>

        <AnimatePresence>
          {result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.6, ease }}
            >
              {/* Winner card */}
              <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16, padding: 28, marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
                  <div>
                    <p style={{ fontSize: 11, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 4 }}>PREDICTED WINNER</p>
                    <h2 style={{ fontSize: 'clamp(1.6rem, 4vw, 2.4rem)', fontWeight: 700, letterSpacing: '-0.02em', color: result.predicted_winner === result.f1_name ? 'var(--blue)' : 'var(--red)' }}>
                      {result.predicted_winner}
                    </h2>
                  </div>
                  <span style={{
                    padding: '6px 14px', borderRadius: 20, fontSize: 11, fontWeight: 600,
                    background: result.confidence === 'high' ? 'rgba(74,222,128,0.12)' : result.confidence === 'medium' ? 'rgba(232,255,90,0.1)' : 'rgba(255,255,255,0.06)',
                    color: result.confidence === 'high' ? 'var(--green)' : result.confidence === 'medium' ? 'var(--accent)' : 'var(--muted)',
                    border: `1px solid currentColor`
                  }}>
                    {result.confidence.toUpperCase()} CONFIDENCE
                  </span>
                </div>
                <ProbBar f1={result.f1_name} f2={result.f2_name} prob={result.f1_win_prob} />
              </div>

              {/* SHAP breakdown */}
              {result.shap_breakdown?.length > 0 && (
                <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16, padding: 28, marginBottom: 16 }}>
                  <p style={{ fontSize: 11, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 20 }}>SHAP FEATURE BREAKDOWN</p>
                  {result.shap_breakdown.map(item => <ShapBar key={item.feature} item={item} f1name={result.f1_name} f2name={result.f2_name} />)}
                  <div style={{ display: 'flex', gap: 16, marginTop: 16, fontSize: 11, color: 'var(--muted)' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--blue)', display: 'inline-block' }} />Favors {result.f1_name.split(' ').slice(-1)[0]} (F1 · blue)</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--red)', display: 'inline-block' }} />Favors {result.f2_name.split(' ').slice(-1)[0]} (F2 · red)</span>
                  </div>
                </div>
              )}

              {/* Stats */}
              {result.stats?.length > 0 && (
                <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16, padding: 28, marginBottom: 16 }}>
                  <p style={{ fontSize: 11, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 20 }}>STAT COMPARISON</p>
                  {result.stats.map(s => {
                    const total = (s.f1 + s.f2) || 1;
                    const p1 = Math.round(s.f1 / total * 100);
                    return (
                      <div key={s.label} style={{ marginBottom: 16 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 5 }}>
                          <span style={{ fontWeight: 600, color: 'var(--blue)', fontFamily: 'Fira Code, monospace' }}>{s.f1}{s.unit}</span>
                          <span style={{ color: 'var(--muted)' }}>{s.label}</span>
                          <span style={{ fontWeight: 600, color: 'var(--red)', fontFamily: 'Fira Code, monospace' }}>{s.f2}{s.unit}</span>
                        </div>
                        <div style={{ height: 6, background: 'var(--surface2)', borderRadius: 3, overflow: 'hidden', display: 'flex' }}>
                          <motion.div initial={{ width: 0 }} animate={{ width: `${p1}%` }} transition={{ duration: 0.8, ease }} style={{ background: 'var(--blue)' }} />
                          <motion.div initial={{ width: 0 }} animate={{ width: `${100 - p1}%` }} transition={{ duration: 0.8, ease }} style={{ background: 'var(--red)' }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Reliability curve */}
              {result.reliability && (
                <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16, padding: 28 }}>
                  <p style={{ fontSize: 11, color: 'var(--muted)', letterSpacing: '0.1em', marginBottom: 20 }}>MODEL RELIABILITY CURVE</p>
                  <ReliabilityChart data={result.reliability} />
                  <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 12 }}>Model closely tracks perfect calibration — probabilities are trustworthy.</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
