# UFC Fight Predictor

XGBoost model trained on 11,280 UFC fights. Type two fighter names. Get a prediction in seconds.

**68.4% cross-validated accuracy** — beats a pure defense heuristic baseline by 7.3 percentage points.

---

## What it does

- Scrapes ufcstats.com for fighter stats and fight history (2015–2025)
- Engineers 12 features from raw stats — striking differentials, grappling edge, defensive ratios
- Trains XGBoost with Optuna hyperparameter tuning across 5-fold cross-validation
- Outputs win probabilities with SHAP explanations for every prediction
- Serves a React frontend with live fuzzy fighter search across 4,450 fighters

---

## Model findings

The most interesting thing the model learned: **striking volume barely matters**.

`td_diff` and `def_ratio` dominate everything. `slpm_diff`, `str_acc_diff`, and `sapm_diff` all have near-zero SHAP importance. Once defense is accounted for, how hard a fighter throws doesn't predict outcomes. Grappling and who absorbs less does.

![SHAP feature importance](assets/shap_importance.png)

The reliability curve shows the model's probabilities are trustworthy — when it says 70%, the fighter wins ~70% of the time.

![Reliability curve](assets/reliability_curve.png)

---

## Validation

| Metric | Result |
|---|---|
| CV accuracy (5-fold) | **68.4%** |
| Best baseline (defense heuristic) | 61.0% |
| Margin over baseline | **+7.3%** |
| Brier score | 0.1958 |
| Leakage drop (time split) | 3.3% |
| Probability range | 0.077 – 0.944 |
| Fights trained on | 11,280 |
| Fighters in database | 4,450 |

---

## Stack

| Layer | Tech |
|---|---|
| Scraping | BeautifulSoup + ufcstats.com |
| Model | XGBoost + Optuna |
| Explainability | SHAP |
| Backend | Express.js |
| Frontend | React + Vite + Framer Motion |
| Deploy | Railway |

---

## Run locally

```bash
# 1. Install and build
npm install
cd frontend && npm install && npm run build && cd ..

# 2. Start
npm start
# → http://localhost:3001
```

---

## Retrain

Open `ufc_predictor.ipynb` in Google Colab. Run all cells. Download `dashboard_v2.json`. Replace `data/dashboard_v2.json`. Redeploy.

Scraping takes ~13 minutes the first time. After that, predictions run in under 2 seconds from the saved model.

---

## API

```
GET  /api/fighters?q=makh     → fuzzy search, returns array of names
POST /api/predict              → { f1, f2 } → win probs + SHAP + stats
```

---

*Built by a Vanderbilt CS/math student who wanted to combine MMA with ML.*
