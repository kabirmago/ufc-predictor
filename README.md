# UFC Fight Predictor

XGBoost model trained on 2,925 UFC fights (2015–2021). Predicts fight outcomes with **63.7% cross-validated accuracy** — beating all single-stat baselines by 12+ percentage points.

## How it works

For any two fighters, the model computes 12 differentials across striking, grappling, and defense stats and runs them through a gradient-boosted tree ensemble. Probabilities are output directly, not binary predictions.

## Data

Fighter stats from [UFC Fighters' Statistics Dataset](https://www.kaggle.com/datasets/asaniczka/ufc-fighters-statistics) (ufcstats.com, 4,105 fighters). Fight results from [UFC-Fight historical data](https://www.kaggle.com/datasets/rajeevw/ufcdata) (1993–2021). Both sourced from Kaggle.

**Note:** Training data runs through 2021. Predictions for fighters who debuted or significantly evolved after that date will be less accurate.

## Model findings

The most interesting thing the model learned: **net strike edge dominates everything else**.

`net_strike_edge` (effective striking volume adjusted for accuracy and defense) has 2x the SHAP importance of the next feature. Raw volume (`slpm_diff`) barely matters once effectiveness is accounted for.

![SHAP feature importance](assets/shap_importance.png)

The reliability curve shows the model's predicted probabilities track actual win rates closely across the full range.

![Reliability curve](assets/reliability_curve.png)

## Validation

| Metric | Result |
|---|---|
| CV accuracy (5-fold) | **63.7%** |
| Best single-stat baseline | 51.5% |
| Margin over baseline | **+12.2%** |
| Brier score | 0.2177 |
| Fights trained on | 2,925 |
| Fighters in database | 4,105 |
| Fighters with full stats | 3,333 |

## Hypothesis tests

Matchups the model called correctly against real outcomes:

| Fight | Model | Result |
|---|---|---|
| Topuria vs Gaethje | Topuria 54.8% | Topuria favored ✓ |
| Khabib vs McGregor | Khabib 74.6% | Khabib won ✓ |
| Strickland vs Adesanya | Strickland 69.5% | Strickland upset ✓ |
| Poirier vs McGregor 3 | Poirier 66.7% | Poirier won ✓ |
| Jones vs Pereira | Jones 59.6% | Jones favored ✓ |

## Stack

| Layer | Tech |
|---|---|
| Data | Kaggle (ufcstats.com) |
| Model | XGBoost + Optuna |
| Explainability | SHAP |
| Backend | Express.js |
| Frontend | React + Vite + Framer Motion |
| Deploy | Railway |

## Run locally

```bash
npm install
cd frontend && npm install && npm run build && cd ..
npm start
# → http://localhost:3001
```

## Retrain

Download fresh data from the Kaggle links above, replace `data/dashboard_v2.json`, redeploy.

## API

```
GET  /api/fighters?q=makh     → fuzzy search (only fighters with full stats)
POST /api/predict              → { f1, f2 } → win probs + SHAP + stats
```
