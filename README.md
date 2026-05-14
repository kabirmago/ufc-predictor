# UFC Fight Predictor

XGBoost ML model trained on 11,280 UFC fights. Predicts fight outcomes with 68.4% cross-validated accuracy.

## Stack
- **Model**: XGBoost trained in Google Colab, exported as logistic approximation
- **Backend**: Express.js serving predictions from `data/dashboard_v2.json`
- **Frontend**: React + Vite + Framer Motion
- **Deploy**: Railway (monorepo)

## Setup

### 1. Add your model data
Copy `dashboard_v2.json` from Colab into `data/dashboard_v2.json`.

### 2. Run locally
```bash
# Build frontend
cd frontend && npm install && npm run build && cd ..

# Start backend
cd backend && npm install && node server.js
```

### 3. Deploy to Railway
1. Push to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Select `kabirmago/ufc-predictor`
4. Railway auto-detects config from `railway.json`
5. Add env var if needed: `PORT=3001`

## Data
To retrain: open `ufc_predictor.ipynb` in Google Colab, run all cells, download `dashboard_v2.json`, replace `data/dashboard_v2.json`, redeploy.

## API
- `GET /api/fighters?q=makh` — fuzzy search fighter names
- `POST /api/predict` `{ f1: "Islam Makhachev", f2: "Charles Oliveira" }` — run prediction
