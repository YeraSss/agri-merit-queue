# AgriMerit Queue … Kazakhstan Agricultural Subsidy System
### Decentrathon 5.0 | Government Case 2b | Stage 1

> Replacing first-come-first-served subsidy allocation with a transparent, merit-based queue — backed by LightGBM and explainable scoring.

---

## The Problem

Kazakhstan's agricultural subsidy system pays 7,615 approved farmers in the order they submitted their applications. When regional budgets run out, legitimate farms wait indefinitely — not because their applications are weak, but because they clicked "submit" one day too late.

**The result:** 42.34 billion tenge frozen across 7,615 applications as of March 2026. Almaty oblast: 14.3B frozen, 0 tenge paid. Zhambyl oblast: 6.3B frozen, 0 tenge paid.

---

## Our Solution

A merit-based queue replacement system that re-ranks the frozen backlog by farm development potential, not submission timestamp — while fully preserving the legal requirement that humans approve all payment decisions.

**Hybrid scoring formula:**
```
hybrid_merit_score = 0.60 × ml_merit_score + 0.40 × rule_merit_score

ml_merit_score   = (1 − P(rejection)) × 100   <- LightGBM, AUC = 0.8235
rule_merit_score = 0.40 × farm_size_score
                 + 0.30 × region_health_score
                 + 0.30 × livestock_priority_score
```

**Key SHAP finding:** The top 2 drivers of rejection risk are *region historical rejection rate* and *day of year* — both entirely outside farmer control. 65% of rejection risk is structural inequality, not farm quality.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset
cp /path/to/subsidies_2025.xlsx data/

# 3. Train model (generates artifacts/)
python train.py

# 4. Start API
uvicorm main:app --reload --port 8000

# 5. View docs
open http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|-------|---------|-------------|
| GET | `/api/health` | Dataset stats + model info |
| GET | `/api/budget/regions` | Regional crisis dashboard |
| GET | `/api/queue?mode=merit\|fcfs` | Ranked backlog (FCFS vs Merit) |
| GET | `/api/queue/{app_id}/score` | Score card + SHAP breakdown |
| POST | `/api/queue/reorder` | Live fairness slider reorder |
| GET | `/api/forecast/regional` | 3-month demand forecast |

---

## Project Structure

```
agri-merit-queue/
├── README.md
├─"�� requirements.txt
├─"�� .gitignore
├─"�� scoring_ml.py        <- Core: hybrid ML+rule merit scorer
├── train.py             <- Training pipeline (run once)
├── main.py              <- FastAPI `upplication
├─"�� data_loader.py        <- Data preprocessing
├─"�� requirements.txt
├─"�� artifacts/           <- Trained model + lookup tables (generated)
├── data/                <- Place dataset here (not committed)
└── tests/
     └── test_scoring.py  <- 23 unit tests
```

---

## Model Details

- **Algorithm:** LightGBM classifier (300 estimators, max_depth=6)
- **Task:** Predict P(rejection) → invert to merit signal
- **AUC (5-fold CV):** 0.8235 ± 0.0042
- **Training set:** 34,390 applications (2025-2026)
- **Features:** 15 engineered features (region stats, farm scale, timing, livestock type)

### Top SHAP Drivers
| Feature | Mean |SHAP| | Meaning |
|--------|------------|--------|
| Region rejection rate | 1.016 | Structural regional inequality |
| Day of year | 0.337 | First-come-first-served timing bias |
| Subsidy norm | 0.327 | Subsidy type friction |
| Log implied head count | 0.173 | Farm scale signal |
| Livestock rejection rate | 0.151 | Sector-level patterns |

---

## Legal Compliance

This system **never auto-rejects**. It ranks and flags only. All payment orders require a human MIO official signature (per Rules paragraphs 21–24). The fairness slider lets committees tune small-farm protection. Every score, flag, and override is logged.

---

## Team

Built for **Decentrathon 5.0** —  3-day sprint, Stage 1 submission.
Dataset: 36,651 anonymized records from subsidy.plem.kz (2025-2026).
