# ML Entry Filter - Retrain Log

Each entry is one scheduled retrain run (see `.github/workflows/retrain_ml_filter.yml` and `retrain_pipeline.py`). A skipped run keeps the previously deployed model - retraining never deploys a model that fails its own out-of-sample floor.

## 2026-09-02T23:01:41.199310+00:00

- Holdout window: 2025-12-06 to 2026-09-02 (270 days, rolling)
- Trained on 40876 rows before the holdout
- Holdout result at threshold 0.55: 53 trades, win rate 41.5%, PF 0.766, CAGR -3.78%, max DD -5.91%
- **SKIPPED**: PF 0.766 below floor 0.9 on its own holdout - kept existing model

## 2026-09-02T23:05:03.576412+00:00

- Holdout window: 2025-12-06 to 2026-09-02 (270 days, rolling)
- Trained on 40876 rows before the holdout
- Holdout result at threshold 0.55: 53 trades, win rate 41.5%, PF 0.766, CAGR -3.78%, max DD -5.91%
- **SKIPPED**: PF 0.766 below floor 0.9 on its own holdout - kept existing model

