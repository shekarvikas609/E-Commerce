# Second Order: Olist customer segments, retention and 30-day demand outlook

RFM segmentation, cohort retention, a 90-day repurchase (churn-risk) model and 30-day demand
forecasts, in one Streamlit dashboard. Built on the public Olist Brazilian E-commerce dataset.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Processed results are in `data/processed`, so the app opens immediately. To rebuild from the raw
CSVs (put the Kaggle `olistbr/brazilian-ecommerce` files in `data/raw`):

```bash
python run_pipeline.py        # about 30 seconds
```

If `data/raw` has no Olist CSVs, the pipeline generates schema-identical demo data instead and the
dashboard shows a banner saying so.

## What is inside

| File | Job |
|---|---|
| `src/data.py` | Load tables, merge same-day orders, trim the sparse start and tail of the export |
| `src/rfm.py` | R and M by quintile, F by fixed steps (97% of customers order once), 8 named segments, playbook |
| `src/cohorts.py` | Monthly cohort retention, repeat-purchase curve, time to second order |
| `src/churn.py` | XGBoost: will the customer order again in 90 days? Time-based validation |
| `src/forecast.py` | Seasonal naive vs Prophet vs XGBoost vs blend, rolling-origin backtest |
| `src/pipeline.py` | Runs everything, writes `data/processed` |
| `app.py`, `src/ui.py` | Dashboard |

## Results on the real data (Jan 2017 to Aug 2018)

- 98% of customers ordered once. Big spenders and lapsed high-value buyers hold about 70% of revenue.
- Repurchase model: AUC about 0.60, top-decile lift about 2.3x. Modest signal; use it to rank, not predict.
- Order forecast: blend beats seasonal naive by about 20% on daily error (WAPE about 20%).

## Design decisions worth knowing

- **Churn is defined, not observed.** No order in the next 90 days. The dashboard shows a relative risk index plus the absolute repurchase chance.
- **Segments are not forecast as time series.** A segment is defined by recent activity, so slicing history by today's segments leaks the future. Segment outlook comes from the repurchase model instead.
- **Backtests only train on earlier data**, three 30-day folds, best model chosen per series.
