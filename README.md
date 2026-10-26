# Hybrid Movie Recommendation System (MovieLens 1M)

Hybrid recommender (collaborative filtering + content-based) with proper ranking evaluation and a Streamlit demo.

## Status
- [x] Week 1 – EDA, data loading, leakage-free split
- [x] Week 2 – Baselines (global mean, popularity, damped mean, bias) + evaluation metrics
- [ ] Week 3 – Collaborative filtering (user/item kNN, SVD)
- [ ] Week 4 – Content-based + hybrid + cold-start
- [ ] Week 5 – Evaluation, tuning, comparison table
- [ ] Week 6 – Streamlit app, README results, demo

## Setup
```bash
pip install -r requirements.txt
# put the extracted dataset here:  data/raw/ml-1m/{ratings,movies,users}.dat
jupyter notebook notebooks/01_eda.ipynb
```

## Key findings from EDA
| Finding | Number | Implication |
|---|---|---|
| Sparsity | 95.53% | need matrix factorization / neighbourhood methods |
| Top 20% movies | 65.2% of ratings | strong popularity baseline; measure coverage & novelty too |
| Global mean rating | 3.582 | user/item bias terms matter |
| Movies with <10 ratings | 12% | cold-start items; min-count / damped mean for rankings |
| Ratings in year 2000 | 90.5% | use per-user temporal split, not random |

## Evaluation protocol
Per-user temporal split: each user's latest 5 ratings -> test, rest -> train (970,009 / 30,200). No leakage (verified in notebook).
Metrics: RMSE, MAE, Precision@K, Recall@K, MAP, NDCG@K, coverage, diversity, novelty.

## Structure
```
data/raw/ml-1m/   dataset (not committed)
notebooks/        01_eda.ipynb ...
src/data.py       loading + splitting + sparse matrix
src/models/       models (coming in later weeks)
app/              Streamlit app
reports/figures/  EDA plots
```
