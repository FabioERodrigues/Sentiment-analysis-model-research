# Intelligent Social Media Analysis with Machine Learning

**Final Year Project — Fabio Rodrigues**

---

## Project overview

This repository contains the work and experiments for the final‑year project: **Intelligent Social Media Analysis with Machine Learning**. The goal of the project is to **compare traditional NLP approaches (TF‑IDF + classical classifiers) against a deep‑learning transfer‑learning approach (BERT)**, quantify their performance, and evaluate the tradeoffs a business must consider (accuracy vs cost, latency, scalability, explainability, and maintenance).

Key objectives:

* Train and evaluate **classical ML** models (TF‑IDF/CountVectorizer + Logistic Regression / SVM / Random Forest) as baselines.
* Fine‑tune **BERT** (and/or other transformer variants) for sentiment classification and compare results.
* Run systematic **hyperparameter tuning** for both model families (grid/search/random search + cross‑validation) to find best configurations.
* Analyse and visulise **tradeoffs** for real‑world adoption: inference latency, compute cost, explainability, training data needs, and ease of deployment.

---

## What makes this project different

Many projects compare models only by accuracy — this work goes further by:

* Performing rigorous **hyperparameter tuning** and reporting tuned results (not just default runs).
* Measuring **operational metrics** (inference time per sample, training time) to highlight deployment costs.
* Producing a structured **tradeoff analysis** aimed at business stakeholders describing when a lightweight model is preferable to a heavyweight transformer and why.

---

## Highlights / Features

* Direct comparison: **Traditional ML (TF‑IDF + classifier)** vs **BERT (fine‑tuned)** with matched evaluation protocols.
* Hyperparameter tuning pipelines for both approaches (scikit‑learn `GridSearchCV` / `RandomizedSearchCV` and Hugging Face `Trainer` with `optuna`/manual sweeps).
* Operational profiling: model size (MB), average inference latency (ms/sample), GPU vs CPU requirements, and recommended deployment patterns.
* Business‑facing tradeoff report summarising cost/benefit decisions.
* Visual outputs: word clouds, confusion matrices, precision/recall curves, and time‑series sentiment dashboards.

---

## Datasets

Used datasets and samples (see `data/README.md` for download & preprocessing instructions):

* **Sentiment140** (Kaggle) — baseline and larger runs.
* Sampled corpus of tweets (\~1.6M) used for scaling experiments and profiling.

> Raw large datasets are not included due to size and licensing. The notebooks include cells to fetch or synthetically reproduce small samples for demo purposes.

---

## Notebooks & scripts

Main artifacts:

* `FinalYear (1).ipynb` — full pipeline: preprocessing → EDA → baseline models → BERT fine‑tuning → NER & LDA → profiling and tradeoff analysis.
* `scripts/` contains helpers for preprocessing, metric collection, and hyperparameter search wrappers.

---

## Reproducing experiments & hyperparameter tuning

1. Install dependencies and follow `data/README.md` to prepare datasets.
2. For classical ML experiments: run the notebook cells that call scikit‑learn pipelines and `GridSearchCV`/`RandomizedSearchCV`. The search spaces and best parameters are saved to `reports/hyperparams/classical/`.
3. For BERT experiments: the notebook uses Hugging Face `Trainer` with example sweep configs (batch size, learning rate, weight decay, number of epochs). You can run automated sweeps via `optuna` or manual runs; checkpoints are saved under `models/bert/`.
4. Profiling: use the included `scripts/profile_model.py` to measure memory, model size and per‑sample latency on CPU and (if available) GPU.

Example hyperparameter search snippets are provided in the notebook; they include suggested search ranges and recommended early stopping criteria.

---

## Tradeoffs — guidance for businesses

This section summarises the kinds of decisions businesses should consider when choosing between traditional ML and BERT:

* **Accuracy vs Cost**: BERT typically yields higher accuracy (especially with subtle language) but requires more compute for training and inference. Classical models often reach acceptable performance at a fraction of the cost.
* **Latency & Throughput**: For real‑time or high‑throughput pipelines, lightweight models (or distilled transformers) are preferable. Reported metrics in `reports/` show concrete ms/sample figures from experiments.
* **Explainability**: Classical models with sparse features (TF‑IDF) are easier to explain to stakeholders. For transformer models, use explainability tools (LIME, SHAP) — additional complexity and compute required.
* **Maintainability & Data Drift**: Smaller models are faster to retrain and deploy; transformers may need periodic fine‑tuning with fresh data. Consider model monitoring for drift.
* **Deployment complexity**: BERT models may require GPU-backed inference or optimized runtimes (ONNX, TorchScript, TensorRT) to meet latency targets.

A concise decision matrix and recommended patterns for small, medium and large businesses are included in the `reports/tradeoffs/` folder.

---

## Results & evaluation summary

The notebook contains the full evaluation: accuracy, precision, recall, F1, confusion matrices, and operational metrics. A short summary of tuned best results should be added after you finalise runs — for now the repository saves all hyperparameter outputs and best checkpoints under `reports/` and `models/`.

---

## Project architecture

```
/data                # dataset download instructions & small samples
/notebooks           # Jupyter notebooks (main analysis notebook here)
/models              # saved model checkpoints (classical & transformer)
/app                 # optional demo app (Flask / Streamlit)
/scripts             # helper scripts for preprocessing, hyperparam search & profiling
/reports             # plots, exported CSVs, hyperparam logs and tradeoff report
/requirements.txt
README.md
```

---

## Technologies & libraries

* Python (3.8+)
* Jupyter / Google Colab
* scikit‑learn, Hugging Face Transformers, Optuna
* spaCy (NER), Gensim (LDA)
* Pandas, Matplotlib
* Streamlit / Flask (optional demo)
* PySpark (large‑scale preprocessing experiments)

---

## Supervisor & contact

* **Student:** Fabio Rodrigues
* **Supervisor:** Ferran Espuny Pujol
* **Email:** [faboorod2@gmail.com](mailto:faboorod2@gmail.com)

---

## Contributing

Contributions and reproducibility fixes are welcome. If you want to run full BERT sweeps you will benefit from GPU access; small sample reproductions run on CPU.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Next steps

* Run final hyperparameter sweeps and populate `reports/results_summary.md` with tuned best models and operational numbers.
* Add a `environment.yml` or Dockerfile and lightweight CI that runs the classical ML pipeline as a smoke test.
* Consider including a distilled transformer (e.g., DistilBERT) as a middle ground for businesses.

---

*README updated to emphasise the explicit comparison between traditional NLP models and BERT, and to document hyperparameter tuning and business tradeoffs.*
