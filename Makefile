# Convenience targets (Linux/macOS, or Windows with make). On Windows without make run the commands directly.
PY ?= python

install:           ## install the full pipeline environment
	$(PY) -m pip install -r requirements.txt

test:              ## leakage tests, track-record rules, dashboard smoke test
	$(PY) -m pytest tests -q

pipeline:          ## ingest -> clean -> features -> train -> evaluate -> reports (about 45 min on a laptop CPU)
	$(PY) -m src.run_all --plus

notebooks:         ## regenerate and execute notebooks/
	$(PY) -m src.build_notebooks

forecast:          ## next delivery day forecast -> reports/forecast_next_day.csv
	$(PY) -m src.predict_next_day

dashboard:         ## open the dashboard (reads reports/ only)
	$(PY) -m streamlit run app/dashboard.py

docker-dashboard:  ## build and run the dashboard container
	docker build -t bg-power-dashboard . && docker run --rm -p 8501:8501 bg-power-dashboard

.PHONY: install test pipeline notebooks forecast dashboard docker-dashboard
