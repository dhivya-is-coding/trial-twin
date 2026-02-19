.PHONY: install data train report app test clean all

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/run_pipeline.py data --config configs/trial_nsclc.yaml

train:
	python scripts/run_pipeline.py train --config configs/trial_nsclc.yaml

report:
	python scripts/run_pipeline.py report --config configs/trial_nsclc.yaml

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/* data/outputs/* outputs/figures/* outputs/reports/*

all: data train report
