install:
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install black cml>=0.20.0

format:
	black *.py

train:
	mkdir -p results
	python train.py

eval:
	echo "# Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./results/model_results.png)' >> report.md

	cml comment create report.md

