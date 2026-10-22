# Salary Predictor

**A Philippine salary predictor.**

Salary Predictor is an upcoming website designed to make salary information easier to explore across different professions in the Philippines.

The idea is simple: enter your career details or upload a resume, review the information, and explore available monthly salary estimates and market benchmarks in Philippine pesos.

## ROADMAP

- A simple way to enter your job, location, experience, education, and skills.
- Resume uploads with a chance to check and edit the extracted details.
- Salary information explained in plain language, including when the available data is limited.
- Support for more occupations and better salary datasets as the project grows.

## Project status

**Local experimental model available.** A real Jobicy USD advertised-pay regressor is trained and inferencable locally; optional historical STEP and PSA evidence can provide transparent PHP scenarios.

The initial reference material includes occupation and skill information from ESCO and Philippine occupational wage statistics. Salary figures will be labeled according to their sources; a market average isn't the same as a personal salary prediction.

## Version history

### 0.0.0-alpha.1
- proj init

## Roadmap


---
## Local website and experimental salary scenarios

```sh
uv sync --extra web --extra train --extra dev --extra ingest
uv run --no-sync python -m salary_predictor.trainer experimental verify
uv run --no-sync python -m salary_predictor.web
```

Open http://127.0.0.1:5000 on the same computer. Choose a manual or PDF/DOCX profile, review and edit it, then select **Run experimental salary scenario**.

Enter an assumed PHP/USD rate for PHP scenarios (e.g. 58; not a live FX quote). Optional STEP survey and PSA 2022 inputs must use exact historical source dimensions. The website shows remote USD offers, historical wage evidence, the hand-weighted blended scenario and important limitations separately. It does not claim a calibrated current domestic salary range.

**Local-only:** the development server binds to 127.0.0.1 and enables experimental mode. The WSGI app factory keeps it disabled by default. The trusted local model is loaded only by a subprocess after hash verification, not by public runtime inference. Restart the server after Python changes.

