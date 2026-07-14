# 5. Project Development Phase

This folder contains the full working implementation.

## Run order

```bash
pip install -r requirements.txt

python data/generate_data.py     # generate/refresh source data
python data/preprocess.py        # clean, engineer features, encode
python data/eda.py                # generate EDA plots (static/images/)
python model/train_models.py     # train & compare 4 classifiers, save best
python app.py                    # launch the Flask web app
```

Then open **http://127.0.0.1:5000**.

Or open `notebook/Credit_Card_Approval_EDA_and_Model_Training.ipynb` in
Jupyter for the interactive, narrated version of the EDA → preprocessing →
training pipeline (produces the same saved artifacts used by `app.py`).

## Folder contents

- `data/` — dataset generation, preprocessing, and EDA scripts + CSVs
- `notebook/` — end-to-end Jupyter notebook (executed, with outputs)
- `model/` — training script + saved model artifacts (`.pkl`) + comparison results
- `templates/`, `static/` — Flask front-end (HTML/CSS)
- `app.py` — Flask application entry point
- `watson_deployment/` — optional IBM Watson ML cloud deployment script
- `requirements.txt` — Python dependencies

See **`7. Project Documentation`** for the full README with architecture and
usage details.
