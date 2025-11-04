# Car Brand Recognition - DAT158

ML project that recognizes car brands from images using k-NN on color histograms.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/train_colorhist.py
python3 -m streamlit run webapp/app.py --server.port=8503
```

Open http://localhost:8503

## Structure

- `src/train_colorhist.py` - train k-NN model
- `src/features.py` - extract RGB histograms  
- `webapp/app.py` - Streamlit interface
- `dataset/train/<brand>/` - training images
- `models/` - saved model

## Model

k-NN (k=5) on 16-bin RGB histograms (48-dim). Accuracy: 55.5% on 2,113 images (Audi, BMW, Mercedes, Toyota).