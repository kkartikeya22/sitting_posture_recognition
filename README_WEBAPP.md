
# Posture Web App (Flask + Bright UI)

This web app exposes your Sitting Posture Recognition model for:
- Single Image analysis (upload)  
- Live Webcam analysis (side-by-side input + output)

## Folder Structure

```
posture_webapp/
├─ app.py              # Flask server (loads your existing model & config)
└─ static/
   ├─ index.html       # Bright UI
   ├─ styles.css
   └─ script.js
```

> The server imports your existing files `config_reader.py`, `model.py`, and `util.py` from the project root.

## Prerequisites

1) Install system dependencies for OpenCV (if needed for your OS).  
2) Install Python packages (these match your repo's `requirements.txt`):

```
pip install -r requirements.txt
pip install flask
```

## Model Weights

Place your `model.h5` at `./model/keras/model.h5` relative to where you run the server, or point to a custom path:
```
export POSTURE_WEIGHTS=/full/path/to/model.h5
```

## Run

From your project root (where `config`, `config_reader.py`, `model.py`, `util.py` live):

```
python -m flask --app posture_webapp/app.py run --host 0.0.0.0 --port 8000
# or
python posture_webapp/app.py
```

Then open: http://localhost:8000

## Notes

- The server holds the model in memory and performs inference on each request.  
- Webcam streaming sends frames periodically (every ~600 ms) to the backend and renders the processed image on the right.
- Make sure the person is **in lateral (side) view** for best posture classification (matching your original repo constraints).
