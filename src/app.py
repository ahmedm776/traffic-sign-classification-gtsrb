"""FastAPI demo: upload a traffic-sign image, get top-5 predictions."""
import io, numpy as np, torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from model import CompactCNN, CLASS_NAMES

torch.set_num_threads(2)
app = FastAPI(title="GTSRB traffic-sign classifier demo")
_model = None

def get_model():
    global _model
    if _model is None:
        m = CompactCNN()
        m.load_state_dict(torch.load("final_model.pt", weights_only=False)["model"])
        m.eval()
        _model = m
    return _model

def predict_pil(im):
    im = im.convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.uint8)).permute(2, 0, 1)[None].float() / 255.0
    x = (x - 0.5) / 0.25
    with torch.no_grad():
        probs = torch.softmax(get_model()(x), dim=1)[0]
    top = probs.topk(5)
    return [{"rank": i + 1, "class_id": int(c), "label": CLASS_NAMES[int(c)],
             "prob": round(float(probs[c]), 4)} for i, c in enumerate(top.indices)]

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Traffic sign classifier</title>
<style>body{font-family:sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem}
li{margin:.2rem 0}small{color:#666}</style></head><body>
<h1>Traffic-sign classifier (GTSRB, 43 classes)</h1>
<p><small>Compact CNN trained from scratch on a 26,640-image subset of the GTSRB training data (Hugging Face mirror). Demo only. Percentages are uncalibrated model confidence scores.</small></p>
<form method=post action=/predict enctype=multipart/form-data>
<input type=file name=file accept="image/*" required> <button type=submit>Classify</button></form>
{RESULT}
</body></html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("{RESULT}", "")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        im = Image.open(io.BytesIO(await file.read()))
    except Exception:
        return JSONResponse({"error": "not a readable image"}, status_code=400)
    res = predict_pil(im)
    if (file.content_type or "").startswith("text/html"):
        pass
    accept = ""
    rows = "".join(f"<li>model score <b>{r['prob']:.1%}</b> - {r['label']}</li>" for r in res)
    return HTMLResponse(PAGE.replace("{RESULT}", f"<h2>Prediction</h2><ol>{rows}</ol>"))

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    try:
        im = Image.open(io.BytesIO(await file.read()))
    except Exception:
        return JSONResponse({"error": "not a readable image"}, status_code=400)
    return {"predictions": predict_pil(im)}

@app.get("/sample", response_class=HTMLResponse)
def sample(idx: int = 7):
    x = np.load("data/test_x.npy", mmap_mode="r")
    y = np.load("data/test_y.npy", mmap_mode="r")
    i = int(idx) % len(y)
    Image.fromarray(np.asarray(x[i])).save("static_sample.png")
    im = Image.open("static_sample.png")
    res = predict_pil(im)
    rows = "".join(f"<li>model score <b>{r['prob']:.1%}</b> - {r['label']}</li>" for r in res)
    truth = CLASS_NAMES[int(y[i])]
    body = (f"<h2>Sample #{i} from the mirror's held-out test split (same size as the official GTSRB test set)</h2>"
            f"<img src=/static_sample.png style='width:128px;image-rendering:pixelated'>"
            f"<p>True class: <b>{truth}</b></p><ol>{rows}</ol>")
    return PAGE.replace("{RESULT}", body)

@app.get("/static_sample.png")
def static_sample():
    from fastapi.responses import FileResponse
    return FileResponse("static_sample.png")
