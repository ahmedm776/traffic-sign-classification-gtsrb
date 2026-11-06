import io, os, sys, numpy as np, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import CompactCNN, CLASS_NAMES
from PIL import Image

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

def test_class_map_complete():
    assert len(CLASS_NAMES) == 43 and len(set(CLASS_NAMES)) == 43

def test_model_forward_shape():
    m = CompactCNN()
    out = m(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 43)

def test_saved_model_reload_deterministic():
    c = torch.load("final_model.pt", weights_only=False)
    m1 = CompactCNN(); m1.load_state_dict(c["model"]); m1.eval()
    m2 = CompactCNN(); m2.load_state_dict(c["model"]); m2.eval()
    x = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(m1(x).argmax(1), m2(x).argmax(1))

def test_metrics_file_reports_full_test_set():
    import json
    m = json.load(open("metrics.json"))
    assert m["test"]["n"] == 12630 and m["reload_deterministic"] is True
    assert 0.9 <= m["test"]["accuracy"] <= 1.0

def test_app_endpoints():
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    assert c.get("/").status_code == 200
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0)
    r = c.post("/api/predict", files={"file": ("x.png", buf, "image/png")})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 5 and abs(sum(p["prob"] for p in preds) - max(p["prob"] for p in preds)) < 1.0
    assert all(0 <= p["class_id"] < 43 for p in preds)
    bad = c.post("/api/predict", files={"file": ("x.txt", io.BytesIO(b"nope"), "text/plain")})
    assert bad.status_code == 400

@pytest.mark.skipif(not os.path.exists("data/test_x.npy"),
                    reason="dataset not bundled; run src/prepare_data.py first (see README)")
def test_real_test_image_prediction_sane():
    x = np.load("data/test_x.npy", mmap_mode="r")[:8]
    y = np.load("data/test_y.npy", mmap_mode="r")[:8]
    c = torch.load("final_model.pt", weights_only=False)
    m = CompactCNN(); m.load_state_dict(c["model"]); m.eval()
    xt = torch.from_numpy(np.asarray(x)).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        pred = m((xt - 0.5) / 0.25).argmax(1)
    assert (pred == torch.from_numpy(np.asarray(y))).float().mean() > 0.5
