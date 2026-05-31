from __future__ import annotations
import argparse, asyncio, json, math
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket
import uvicorn
from cardiotwin.constants import LEADS_12
from cardiotwin.models.baseline_ml import load_model
from cardiotwin.pipeline.inference import run_inference

app = FastAPI(title='CardioTwin-AI 12L Realtime Demo Server', version='3.0')
MODEL_BUNDLE = None
SIMULATE = True

def synthetic_window(fs=500, seconds=10):
    t=np.arange(int(fs*seconds))/fs
    base=0.04*np.sin(2*np.pi*1.2*t)
    beat=np.zeros_like(t)
    for r in np.arange(0.6, seconds, 0.82):
        beat += 1.0*np.exp(-((t-r)/0.015)**2)
        beat += 0.18*np.exp(-((t-(r+0.22))/0.08)**2)
        beat += 0.08*np.exp(-((t-(r-0.18))/0.045)**2)
    leads=[]
    for i in range(12):
        leads.append((0.7+0.04*i)*beat + base + 0.01*np.random.randn(len(t)))
    return np.vstack(leads).T.astype(np.float32), fs, LEADS_12

@app.get('/health')
def health():
    return {'ok': True, 'model_loaded': MODEL_BUNDLE is not None, 'mode': 'simulate' if SIMULATE else 'hardware_scaffold'}

@app.websocket('/ws/live')
async def live(ws: WebSocket):
    await ws.accept()
    while True:
        signal, fs, leads = synthetic_window()
        state = run_inference(MODEL_BUNDLE, signal, fs, leads, record_id='live_simulated_window') if MODEL_BUNDLE else {'error':'model_not_loaded'}
        await ws.send_text(json.dumps({'type':'cardiotwin_state','state':state}, ensure_ascii=False))
        await asyncio.sleep(2.0)

def main():
    global MODEL_BUNDLE, SIMULATE
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-path', default='artifacts/models/baseline_model_v12_safety.joblib')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--simulate', action='store_true')
    args=ap.parse_args()
    SIMULATE = args.simulate
    path=Path(args.model_path)
    if not path.exists(): path=Path('artifacts/models/baseline_model.joblib')
    if path.exists(): MODEL_BUNDLE=load_model(path)
    print('Low-cost hardware proof-of-concept server')
    print('ADS1298 hardware scaffold: stream samples as 12-lead arrays to /ws/live or replace synthetic_window with serial acquisition.')
    print('Safety: battery power, USB isolation, enclosure, lead-off detection, no mains-connected body interface.')
    uvicorn.run(app, host=args.host, port=args.port)

if __name__=='__main__': main()
