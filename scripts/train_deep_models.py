from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from cardiotwin.constants import PTBXL_SUPERCLASSES
from cardiotwin.models.deep_ecg import make_deep_model

class NPZECGDataset(Dataset):
    def __init__(self, records_index: pd.DataFrame, split: str, max_records=None, augment=False):
        df = records_index[records_index['split'] == split].copy()
        if max_records:
            df = df.iloc[:max_records]
        self.df = df.reset_index(drop=True)
        self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        d = np.load(row['npz_path'], allow_pickle=True)
        x = d['signal'].astype(np.float32).T  # leads x samples
        y = d['labels'].astype(np.float32)
        if self.augment:
            x = augment_ecg(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def augment_ecg(x):
    y = np.array(x, copy=True)
    if np.random.rand() < 0.5:
        y += np.random.normal(0, 0.015, size=y.shape).astype(np.float32)
    if np.random.rand() < 0.3:
        scale = np.random.uniform(0.9, 1.1, size=(y.shape[0], 1)).astype(np.float32)
        y *= scale
    if np.random.rand() < 0.2:
        lead = np.random.randint(0, y.shape[0])
        y[lead] *= np.random.uniform(0.0, 0.4)
    return y.astype(np.float32)

def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device)
            p=torch.sigmoid(model(x)).cpu().numpy()
            ys.append(y.cpu().numpy()); ps.append(p)
    y_true=np.vstack(ys); y_prob=np.vstack(ps); y_pred=(y_prob>=0.5).astype(int)
    def safe(fn, default=np.nan):
        try: return float(fn())
        except Exception: return float(default)
    return {
        'auroc_macro': safe(lambda: roc_auc_score(y_true, y_prob, average='macro')),
        'auprc_macro': safe(lambda: average_precision_score(y_true, y_prob, average='macro')),
        'macro_f1': safe(lambda: f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_precision': safe(lambda: precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall_sensitivity': safe(lambda: recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }, y_true, y_prob

def class_pos_weight(records_index):
    y = records_index[PTBXL_SUPERCLASSES].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)

def train_one(name, records_index, out_dir, epochs=5, batch_size=128, lr=1e-3, max_records=None, device='cpu'):
    train_ds=NPZECGDataset(records_index, 'train', max_records=max_records, augment=True)
    val_ds=NPZECGDataset(records_index, 'val', max_records=max_records//8 if max_records else None, augment=False)
    test_ds=NPZECGDataset(records_index, 'test', max_records=max_records//8 if max_records else None, augment=False)
    train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model=make_deep_model(name, in_leads=12, n_classes=len(PTBXL_SUPERCLASSES)).to(device)
    pos_weight=class_pos_weight(records_index[records_index['split']=='train']).to(device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best=-1; bad=0; best_state=None
    for ep in range(1, epochs+1):
        model.train(); losses=[]
        for x,y in train_loader:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(x), y); loss.backward(); opt.step(); losses.append(float(loss.item()))
        val_metrics,_,_=evaluate(model,val_loader,device)
        score=val_metrics.get('macro_f1',0.0)
        print(f'{name} epoch={ep} loss={np.mean(losses):.4f} val_macro_f1={score:.4f}')
        if score > best:
            best=score; bad=0; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    if best_state: model.load_state_dict(best_state)
    t0=time.perf_counter(); metrics, y_true, y_prob = evaluate(model,test_loader,device); elapsed=time.perf_counter()-t0
    metrics['model']=name; metrics['test_eval_seconds']=elapsed; metrics['device']=device
    path=out_dir/f'{name}_model.pt'
    torch.save({'state_dict': model.state_dict(), 'model_name': name, 'labels': PTBXL_SUPERCLASSES, 'metrics': metrics}, path)
    metrics['model_path']=str(path)
    return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processed-dir', default='artifacts/processed')
    ap.add_argument('--out-dir', default='artifacts/deep_models')
    ap.add_argument('--models', default='resnet1d,inceptiontime,transformer')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-records', type=int, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    records=pd.read_csv(Path(args.processed_dir)/'records_index.csv')
    rows=[]
    for name in [x.strip() for x in args.models.split(',') if x.strip()]:
        rows.append(train_one(name, records, out, epochs=args.epochs, batch_size=args.batch_size, max_records=args.max_records, device=args.device))
    lb=pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
    lb.to_csv(out/'model_leaderboard.csv', index=False)
    (out/'deep_metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(lb)
    print(f'Saved leaderboard: {out/"model_leaderboard.csv"}')

if __name__=='__main__':
    main()
