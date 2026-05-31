from __future__ import annotations
import argparse, json
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from cardiotwin.reports.evidence_pack import build_research_cards, build_manifest, zip_evidence

def pdf_report(cards, manifest, out_pdf):
    out_pdf=Path(out_pdf); out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        fig=plt.figure(figsize=(8.27,11.69)); plt.axis('off')
        txt='CardioTwin-AI 12L Research Readiness Report\n\n'
        txt+='Clinical boundary: preliminary research-use screening and visual explanation only; not final diagnosis.\n\n'
        txt+='Cards generated: '+', '.join(cards.keys())+'\n'
        txt+=f'Evidence files indexed: {len(manifest)}\n\n'
        txt+='Key readiness dimensions:\n- TRIPOD+AI aligned checklist\n- STARD-AI aligned checklist\n- DECIDE-AI early-stage AI workflow checklist\n- Model card\n- Dataset card\n- Limitation card\n- Risk management summary\n- Evidence manifest with hashes\n'
        plt.text(0.05,0.95,txt,va='top',fontsize=12,wrap=True)
        pdf.savefig(fig); plt.close(fig)
        for name,obj in cards.items():
            fig=plt.figure(figsize=(8.27,11.69)); plt.axis('off')
            pretty=json.dumps(obj, indent=2, ensure_ascii=False)[:5000]
            plt.text(0.04,0.96,name+'\n\n'+pretty,va='top',fontsize=8,wrap=True)
            pdf.savefig(fig); plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='artifacts/research_evidence_pack')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cards=build_research_cards(out)
    manifest=build_manifest(out)
    pdf_report(cards, manifest, out/'research_readiness_report.pdf')
    zip_evidence(out, 'artifacts/evidence_bundle.zip')
    print(f'Saved evidence_bundle.zip')
    print(f'Saved {out/"research_readiness_report.pdf"}')

if __name__=='__main__': main()
