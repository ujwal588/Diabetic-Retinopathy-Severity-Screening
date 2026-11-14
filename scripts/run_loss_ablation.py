"""Compare inverse-square-root weighted CE to plain CE on validation only."""
import csv, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"
CHECKPOINTS=ROOT/"artifacts"/"checkpoints"
def run(*args): subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def main():
    weighted=CHECKPOINTS/"best_model.pt"; unweighted=CHECKPOINTS/"best_model_unweighted.pt"
    if not weighted.is_file(): raise FileNotFoundError("Train the default weighted baseline first: python -m src.pipeline train")
    # Compare only on validation; the official test partition is not used.
    run("-m","src.pipeline","evaluate","--split","validation","--checkpoint",str(weighted))
    run("-m","src.pipeline","train","--loss-weighting","none","--checkpoint-name",unweighted.name)
    run("-m","src.pipeline","evaluate","--split","validation","--checkpoint",str(unweighted))
    summaries=[]
    for label,checkpoint in (("inverse_sqrt_weighted",weighted),("unweighted",unweighted)):
        data=json.loads((REPORTS/f"validation_{checkpoint.stem}_metrics.json").read_text(encoding="utf-8"))
        summaries.append({"loss_weighting":label,"checkpoint":checkpoint.name,"n_validation":data["n"],
            "accuracy":data["accuracy"],"macro_f1":data["macro_f1"],
            "quadratic_weighted_kappa":data["quadratic_weighted_kappa"],
            **{f"recall_grade_{i}":data["per_class"][name]["recall"] for i,name in enumerate(("No DR","Mild","Moderate","Severe","Proliferative DR"))}})
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS/"loss_weighting_ablation.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5)); keys=("accuracy","macro_f1","quadratic_weighted_kappa"); x=range(len(keys)); width=.36
    for offset,row,label in ((-.18,summaries[0],"Weighted CE"),(.18,summaries[1],"Unweighted CE")):
        axes[0].bar([i+offset for i in x],[row[k] for k in keys],width,label=label)
    axes[0].set_xticks(list(x),["Accuracy","Macro-F1","QWK"]); axes[0].set_ylim(0,1); axes[0].set_title("Validation metrics"); axes[0].legend()
    grades=range(5)
    for offset,row,label in ((-.18,summaries[0],"Weighted CE"),(.18,summaries[1],"Unweighted CE")):
        axes[1].bar([i+offset for i in grades],[row[f"recall_grade_{i}"] for i in grades],width,label=label)
    axes[1].set_xticks(list(grades),["0","1","2","3","4"]); axes[1].set_ylim(0,1); axes[1].set_title("Per-grade validation recall"); axes[1].set_xlabel("True grade"); axes[1].legend()
    fig.suptitle("Class-weighting ablation — validation only"); fig.tight_layout(); fig.savefig(REPORTS/"loss_weighting_ablation.png",dpi=160); plt.close(fig)
    print((REPORTS/"loss_weighting_ablation.csv").read_text(encoding="utf-8"))
    print("Validation only; no test-set selection or scoring.")
if __name__=="__main__": main()
