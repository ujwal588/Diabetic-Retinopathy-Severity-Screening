"""Stratified paired bootstrap CIs from saved held-out predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"
def main(repetitions=5000,seed=42):
    path=REPORTS/"test_predictions.csv"
    if not path.is_file(): raise FileNotFoundError("Run python demo.py --accuracy first to create test_predictions.csv")
    frame=pd.read_csv(path); y=frame.true_grade.to_numpy(dtype=int); pred=frame.predicted_grade.to_numpy(dtype=int)
    strata=[np.flatnonzero(y==grade) for grade in range(5)]; rng=np.random.default_rng(seed)
    values={k:[] for k in ("accuracy","macro_f1","quadratic_weighted_kappa","referable_sensitivity","referable_specificity")}
    for _ in range(repetitions):
        idx=np.concatenate([rng.choice(ids,size=len(ids),replace=True) for ids in strata])
        truth=y[idx]; guess=pred[idx]; ref=truth>=2; refp=guess>=2
        values["accuracy"].append(accuracy_score(truth,guess))
        values["macro_f1"].append(f1_score(truth,guess,labels=np.arange(5),average="macro",zero_division=0))
        values["quadratic_weighted_kappa"].append(cohen_kappa_score(truth,guess,weights="quadratic"))
        values["referable_sensitivity"].append(np.sum(ref&refp)/np.sum(ref))
        values["referable_specificity"].append(np.sum(~ref&~refp)/np.sum(~ref))
    point=json.loads((REPORTS/"test_metrics.json").read_text(encoding="utf-8"))
    result={"method":"stratified paired bootstrap; resampled within true grade","seed":seed,"repetitions":repetitions,"n":len(y),"interval":"percentile 95%","metrics":{}}
    for key,samples in values.items():
        result["metrics"][key]={"estimate":point[key],"lower_95":float(np.percentile(samples,2.5)),"upper_95":float(np.percentile(samples,97.5))}
    out=REPORTS/"test_bootstrap_ci.json"; out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(out.read_text(encoding="utf-8"))
if __name__=="__main__": main()
