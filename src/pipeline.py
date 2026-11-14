"""IDRiD EfficientNet-B0 training, held-out evaluation, inference, and Grad-CAM."""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, cohen_kappa_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"B. Disease Grading"
CLASSES=("No DR","Mild","Moderate","Severe","Proliferative DR")

def _read(part):
    pfx="a" if part=="Training" else "b"
    matches=list((DATA/"2. Groundtruths").glob(f"{pfx}.*Labels.csv"))
    if len(matches)!=1: raise FileNotFoundError(f"Expected {part} IDRiD labels in {DATA/'2. Groundtruths'}")
    df=pd.read_csv(matches[0],skipinitialspace=True); df.columns=[str(c).strip() for c in df.columns]
    idc=next((c for c in df if c.lower().startswith("image name")),None)
    lab=next((c for c in df if c.lower()=="retinopathy grade"),None)
    if idc is None or lab is None: raise ValueError(f"Unexpected IDRiD label columns: {list(df.columns)}")
    folder=DATA/"1. Original Images"/("a. Training Set" if part=="Training" else "b. Testing Set")
    out=df[[idc,lab]].rename(columns={idc:"image_id",lab:"label"})
    out.image_id=out.image_id.astype(str).str.strip(); out.label=pd.to_numeric(out.label,errors="raise").astype(int)
    if out.image_id.duplicated().any() or not out.label.between(0,4).all(): raise ValueError("Duplicate IDs or invalid labels")
    out["image_path"]=out.image_id.map(lambda x:str(folder/f"{x}.jpg"))
    missing=[p for p in out.image_path if not Path(p).is_file()]
    if missing: raise FileNotFoundError(f"Missing image: {missing[0]}")
    return out

def manifest(val_fraction=.2,seed=42,save=True):
    if not 0<val_fraction<1: raise ValueError("validation_fraction must be between 0 and 1")
    tv=_read("Training"); test=_read("Testing")
    tr,va=train_test_split(tv,test_size=val_fraction,random_state=seed,stratify=tv.label)
    frames=[]
    for name,frame in (("train",tr),("validation",va),("test",test)):
        frame=frame.copy(); frame["split"]=name; frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    hashes={}; keep=[]; excluded=[]
    for idx,r in enumerate(result.itertuples()):
        h=hashlib.sha256(Path(r.image_path).read_bytes()).hexdigest()
        if h in hashes:
            prior=hashes[h]
            excluded.append({"image_id":r.image_id,"split":r.split,"duplicate_of":prior[1],"duplicate_split":prior[0]})
            continue
        hashes[h]=(r.split,r.image_id); keep.append(idx)
    if excluded:
        print("Excluded exact duplicate image(s): "+", ".join(f"{x['image_id']} ({x['split']}) duplicates {x['duplicate_of']} ({x['duplicate_split']})" for x in excluded))
    result=result.iloc[keep].reset_index(drop=True)
    if save:
        dest=ROOT/"data"/"splits"/"idrid_split.csv"; dest.parent.mkdir(parents=True,exist_ok=True)
        saved=result.sort_values(["split","image_id"]).copy()
        saved["image_path"]=saved["image_path"].map(lambda p:Path(p).resolve().relative_to(ROOT).as_posix())
        saved.to_csv(dest,index=False)
        pd.DataFrame(excluded,columns=["image_id","split","duplicate_of","duplicate_split"]).to_csv(dest.parent/"excluded_exact_duplicates.csv",index=False)
    return result

def transform(size,train):
    ops=[transforms.Resize((size,size))]
    if train: ops += [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(12)]
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

class Fundus(Dataset):
    def __init__(self,rows,size,train): self.rows=rows.reset_index(drop=True); self.tf=transform(size,train)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        try:
            with Image.open(r.image_path) as im: x=self.tf(im.convert("RGB"))
            return x,int(r.label),str(r.image_id)
        except Exception as e: raise RuntimeError(f"Image load failed: {r.image_path}: {e}") from e

def model(pretrained=False):
    net=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    net.classifier[1]=nn.Linear(net.classifier[1].in_features,5)
    return net

def load_model(checkpoint,device):
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    net=model(False); net.load_state_dict(ck["model_state"]); return net.to(device).eval(),ck

def train(args):
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); seed=int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=manifest(float(cfg["validation_fraction"]),seed)
    frames={s:rows[rows.split==s].reset_index(drop=True) for s in ("train","validation")}
    for s,f in frames.items(): print(f"{s}: {len(f)}; grades={f.label.value_counts().sort_index().to_dict()}")
    loaders={s:DataLoader(Fundus(f,int(cfg["image_size"]),s=="train"),batch_size=int(cfg["batch_size"]),shuffle=s=="train",num_workers=0,pin_memory=device.type=="cuda") for s,f in frames.items()}
    net=model(not args.no_pretrained).to(device)
    counts=frames["train"].label.value_counts().reindex(range(5),fill_value=0).to_numpy()
    if (counts==0).any(): raise ValueError(f"Training split misses class: {counts.tolist()}")
    class_weights=None
    if args.loss_weighting=="inverse_sqrt":
        class_weights=torch.tensor(np.sqrt(counts.sum()/counts),dtype=torch.float32,device=device); class_weights/=class_weights.mean()
    loss_fn=nn.CrossEntropyLoss(weight=class_weights); opt=torch.optim.AdamW(net.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    out=ROOT/"artifacts"/"checkpoints"; out.mkdir(parents=True,exist_ok=True); best=float("inf"); stale=0; history=[]
    for epoch in range(1,int(cfg["epochs"])+1):
        row={"epoch":epoch}
        for phase in ("train","validation"):
            net.train(phase=="train"); total=correct=n=0
            for x,y,_ in loaders[phase]:
                x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=="train"):
                    z=net(x); loss=loss_fn(z,y)
                    if phase=="train": loss.backward(); opt.step()
                total+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); n+=len(y)
            row[phase+"_loss"]=total/n; row[phase+"_accuracy"]=correct/n
        history.append(row); print(f"epoch {epoch}: train loss {row['train_loss']:.4f}; val loss {row['validation_loss']:.4f}; val acc {row['validation_accuracy']:.3f}")
        if row["validation_loss"]<best:
            best=row["validation_loss"]; stale=0
            torch.save({"model_state":net.state_dict(),"config":cfg,"seed":seed,"loss_weighting":args.loss_weighting,"best_validation_loss":best},out/args.checkpoint_name)
        else:
            stale+=1
            if stale>=int(cfg["patience"]): print("Early stopping."); break
    (ROOT/"reports").mkdir(exist_ok=True); (ROOT/"reports"/f"training_history_{Path(args.checkpoint_name).stem}.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    print(f"Checkpoint: {out/args.checkpoint_name}\nNext: python demo.py --accuracy")

def evaluate(checkpoint=None, split="test"):
    path=Path(checkpoint or ROOT/"artifacts"/"checkpoints"/"best_model.pt")
    if not path.is_file(): raise FileNotFoundError(f"Train first; missing checkpoint {path}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); net,ck=load_model(path,device); cfg=ck["config"]
    if split not in ("validation","test"): raise ValueError("split must be validation or test")
    rows=manifest(float(cfg["validation_fraction"]),int(ck["seed"]),save=False); subset=rows[rows.split==split].reset_index(drop=True)
    loader=DataLoader(Fundus(subset,int(cfg["image_size"]),False),batch_size=int(cfg["batch_size"]),shuffle=False,num_workers=0)
    ys=[]; ps=[]; names=[]; probs=[]
    with torch.inference_mode():
        for x,y,n in loader:
            q=net(x.to(device)).softmax(1).cpu().numpy(); probs.extend(q); ps.extend(q.argmax(1)); ys.extend(y.numpy()); names.extend(n)
    y=np.array(ys); p=np.array(ps); cm=confusion_matrix(y,p,labels=np.arange(5))
    rep=classification_report(y,p,labels=np.arange(5),target_names=CLASSES,output_dict=True,zero_division=0)
    ry=y>=2; rp=p>=2; tp=int((ry&rp).sum()); fn=int((ry&~rp).sum()); tn=int((~ry&~rp).sum()); fp=int((~ry&rp).sum())
    result={"dataset":"IDRiD","split":"official testing set" if split=="test" else "validation partition","n":len(y),"accuracy":float(accuracy_score(y,p)),
      "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),"quadratic_weighted_kappa":float(cohen_kappa_score(y,p,weights="quadratic")),
      "per_class":{c:{k:float(rep[c][k]) for k in ("precision","recall","f1-score","support")} for c in CLASSES},
      "referable_definition":"five-class argmax grades 2-4","referable_sensitivity":tp/(tp+fn) if tp+fn else None,
      "referable_specificity":tn/(tn+fp) if tn+fp else None,"referable_confusion":{"tn":tn,"fp":fp,"fn":fn,"tp":tp}}
    out=ROOT/"reports"; out.mkdir(exist_ok=True)
    prefix="test" if split=="test" else f"validation_{path.stem}"
    (out/f"{prefix}_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    import csv
    with (out/f"{prefix}_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["image_id","true_grade","predicted_grade",*[f"prob_grade_{i}" for i in range(5)]])
        w.writerows([[names[i],int(y[i]),int(p[i]),*map(float,probs[i])] for i in range(len(y))])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,6)); ax.imshow(cm,cmap="Blues"); ax.set(xticks=range(5),yticks=range(5),xticklabels=range(5),yticklabels=range(5),xlabel="Predicted grade",ylabel="True grade",title=f"IDRiD {split} confusion matrix")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    fig.tight_layout(); fig.savefig(out/f"{prefix}_confusion_matrix.png",dpi=160); plt.close(fig)
    print(json.dumps(result,indent=2)); return result

def predict(image_path,checkpoint=None):
    path=Path(checkpoint or ROOT/"artifacts"/"checkpoints"/"best_model.pt")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); net,ck=load_model(path,device); size=int(ck["config"]["image_size"])
    with Image.open(image_path) as im: image=im.convert("RGB"); x=transform(size,False)(image).unsqueeze(0).to(device)
    z=net(x); p=z.softmax(1)[0].detach().cpu().tolist(); grade=int(np.argmax(p))
    # Grad-CAM from EfficientNet's final feature block.
    acts=[]; grads=[]; layer=net.features[-1]
    h1=layer.register_forward_hook(lambda m,i,o:acts.append(o))
    h2=layer.register_full_backward_hook(lambda m,gi,go:grads.append(go[0]))
    net.zero_grad(set_to_none=True); net(x)[0,grade].backward(); h1.remove(); h2.remove()
    cam=torch.relu((grads[-1][0].mean((1,2),keepdim=True)*acts[-1][0]).sum(0))
    cam=torch.nn.functional.interpolate(cam[None,None],size=(image.height,image.width),mode="bilinear",align_corners=False)[0,0].detach().cpu().numpy()
    cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8); base=np.asarray(image).astype(np.float32); heat=np.zeros_like(base); heat[:,:,0]=cam*255
    out=(.58*base+.42*heat).clip(0,255).astype("uint8")
    dest=ROOT/"artifacts"/"gradcam"; dest.mkdir(parents=True,exist_ok=True); dest=dest/(Path(image_path).stem+"_gradcam.png"); Image.fromarray(out).save(dest)
    print(f"Grade {grade}: {CLASSES[grade]}; referable: {'YES' if grade>=2 else 'NO'}")
    print("Probabilities: "+", ".join(f"{CLASSES[i]}={p[i]:.3f}" for i in range(5))); print(f"Grad-CAM: {dest}")
    print("Research prototype only; not a clinical diagnosis.")

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train"); t.add_argument("--config",default=str(ROOT/"configs"/"baseline.yaml")); t.add_argument("--no-pretrained",action="store_true")
    t.add_argument("--loss-weighting",choices=("inverse_sqrt","none"),default="inverse_sqrt")
    t.add_argument("--checkpoint-name",default="best_model.pt")
    e=sub.add_parser("evaluate"); e.add_argument("--checkpoint"); e.add_argument("--split",choices=("validation","test"),default="test")
    p=sub.add_parser("predict"); p.add_argument("--image",required=True); p.add_argument("--checkpoint")
    a=ap.parse_args()
    if a.cmd=="train": train(a)
    elif a.cmd=="evaluate": evaluate(a.checkpoint,a.split)
    else: predict(a.image,a.checkpoint)

if __name__=="__main__": main()
