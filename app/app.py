"""Medical-themed local dashboard for the IDRiD research model."""
import io,json,time
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
import streamlit as st
import torch
from PIL import Image,UnidentifiedImageError
from src.pipeline import CLASSES,ROOT,load_model,predict_image

CHECKPOINT=ROOT/"artifacts"/"checkpoints"/"best_model.pt"
REPORTS=ROOT/"reports"
st.set_page_config(page_title="RetinaScreen | DR Research",page_icon="◉",layout="wide")
st.markdown("""
<style>
.stApp{background:#fff;color:#30342c}.block-container{max-width:1280px;padding-top:1.3rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #edf0e7}
[data-testid="stMetric"]{background:#fff;border:1px solid #e9ecdf;padding:.9rem 1rem;border-radius:14px;box-shadow:0 4px 18px #25291c08}
[data-testid="stMetricLabel"] p{color:#70736a!important;font-size:.82rem}[data-testid="stMetricValue"]{color:#30342c;font-weight:700}
.stTabs [data-baseweb="tab-list"]{gap:.45rem;border-bottom:1px solid #eceee5}
.stTabs [aria-selected="true"]{color:#343a2e!important;border-bottom:3px solid #a6d834!important}
.stButton>button{background:#b9e34c;color:#303820;border:1px solid #a9d335;border-radius:10px;font-weight:700;min-height:2.8rem;transition:transform .15s,box-shadow .15s}
.stButton>button:hover{background:#c8ed68;border-color:#a9d335;color:#303820;box-shadow:0 5px 14px #a6d83438;transform:translateY(-1px)}
[data-testid="stFileUploader"] section{border:1px dashed #b4d65e;background:#fff;border-radius:14px}
.hero{position:relative;overflow:hidden;background:#fff;border:1px solid #edf0e7;border-left:7px solid #b9e34c;border-radius:20px;padding:1.2rem 1.7rem;margin:.25rem 0 .7rem;box-shadow:0 8px 28px #25291c08}
.hero:after{content:"";position:absolute;width:160px;height:160px;border:20px solid #eff7d8;border-radius:50%;right:48px;top:-66px;pointer-events:none}
.hero .eyebrow{color:#718d2b;text-transform:uppercase;letter-spacing:.13em;font-weight:750;font-size:.72rem}
.hero h1{color:#30342c;margin:.35rem 0 .25rem;font-size:1.85rem}.hero p{color:#666a60;margin:0;max-width:780px}
.note{background:#f8faef;border:1px solid #edf1dd;border-left:4px solid #b9e34c;padding:.72rem 1rem;border-radius:9px;color:#494d42;margin:.6rem 0 1rem}
.card{background:#fff;border:1px solid #e9ecdf;border-radius:16px;padding:1.15rem;box-shadow:0 5px 18px #25291c06}.muted{color:#74776d;font-size:.88rem}
.grade{color:#30342c;font-size:1.6rem;font-weight:750}.pill{display:inline-block;padding:.3rem .65rem;border-radius:99px;font-weight:700;background:#f0f7d9;color:#5b741d}
.eyebrow{color:#748d31;text-transform:uppercase;letter-spacing:.12em;font-weight:750;font-size:.72rem}
.upload-card{background:#fbfcf7;border:1px solid #edf0e4;border-radius:16px;padding:1.1rem 1.2rem}
.step-dot{display:inline-block;background:#eaf5c9;color:#617d1c;border-radius:50%;width:1.65rem;height:1.65rem;text-align:center;line-height:1.65rem;font-weight:750;margin-right:.45rem}
.small-note{color:#74776d;font-size:.82rem;line-height:1.5}
</style>
""",unsafe_allow_html=True)

def load_json(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError,json.JSONDecodeError,OSError):return None
def pct(x):return "—" if x is None else f"{100*x:.1f}%"
@st.cache_resource(show_spinner="Loading EfficientNet-B0…")
def cached_model(path,modified):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,ck=load_model(path,device);return model,ck,device

metrics=load_json(REPORTS/"test_metrics.json")
bootstrap=load_json(REPORTS/"test_bootstrap_ci.json")
history=load_json(REPORTS/"training_history.json")
ready=CHECKPOINT.is_file()

with st.sidebar:
    st.markdown("### ◉ RetinaScreen");st.caption("DR severity · research prototype");st.divider()
    st.markdown("**Model**");st.write("EfficientNet-B0 · five grades");st.write("Dataset · IDRiD")
    if ready:st.success("Trained checkpoint ready")
    else:st.warning("Checkpoint missing");st.code("python -m src.pipeline train",language="powershell")
    st.caption("Uploads are processed in memory. Not a diagnosis.")

st.markdown("""<div class="hero"><div class="eyebrow">A small research project · IDRiD</div><h1>Looking closer, learning more.</h1><p>A five-grade retinal image study, with its results and model reasoning made easier to explore.</p></div><div class="note"><b>Made for learning.</b> This university prototype is experimental and must not guide medical care.</div>""",unsafe_allow_html=True)
overview,screen,performance,about=st.tabs(["Overview","Screen an image","Performance","About"])

with overview:
    st.subheader("Measured baseline")
    if metrics:
        cols=st.columns(5)
        for col,(label,key) in zip(cols,[("Test accuracy","accuracy"),("Macro-F1","macro_f1"),("Quadratic weighted kappa","quadratic_weighted_kappa"),("Referable sensitivity","referable_sensitivity"),("Referable specificity","referable_specificity")]):
            col.metric(label,pct(metrics.get(key)))
            ci=(bootstrap or {}).get("metrics",{}).get(key)
            if ci:col.caption(f"95% CI {pct(ci['lower_95'])}–{pct(ci['upper_95'])}")
        st.caption(f"Deduplicated official IDRiD test partition · n={metrics.get('n',0)}. Intervals are stratified bootstrap estimates, not external validation.")
    else:st.info("Run python demo.py --accuracy to populate held-out metrics.")
    left,right=st.columns([1.1,.9],gap="large")
    with left:
        st.markdown("#### Training history")
        if history:
            h=pd.DataFrame(history).set_index("epoch")
            if {"train_loss","validation_loss"}.issubset(h.columns):st.line_chart(h[["train_loss","validation_loss"]],height=235,color=["#9cc83b","#b6b9ae"])
            if {"train_accuracy","validation_accuracy"}.issubset(h.columns):st.line_chart(h[["train_accuracy","validation_accuracy"]],height=215,color=["#9cc83b","#b6b9ae"])
            st.caption("Checkpoint selected by validation loss. Validation contains 80 images.")
        else:st.info("Training curves appear after a training run.")
    with right:
        st.markdown("#### Referable DR rule")
        st.markdown('<div class="card"><b>Non-referable</b> · predicted grade 0–1<br><br><b>Referable</b> · predicted grade 2–4<br><br><span class="muted">Derived from the five-grade argmax.</span></div>',unsafe_allow_html=True)
        st.markdown("#### Severity classes")
        for i,n in enumerate(CLASSES):st.write(f"**{i}** · {n}")

with screen:
    st.markdown('<div class="eyebrow">A closer look · one image at a time</div>',unsafe_allow_html=True)
    st.subheader("Try the model on a retinal image")
    st.markdown('<p class="small-note">Choose a clear, centered fundus photo. Your image stays in this local session and is not saved by the app.</p>',unsafe_allow_html=True)
    if not ready:st.error("The trained checkpoint is missing. Train the model before screening.")
    else:
        upload=st.file_uploader("Choose a retinal image",type=["jpg","jpeg","png"],key="fundus",help="JPEG or PNG; 224 × 224 minimum, 24 megapixels maximum.")
        image=None
        if upload:
            raw=upload.getvalue()
            try:
                with Image.open(io.BytesIO(raw)) as probe:probe.verify()
                with Image.open(io.BytesIO(raw)) as decoded:image=decoded.convert("RGB")
                if min(image.size)<224:st.error("Image dimensions must be at least 224 × 224.")
                elif image.width*image.height>24_000_000:st.error("Image exceeds the 24-megapixel limit.")
            except (UnidentifiedImageError,OSError,ValueError):st.error("Could not decode this image. Choose a valid JPEG or PNG.")
        result=st.session_state.get("screen_result")
        if upload and image and min(image.size)>=224 and image.width*image.height<=24_000_000:
            preview,details=st.columns([.95,1.05],gap="large",vertical_alignment="top")
            with preview:
                with st.container(border=True):
                    st.markdown('<div class="eyebrow">Your image</div>',unsafe_allow_html=True)
                    st.image(image,width=300,caption=f"{image.width} × {image.height} · {upload.name}")
                    if st.button("Run screening and create map",type="primary",width="stretch"):
                        try:
                            net,ck,device=cached_model(str(CHECKPOINT.resolve()),CHECKPOINT.stat().st_mtime_ns)
                            start=time.perf_counter()
                            with st.spinner("Reading image features and creating the sensitivity map…"):out=predict_image(image,net,ck["config"],device)
                            st.session_state["screen_result"]={**out,"elapsed":time.perf_counter()-start,"filename":upload.name}
                        except Exception as exc:st.error(f"Inference failed: {exc}")
            with details:
                with st.container(border=True):
                    st.markdown('<div class="eyebrow">What happens next</div>',unsafe_allow_html=True)
                    st.markdown('<br><span class="step-dot">1</span><b>Classify</b><p class="small-note">The model estimates probabilities across five DR severity grades.</p><span class="step-dot">2</span><b>Summarize</b><p class="small-note">A simple grade 2–4 rule marks the prediction as referable for this demo.</p><span class="step-dot">3</span><b>Visualize</b><p class="small-note">Grad-CAM highlights image regions with greater influence on its selected class.</p><hr><span class="small-note">A sensitivity view is a model explanation, not evidence of a lesion. Results are uncalibrated and educational.</span>',unsafe_allow_html=True)
        elif not upload:
            st.markdown('<div class="upload-card"><div class="eyebrow">A simple three-step study</div><br><span class="step-dot">1</span>Choose a JPEG or PNG&nbsp;&nbsp; <span class="step-dot">2</span>Run the local model&nbsp;&nbsp; <span class="step-dot">3</span>Explore the prediction and map<br><br><span class="small-note">For best results, use a centered, well-lit fundus image at least 224 pixels on each side.</span></div>',unsafe_allow_html=True)
        if upload and result and result.get("filename")==upload.name:
            st.divider();st.markdown('<div class="eyebrow">Your model readout</div>',unsafe_allow_html=True);st.markdown("### Prediction and visual explanation");left,right=st.columns([.9,1.1],gap="large")
            with left:
                flag="Referable · grade 2–4" if result["referable"] else "Non-referable · grade 0–1"
                st.markdown(f'<div class="card"><div class="muted">Predicted severity</div><div class="grade">Grade {result["grade"]} · {result["class_name"]}</div><p><span class="pill">{flag}</span></p><span class="muted">Processed in {result["elapsed"]:.2f} s · uncalibrated probabilities</span></div>',unsafe_allow_html=True)
                st.markdown("#### Class probabilities")
                st.bar_chart(pd.DataFrame({"Probability":result["probabilities"]},index=[f"{i} · {n}" for i,n in enumerate(CLASSES)]),color="#86ad35",height=270)
            with right:
                st.markdown("#### Grad-CAM sensitivity map")
                st.image(result["gradcam"],caption="Red overlay marks regions with greater gradient influence on the selected class.",width=480)
                st.caption("Grad-CAM does not prove highlighted areas are lesions or establish clinical reasoning.")

with performance:
    st.subheader("Held-out performance")
    if metrics:
        st.write(f"Metrics use the deduplicated official test set (**n={metrics['n']}**).")
        left,right=st.columns(2,gap="large")
        with left:
            st.markdown("#### Per-grade results");rows=[]
            for i,n in enumerate(CLASSES):
                r=metrics.get("per_class",{}).get(n,{})
                rows.append({"Grade":i,"Support":int(r.get("support",0)),"Precision":r.get("precision",0),"Recall":r.get("recall",0),"F1":r.get("f1-score",0)})
            table=pd.DataFrame(rows,index=list(CLASSES))
            st.dataframe(table.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}),width="stretch")
            st.markdown("#### Recall and F1");st.bar_chart(table[["Recall","F1"]],color=["#85a93e","#315b4e"],height=280)
        with right:
            st.markdown("#### Confusion matrix");cm=REPORTS/"test_confusion_matrix.png"
            if cm.is_file():st.image(str(cm),caption="Rows are true grades; columns are predictions.",width="stretch")
            st.markdown("#### Binary screening");a,b=st.columns(2);a.metric("Sensitivity",pct(metrics.get("referable_sensitivity")));b.metric("Specificity",pct(metrics.get("referable_specificity")))
            st.caption("Referable means grades 2–4, derived from five-class predictions.")
        if bootstrap:
            st.markdown("#### Test-set uncertainty · stratified bootstrap");items=[]
            labels={"accuracy":"Accuracy","macro_f1":"Macro-F1","quadratic_weighted_kappa":"QWK","referable_sensitivity":"Sensitivity","referable_specificity":"Specificity"}
            for key,label in labels.items():
                q=bootstrap["metrics"][key];items.append({"Metric":label,"Estimate":q["estimate"],"Lower 95%":q["lower_95"],"Upper 95%":q["upper_95"]})
            st.dataframe(pd.DataFrame(items).set_index("Metric").style.format("{:.3f}"),width="stretch")
    else:st.info("Run python demo.py --accuracy after training to generate metrics.")
    ablation=REPORTS/"loss_weighting_ablation.csv"
    if ablation.is_file():
        st.divider();st.markdown("#### Class-weighting ablation · validation only")
        st.caption("Same 80-image validation split; official test set was not used.")
        plot=REPORTS/"loss_weighting_ablation.png"
        if plot.is_file():st.image(str(plot),width="stretch")
        st.dataframe(pd.read_csv(ablation),width="stretch",hide_index=True)

with about:
    st.markdown('<div class="eyebrow">The choices behind this demo</div>',unsafe_allow_html=True)
    st.subheader("A small, transparent baseline")
    st.write("This project explores how a familiar image-classification model handles a five-level diabetic-retinopathy grading task. The goal is to make the experiment inspectable: show the training history, report held-out results, and give a visual view of what influenced each prediction.")
    left,right=st.columns(2,gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Model notes")
            st.markdown("**Architecture** · ImageNet EfficientNet-B0 with five outputs.  \n**Input** · 224 × 224, ImageNet normalization.  \n**Training** · AdamW, inverse-square-root class weighting, seed 42, early stopping on validation loss.  \n**Explanation** · Grad-CAM at the final feature block.")
    with right:
        with st.container(border=True):
            st.markdown("#### Data and care")
            st.markdown("**Dataset** · Indian Diabetic Retinopathy Image Dataset (IDRiD), Disease Grading task.  \n**Source** · IEEE DataPort; IDRiD was introduced by Porwal and colleagues.  \n**Privacy** · Uploaded image bytes are used for local inference and are not saved by this UI.  \n**Limit** · Small test set, uncalibrated scores, no external validation.")
    st.warning("Academic demonstration only. Small test set and low recall for some grades. Not clinically validated; do not use for care.")
    st.markdown("#### References")
    st.markdown("- [IDRiD on IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)  \n- [IDRiD dataset details and grading task](https://idrid.grand-challenge.org/Data/)  \n- [IDRiD data descriptor · Porwal et al.](https://www.mdpi.com/2306-5729/3/3/25)")
