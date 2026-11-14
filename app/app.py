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
.stApp{background:#f7faf7;color:#18352d}.block-container{max-width:1320px;padding-top:1.35rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e2eae3}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2eae3;padding:.9rem 1rem;border-radius:14px;box-shadow:0 4px 18px #18352d08}
[data-testid="stMetricLabel"] p{color:#60726b!important;font-size:.82rem}[data-testid="stMetricValue"]{color:#18352d;font-weight:700}
.stTabs [data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid #e2eae3}
.stTabs [aria-selected="true"]{color:#18352d!important;border-bottom:3px solid #719b27!important}
.stButton>button{background:#b8df59;color:#20351a;border:1px solid #a8ce4b;border-radius:10px;font-weight:700;min-height:2.8rem}
[data-testid="stFileUploader"] section{border:1px dashed #a9bf99;background:#fff;border-radius:14px}
.hero{background:linear-gradient(115deg,#173b32,#285846 70%,#477248);color:#fff;border-radius:22px;padding:2rem 2.2rem;margin:.4rem 0 1rem}
.hero .eyebrow{color:#d5efa0;text-transform:uppercase;letter-spacing:.14em;font-weight:700;font-size:.75rem}
.hero h1{color:#fff;margin:.55rem 0 .45rem;font-size:2rem}.hero p{color:#e3eee4;margin:0;max-width:780px}
.note{background:#f0f6df;border-left:4px solid #719b27;padding:.75rem 1rem;border-radius:8px;color:#344a2a;margin:.6rem 0 1rem}
.card{background:#fff;border:1px solid #e2eae3;border-radius:16px;padding:1.15rem}.muted{color:#60726b;font-size:.88rem}
.grade{color:#18352d;font-size:1.6rem;font-weight:750}.pill{display:inline-block;padding:.3rem .65rem;border-radius:99px;font-weight:700;background:#f0f6df;color:#4e6d1e}
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

st.markdown("""<div class="hero"><div class="eyebrow">Explainable AI · Retinal image research</div><h1>See what the model sees.</h1><p>Explore measured IDRiD performance or upload one fundus image for a five-grade prediction, referral flag, and Grad-CAM sensitivity map.</p></div><div class="note"><b>Academic prototype, not a medical device.</b> Predictions and visual explanations are experimental and must not guide care.</div>""",unsafe_allow_html=True)
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
            if {"train_loss","validation_loss"}.issubset(h.columns):st.line_chart(h[["train_loss","validation_loss"]],height=235,color=["#84aa38","#244f40"])
            if {"train_accuracy","validation_accuracy"}.issubset(h.columns):st.line_chart(h[["train_accuracy","validation_accuracy"]],height=215,color=["#84aa38","#244f40"])
            st.caption("Checkpoint selected by validation loss. Validation contains 80 images.")
        else:st.info("Training curves appear after a training run.")
    with right:
        st.markdown("#### Referable DR rule")
        st.markdown('<div class="card"><b>Non-referable</b> · predicted grade 0–1<br><br><b>Referable</b> · predicted grade 2–4<br><br><span class="muted">Derived from the five-grade argmax.</span></div>',unsafe_allow_html=True)
        st.markdown("#### Severity classes")
        for i,n in enumerate(CLASSES):st.write(f"**{i}** · {n}")

with screen:
    st.subheader("Screen one fundus image")
    st.write("Run the local trained model on one uploaded retinal image. The app processes it in memory and does not save the original.")
    if not ready:st.error("The trained checkpoint is missing. Train the model before screening.")
    else:
        upload=st.file_uploader("Choose a retinal image",type=["jpg","jpeg","png"],key="fundus",help="JPEG or PNG; 224 × 224 minimum, 24 megapixels maximum.")
        if upload:
            raw=upload.getvalue()
            try:
                with Image.open(io.BytesIO(raw)) as probe:probe.verify()
                with Image.open(io.BytesIO(raw)) as decoded:image=decoded.convert("RGB")
                if min(image.size)<224:st.error("Image dimensions must be at least 224 × 224.")
                elif image.width*image.height>24_000_000:st.error("Image exceeds the 24-megapixel limit.")
                else:
                    st.image(image,caption=f"{image.width} × {image.height} · {upload.name}",use_container_width=True)
                    if st.button("Run screening and generate Grad-CAM",type="primary",use_container_width=True):
                        try:
                            net,ck,device=cached_model(str(CHECKPOINT.resolve()),CHECKPOINT.stat().st_mtime_ns)
                            start=time.perf_counter()
                            with st.spinner("Running the model and creating the sensitivity map…"):out=predict_image(image,net,ck["config"],device)
                            st.session_state["screen_result"]={**out,"elapsed":time.perf_counter()-start,"filename":upload.name}
                        except Exception as exc:st.error(f"Inference failed: {exc}")
            except (UnidentifiedImageError,OSError,ValueError):st.error("Could not decode this image. Choose a valid JPEG or PNG.")
        result=st.session_state.get("screen_result")
        if upload and result and result.get("filename")==upload.name:
            st.divider();st.markdown("### Model output");left,right=st.columns([.9,1.1],gap="large")
            with left:
                flag="Referable · grade 2–4" if result["referable"] else "Non-referable · grade 0–1"
                st.markdown(f'<div class="card"><div class="muted">Predicted severity</div><div class="grade">Grade {result["grade"]} · {result["class_name"]}</div><p><span class="pill">{flag}</span></p><span class="muted">Processed in {result["elapsed"]:.2f} s · uncalibrated probabilities</span></div>',unsafe_allow_html=True)
                st.markdown("#### Class probabilities")
                st.bar_chart(pd.DataFrame({"Probability":result["probabilities"]},index=[f"{i} · {n}" for i,n in enumerate(CLASSES)]),color="#86ad35",height=270)
            with right:
                st.markdown("#### Grad-CAM sensitivity map")
                st.image(result["gradcam"],caption="Red overlay marks regions with greater gradient influence on the selected class.",use_container_width=True)
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
            st.dataframe(table.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}),use_container_width=True)
            st.markdown("#### Recall and F1");st.bar_chart(table[["Recall","F1"]],color=["#85a93e","#315b4e"],height=280)
        with right:
            st.markdown("#### Confusion matrix");cm=REPORTS/"test_confusion_matrix.png"
            if cm.is_file():st.image(str(cm),caption="Rows are true grades; columns are predictions.",use_container_width=True)
            st.markdown("#### Binary screening");a,b=st.columns(2);a.metric("Sensitivity",pct(metrics.get("referable_sensitivity")));b.metric("Specificity",pct(metrics.get("referable_specificity")))
            st.caption("Referable means grades 2–4, derived from five-class predictions.")
        if bootstrap:
            st.markdown("#### Test-set uncertainty · stratified bootstrap");items=[]
            labels={"accuracy":"Accuracy","macro_f1":"Macro-F1","quadratic_weighted_kappa":"QWK","referable_sensitivity":"Sensitivity","referable_specificity":"Specificity"}
            for key,label in labels.items():
                q=bootstrap["metrics"][key];items.append({"Metric":label,"Estimate":q["estimate"],"Lower 95%":q["lower_95"],"Upper 95%":q["upper_95"]})
            st.dataframe(pd.DataFrame(items).set_index("Metric").style.format("{:.3f}"),use_container_width=True)
    else:st.info("Run python demo.py --accuracy after training to generate metrics.")
    ablation=REPORTS/"loss_weighting_ablation.csv"
    if ablation.is_file():
        st.divider();st.markdown("#### Class-weighting ablation · validation only")
        st.caption("Same 80-image validation split; official test set was not used.")
        plot=REPORTS/"loss_weighting_ablation.png"
        if plot.is_file():st.image(str(plot),use_container_width=True)
        st.dataframe(pd.read_csv(ablation),use_container_width=True,hide_index=True)

with about:
    st.subheader("Model and method");left,right=st.columns(2,gap="large")
    left.markdown("**Architecture** · ImageNet EfficientNet-B0, five outputs.  \n**Input** · 224 × 224, ImageNet normalization.  \n**Training** · AdamW, inverse-square-root weighted cross-entropy, seed 42, early stopping on validation loss.  \n**Dataset** · IDRiD Disease Grading.")
    right.markdown("**Explainability** · Grad-CAM at the final feature block.  \n**Referral rule** · Grades 2–4 derived from the prediction.  \n**Data hygiene** · Exact duplicates excluded across partitions.  \n**Privacy** · Upload is processed in memory, not saved by this UI.")
    st.warning("Academic demonstration only. Small test set and low recall for some grades. Not clinically validated; do not use for care.")
    st.markdown("#### References")
    st.markdown("- [IDRiD dataset](https://idrid.grand-challenge.org/Data/)  \n- [Related IDRiD feature-classification project](https://github.com/DeepakSkandh/diabetic-retinopathy-feature-classification)")
