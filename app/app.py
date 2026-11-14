"""Medical-themed local dashboard for the IDRiD research model."""
import base64,io,json,time
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
.why-card{background:#f8faef;border:1px solid #edf1dd;border-radius:14px;padding:1rem;margin:.8rem 0;color:#494d42;line-height:1.55}
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
def training_plot(history):
    """Render fixed-scale training curves as a non-interactive PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    frame=pd.DataFrame(history).set_index("epoch")
    fig,axes=plt.subplots(2,1,figsize=(8.4,5.3),dpi=120)
    for ax,(train_key,val_key,title) in zip(axes,(("train_loss","validation_loss","Loss per epoch"),("train_accuracy","validation_accuracy","Accuracy per epoch"))):
        if train_key in frame:ax.plot(frame.index,frame[train_key],color="#a6d834",linewidth=2.5,label="Training")
        if val_key in frame:ax.plot(frame.index,frame[val_key],color="#85897e",linewidth=2.2,label="Validation")
        ax.set_title(title,loc="left",fontsize=11,fontweight="semibold",color="#30342c")
        ax.set_xlim(1,max(int(frame.index.max()),2));ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if "accuracy" in title.lower():ax.set_ylim(0,1)
        ax.set_xlabel("Epoch");ax.grid(axis="y",color="#e8ebdf",linewidth=.8);ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False,loc="best")
    fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format="png",dpi=150,bbox_inches="tight");plt.close(fig)
    return buffer.getvalue()
def screening_report(result):
    """Create a one-page PDF report in memory for a completed screening demo."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fig=plt.figure(figsize=(8.27,11.69),facecolor="white")
    grid=fig.add_gridspec(3,2,left=.09,right=.91,top=.93,bottom=.10,height_ratios=[.72,1.5,.5],hspace=.42,wspace=.20)
    header=fig.add_subplot(grid[0,:]);header.axis("off")
    header.add_patch(plt.Rectangle((0,.05),.018,.87,transform=header.transAxes,color="#a6d834",clip_on=False))
    header.text(.06,.84,"RETINASCREEN  ·  UNIVERSITY RESEARCH DEMO",fontsize=9,color="#718d2b",weight="bold",transform=header.transAxes)
    header.text(.06,.55,"Retinal image screening report",fontsize=19,color="#30342c",weight="bold",transform=header.transAxes)
    header.text(.06,.20,f"Predicted grade {result['grade']} · {result['class_name']}     |     {('Referable' if result['referable'] else 'Non-referable')} by the project rule",fontsize=10,color="#55594f",transform=header.transAxes)
    bars=fig.add_subplot(grid[1,0]);indices=list(range(len(CLASSES)));values=result["probabilities"]
    bars.barh(indices,values,color=["#a6d834" if i==result["grade"] else "#d8ddcb" for i in indices],height=.62)
    bars.set_yticks(indices,[f"{i} · {name}" for i,name in enumerate(CLASSES)]);bars.invert_yaxis();bars.set_xlim(0,1);bars.set_xlabel("Uncalibrated model score")
    bars.set_title("Five-class outputs",loc="left",fontsize=12,weight="bold",color="#30342c");bars.grid(axis="x",color="#e8ebdf",linewidth=.8);bars.set_axisbelow(True);bars.spines[["top","right","left"]].set_visible(False)
    cam=fig.add_subplot(grid[1,1]);cam.imshow(result["gradcam"]);cam.set_title("Grad-CAM sensitivity map",loc="left",fontsize=12,weight="bold",color="#30342c");cam.axis("off")
    footer=fig.add_subplot(grid[2,:]);footer.axis("off");ranked=sorted(enumerate(values),key=lambda pair:pair[1],reverse=True);first,second=ranked[:2]
    footer.text(0,.92,"How to read this result",fontsize=12,weight="bold",color="#30342c",transform=footer.transAxes)
    footer.text(0,.63,f"The selected class had the highest output score ({pct(first[1])}); the next highest was {CLASSES[second[0]]} ({pct(second[1])}). Grades 2–4 are labeled referable by this project's simple rule.",fontsize=9.5,color="#55594f",wrap=True,transform=footer.transAxes)
    footer.text(0,.08,"Grad-CAM is a post-hoc visual explanation, not lesion detection or clinical reasoning. This experimental university project is not clinically validated and must not guide care.",fontsize=8.5,color="#666a60",wrap=True,transform=footer.transAxes)
    buffer=io.BytesIO()
    with PdfPages(buffer) as pdf:pdf.savefig(fig,bbox_inches="tight",facecolor="white")
    plt.close(fig);return buffer.getvalue()
@st.cache_resource(show_spinner="Loading EfficientNet-B0…")
def cached_model(path,modified):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,ck=load_model(path,device);return model,ck,device

metrics=load_json(REPORTS/"test_metrics.json")
bootstrap=load_json(REPORTS/"test_bootstrap_ci.json")
history=load_json(REPORTS/"training_history.json")
ready=CHECKPOINT.is_file()

with st.sidebar:
    st.markdown("### ◉ RetinaScreen");st.caption("University research demo");st.divider()
    st.markdown("A small study of retinal images and five-level DR grading.")
    st.caption("For learning and demonstration only. Not for medical decisions.")

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
            chart=base64.b64encode(training_plot(history)).decode("ascii")
            st.html(f'<img src="data:image/png;base64,{chart}" alt="Fixed-scale training and validation loss and accuracy curves" style="display:block;width:100%;height:auto;pointer-events:none;user-select:none;border-radius:8px;">')
            st.caption("Fixed-scale static charts · checkpoint selected by validation loss · validation contains 80 images.")
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
                        with st.status("Preparing the image analysis…",expanded=True) as progress:
                            try:
                                progress.write("Loading the trained EfficientNet-B0 checkpoint.")
                                net,ck,device=cached_model(str(CHECKPOINT.resolve()),CHECKPOINT.stat().st_mtime_ns)
                                progress.write("Calculating the five class scores and Grad-CAM map.")
                                start=time.perf_counter()
                                with st.spinner("The model is analyzing your image…"):out=predict_image(image,net,ck["config"],device)
                                st.session_state["screen_result"]={**out,"elapsed":time.perf_counter()-start,"filename":upload.name}
                                progress.update(label="Analysis complete · prediction and map are ready",state="complete",expanded=False)
                            except Exception as exc:
                                progress.update(label="Analysis could not be completed",state="error",expanded=True)
                                st.error(f"Inference failed: {exc}")
            with details:
                with st.container(border=True):
                    st.markdown('<div class="eyebrow">What happens next</div>',unsafe_allow_html=True)
                    st.markdown('<br><span class="step-dot">1</span><b>Classify</b><p class="small-note">The model estimates probabilities across five DR severity grades.</p><span class="step-dot">2</span><b>Summarize</b><p class="small-note">A simple grade 2–4 rule marks the prediction as referable for this demo.</p><span class="step-dot">3</span><b>Visualize</b><p class="small-note">Grad-CAM highlights image regions with greater influence on its selected class.</p><hr><span class="small-note">A sensitivity view is a model explanation, not evidence of a lesion. Results are uncalibrated and educational.</span>',unsafe_allow_html=True)
        elif not upload:
            st.markdown('<div class="upload-card"><div class="eyebrow">A simple three-step study</div><br><span class="step-dot">1</span>Choose a JPEG or PNG&nbsp;&nbsp; <span class="step-dot">2</span>Run the local model&nbsp;&nbsp; <span class="step-dot">3</span>Explore the prediction and map<br><br><span class="small-note">For best results, use a centered, well-lit fundus image at least 224 pixels on each side.</span></div>',unsafe_allow_html=True)
        if upload and result and result.get("filename")==upload.name:
            st.divider();title,download=st.columns([.72,.28],vertical_alignment="center")
            with title:st.markdown('<div class="eyebrow">Your model readout</div><h3>Prediction and visual explanation</h3>',unsafe_allow_html=True)
            with download:st.download_button("Download report (PDF)",data=screening_report(result),file_name="retinascreen_screening_report.pdf",mime="application/pdf",width="stretch")
            left,right=st.columns([.9,1.1],gap="large")
            with left:
                flag="Referable · grade 2–4" if result["referable"] else "Non-referable · grade 0–1"
                st.markdown(f'<div class="card"><div class="muted">Predicted severity</div><div class="grade">Grade {result["grade"]} · {result["class_name"]}</div><p><span class="pill">{flag}</span></p><span class="muted">Processed in {result["elapsed"]:.2f} s · uncalibrated probabilities</span></div>',unsafe_allow_html=True)
                ranked=sorted(enumerate(result["probabilities"]),key=lambda pair:pair[1],reverse=True)
                first,second=ranked[:2]
                st.markdown(f'<div class="why-card"><div class="eyebrow">Why this grade?</div><br>The classifier selected <b>{CLASSES[first[0]]}</b> because it had the highest model score ({pct(first[1])}). The next closest class was <b>{CLASSES[second[0]]}</b> ({pct(second[1])}). The referral badge follows the project rule: grades 2–4 are marked referable.<br><br><span class="muted">These scores are not calibrated certainty. The image map shows influential regions, not a list of detected lesions.</span></div>',unsafe_allow_html=True)
                st.markdown("#### Class probabilities")
                st.bar_chart(pd.DataFrame({"Probability":result["probabilities"]},index=[f"{i} · {n}" for i,n in enumerate(CLASSES)]),color="#86ad35",height=270)
            with right:
                st.markdown("#### What influenced this output?")
                st.caption("Grad-CAM · post-hoc visual explanation")
                st.image(result["gradcam"],caption="Red overlay marks regions with greater gradient influence on the selected class.",width=480)
                st.caption("This heatmap can help inspect the model's focus, but it does not prove highlighted areas are lesions or explain clinical reasoning.")

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
            st.markdown("#### Model at a glance")
            status="Trained checkpoint available" if ready else "Checkpoint missing · train before screening"
            st.markdown(f"**Status** · {status}  \n**Architecture** · ImageNet EfficientNet-B0 with five outputs.  \n**Input** · 224 × 224, ImageNet normalization.  \n**Training** · AdamW, inverse-square-root class weighting, seed 42, early stopping on validation loss.  \n**Explanation** · Grad-CAM at the final feature block.")
    with right:
        with st.container(border=True):
            st.markdown("#### Data and care")
            st.markdown("**Dataset** · Indian Diabetic Retinopathy Image Dataset (IDRiD), Disease Grading task.  \n**Source** · IEEE DataPort; IDRiD was introduced by Porwal and colleagues.  \n**Privacy** · Uploaded image bytes are used for local inference and are not saved by this UI.  \n**Limit** · Small test set, uncalibrated scores, no external validation.")
    st.warning("Academic demonstration only. Small test set and low recall for some grades. Not clinically validated; do not use for care.")
    st.markdown("#### References")
    st.markdown("- [IDRiD on IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)  \n- [IDRiD dataset details and grading task](https://idrid.grand-challenge.org/Data/)  \n- [IDRiD data descriptor · Porwal et al.](https://www.mdpi.com/2306-5729/3/3/25)")
