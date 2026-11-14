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
.stApp{background:radial-gradient(ellipse at 8% 5%,#edf6d9 0,transparent 38%),radial-gradient(ellipse at 94% 67%,#f1f5df 0,transparent 40%),linear-gradient(135deg,#fbfcf5 0%,#f5f8eb 100%);color:#30342c}
header[data-testid="stHeader"]{background:transparent!important;box-shadow:none!important}
header[data-testid="stHeader"] [data-testid="stToolbar"]{background:transparent!important}
.block-container{max-width:1280px;padding-top:2.1rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:rgba(250,252,244,.95);border-right:1px solid #e2e9d1}
[data-testid="stMetric"]{background:linear-gradient(145deg,#fff 55%,#fbfdf3);border:1px solid #e1e8d3;padding:.9rem 1rem;border-radius:14px;box-shadow:0 7px 22px #43542712}
[data-testid="stMetricLabel"] p{color:#70736a!important;font-size:.82rem}[data-testid="stMetricValue"]{color:#30342c;font-weight:700}
.stTabs [data-baseweb="tab-list"]{gap:.45rem;border-bottom:1px solid #e1e8d4}
.stTabs [aria-selected="true"]{color:#343a2e!important;border-bottom:3px solid #a6d834!important;background:#f2f7e4!important;border-radius:9px 9px 0 0}
.stButton>button{background:#b9e34c;color:#303820;border:1px solid #a9d335;border-radius:10px;font-weight:700;min-height:2.8rem;transition:transform .15s,box-shadow .15s}
.stButton>button:hover{background:#c8ed68;border-color:#a9d335;color:#303820;box-shadow:0 5px 14px #a6d83438;transform:translateY(-1px)}
[data-testid="stFileUploader"] section{border:1px dashed #b4d65e;background:#fff;border-radius:14px}
.hero{position:relative;overflow:hidden;background:linear-gradient(115deg,#fff 55%,#fbfdf3);border:1px solid #e5ebd8;border-left:7px solid #b9e34c;border-radius:20px;padding:1.2rem 1.7rem;margin:.25rem 0 .7rem;box-shadow:0 10px 30px #43542712}
.hero:after{content:"";position:absolute;width:160px;height:160px;border:20px solid #eff7d8;border-radius:50%;right:48px;top:-66px;pointer-events:none}
.hero:before{content:"✚";position:absolute;right:2.5rem;bottom:1rem;color:#dce9d4;font-size:1.65rem;pointer-events:none}
.hero .eyebrow{color:#718d2b;text-transform:uppercase;letter-spacing:.13em;font-weight:750;font-size:.72rem}
.hero h1{color:#30342c;margin:.35rem 0 .25rem;font-size:1.85rem}.hero p{color:#666a60;margin:0;max-width:780px}
.note{background:#f7faeb;border:1px solid #e8edda;border-left:4px solid #b9e34c;padding:.72rem 1rem;border-radius:9px;color:#494d42;margin:.6rem 0 1rem;box-shadow:0 5px 16px #4354270b}
.card{background:linear-gradient(145deg,#fff 58%,#fbfdf4);border:1px solid #e1e8d3;border-radius:16px;padding:1.15rem;box-shadow:0 8px 24px #43542712}.muted{color:#74776d;font-size:.88rem}
.why-card{background:linear-gradient(135deg,#f7faeb,#fff);border:1px solid #e5ebd7;border-radius:14px;padding:1rem;margin:.8rem 0;color:#494d42;line-height:1.55;box-shadow:0 7px 20px #43542710}
.guidance-card{background:linear-gradient(135deg,#fff,#fbfdf3);border:1px solid #e2e9d5;border-left:4px solid #a6d834;border-radius:12px;padding:.9rem 1rem;margin:.65rem 0;color:#494d42;line-height:1.55;font-size:.92rem;box-shadow:0 6px 18px #4354270d}
.grade{color:#30342c;font-size:1.6rem;font-weight:750}.pill{display:inline-block;padding:.3rem .65rem;border-radius:99px;font-weight:700;background:#f0f7d9;color:#5b741d}
.eyebrow{color:#748d31;text-transform:uppercase;letter-spacing:.12em;font-weight:750;font-size:.72rem}
.upload-card{background:linear-gradient(140deg,#fff,#f8fbed);border:1px solid #e1e8d3;border-radius:16px;padding:1.1rem 1.2rem;box-shadow:0 8px 22px #43542710}
[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.78);border-radius:16px;box-shadow:0 7px 22px #43542710}
.step-dot{display:inline-block;background:#eaf5c9;color:#617d1c;border-radius:50%;width:1.65rem;height:1.65rem;text-align:center;line-height:1.65rem;font-weight:750;margin-right:.45rem}
.small-note{color:#74776d;font-size:.82rem;line-height:1.5}
</style>
""",unsafe_allow_html=True)

def load_json(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError,json.JSONDecodeError,OSError):return None
def pct(x):return "—" if x is None else f"{100*x:.1f}%"
GRADE_GUIDE=(
    "No abnormalities visible on the clinical grading scale.",
    "Mild NPDR: microaneurysms only.",
    "Moderate NPDR: more findings than mild, below severe criteria.",
    "Severe NPDR criteria (4-2-1 rule), without proliferative changes.",
    "New abnormal vessels or bleeding inside or in front of the retina.",
)
def follow_up_note(grade):
    if grade>=4:
        return "This is the model's highest category and the demo flags it as referable. If this is a real image, use the result as a reason to arrange prompt eye-care review, not as a diagnosis. This unvalidated model cannot set urgency."
    if grade>=2:
        return "This demo places the image in its referable group. If this is a real image from someone with diabetes, arrange review by an eye-care professional. This model cannot diagnose, determine urgency, or choose a follow-up interval."
    return "This demo places the image in its non-referable group, which does not rule out diabetic retinopathy. People with diabetes should keep their clinician-recommended dilated eye exams; the National Eye Institute advises a comprehensive dilated exam at least yearly."
def draw_probability_axes(ax,probabilities,scale=1.0,show_title=True):
    indices=list(range(len(CLASSES)));labels=[f"{i} · {name}" for i,name in enumerate(CLASSES)]
    grade=max(indices,key=lambda i:probabilities[i])
    short_labels=[f"{i}\n{name}" for i,name in enumerate(CLASSES)]
    bars=ax.bar(indices,probabilities,color=["#a6d834" if i==grade else "#d8ddcb" for i in indices],width=.62)
    ax.set_xticks(indices,short_labels);ax.set_ylim(0,1.15);ax.set_yticks([0,.25,.5,.75,1.0],["0%","25%","50%","75%","100%"])
    if show_title:ax.set_title("Five-class model scores",loc="left",fontsize=13*scale,fontweight="bold",color="#30342c",pad=10*scale)
    ax.set_ylabel("Uncalibrated score",fontsize=10*scale,color="#55594f");ax.tick_params(axis="x",labelsize=9*scale,colors="#30342c",pad=5);ax.tick_params(axis="y",labelsize=9*scale,colors="#666a60")
    ax.grid(axis="y",color="#e8ebdf",linewidth=.8);ax.set_axisbelow(True);ax.spines[["top","right"]].set_visible(False)
    for bar,value in zip(bars,probabilities):ax.text(bar.get_x()+bar.get_width()/2,min(value+.025,1.08),f"{value:.1%}",va="bottom",ha="center",fontsize=9*scale,color="#45493f")
def probability_plot(probabilities):
    """Render the same fixed-scale vertical probability chart for UI and PDF."""
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.2,4.0),dpi=130);draw_probability_axes(ax,probabilities)
    fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format="png",dpi=160,bbox_inches="tight");plt.close(fig)
    return buffer.getvalue()
def report_heading(fig,y,symbol,title,x=.085):
    from matplotlib.patches import Circle
    fig.add_artist(Circle((x+.012,y-.007),.012,transform=fig.transFigure,facecolor="#f1f8db",edgecolor="#b7d956",linewidth=.8,zorder=2))
    fig.text(x+.012,y-.007,symbol,fontsize=8.5,weight="bold",color="#617a25",ha="center",va="center",zorder=3)
    fig.text(x+.035,y,title,fontsize=12.5,weight="bold",color="#30342c",va="top")
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
    """Create a clean, one-page educational PDF report in memory."""
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Circle,Ellipse,FancyBboxPatch
    fig=plt.figure(figsize=(8.27,11.69),facecolor="#fbfcf8")
    ink="#30382f";muted="#687166";lime="#a6d834"
    def panel(x,y,w,h,color,edge="#e8eddf"):
        fig.add_artist(FancyBboxPatch((x,y),w,h,transform=fig.transFigure,boxstyle="round,pad=.008,rounding_size=.018",facecolor=color,edgecolor=edge,linewidth=.8,zorder=-1))
    # Brand and title
    fig.add_artist(plt.Line2D([.075,.075],[.934,.978],color=lime,linewidth=5,solid_capstyle="round"))
    fig.add_artist(Ellipse((.098,.958),.028,.015,transform=fig.transFigure,facecolor="none",edgecolor="#829f32",linewidth=1.4))
    fig.add_artist(Circle((.098,.958),.004,transform=fig.transFigure,facecolor=lime,edgecolor="none"))
    fig.text(.122,.974,"RETINASCREEN  ·  UNIVERSITY RESEARCH DEMO",fontsize=9,color="#718d2b",weight="bold",va="top")
    fig.text(.122,.946,"Retinal image screening report",fontsize=19,color=ink,weight="bold",va="top")
    # Verdict receives the strongest visual emphasis.
    panel(.075,.805,.85,.105,"#eff7da",edge="#dcebb9")
    fig.text(.105,.887,"MODEL VERDICT",fontsize=8.5,color="#718d2b",weight="bold",va="top")
    fig.text(.105,.858,f"Grade {result['grade']}  ·  {result['class_name']}",fontsize=17,color=ink,weight="bold",va="top")
    flag="REFERABLE" if result["referable"] else "NON-REFERABLE"
    flag_color="#8b5c16" if result["referable"] else "#56731f"
    fig.text(.895,.855,flag,fontsize=9,color=flag_color,weight="bold",ha="right",va="top",bbox=dict(boxstyle="round,pad=.55",fc="#fff8df" if result["referable"] else "#f7faed",ec="none"))
    fig.text(.105,.823,"A model output for learning; it is not a diagnosis or a measure of urgency.",fontsize=9.5,color=muted,va="top")
    # Compact context and severity scale
    fig.text(.085,.782,"ABOUT THIS SCREEN",fontsize=8.5,color="#718d2b",weight="bold",va="top")
    intro="Diabetes can damage retinal blood vessels. Early changes may have no symptoms; screening photos do not replace an eye exam."
    fig.text(.085,.757,textwrap.fill(intro,95),fontsize=9.5,color=ink,va="top",linespacing=1.25)
    panel(.075,.642,.85,.074,"#f0f5fb",edge="#e2eaf2")
    fig.text(.095,.699,"FIVE-GRADE SCALE",fontsize=8,color="#627c93",weight="bold",va="top")
    names=["No DR","Mild","Moderate","Severe","Proliferative"]
    for i,name in enumerate(names):
        x=.105+i*.163
        fig.add_artist(Circle((x,.666),.010,transform=fig.transFigure,facecolor="#fff",edgecolor="#b7cbd8",linewidth=.8))
        fig.text(x,.666,str(i),fontsize=7.5,color="#587286",weight="bold",ha="center",va="center")
        fig.text(x+.017,.666,name,fontsize=8.2,color=ink,va="center")
    fig.text(.095,.648,"Dataset severity labels; not confirmed findings for this image.",fontsize=8.2,color=muted,va="center")
    # Fixed-scale chart in its own lightly tinted section.
    panel(.075,.363,.85,.252,"#ffffff",edge="#e6ebdf")
    fig.text(.095,.595,"MODEL SCORES",fontsize=8.5,color="#718d2b",weight="bold",va="top")
    fig.text(.095,.572,"Uncalibrated scores across the five classes",fontsize=9,color=muted,va="top")
    probability_ax=fig.add_axes([.13,.386,.75,.164]);draw_probability_axes(probability_ax,result["probabilities"],scale=.9,show_title=False)
    probability_ax.set_ylabel("");probability_ax.tick_params(axis="x",labelsize=8,pad=3);probability_ax.tick_params(axis="y",labelsize=8,pad=2)
    # Separate image explanation and human follow-up panels.
    panel(.075,.075,.385,.263,"#f0f7e4",edge="#e3edcf")
    panel(.515,.075,.410,.263,"#fff7e8",edge="#f0e6cf")
    fig.text(.098,.316,"IMAGE EXPLANATION",fontsize=8.5,color="#718d2b",weight="bold",va="top")
    fig.text(.098,.292,"Grad-CAM highlights model influence, not lesions.",fontsize=8.5,color=muted,va="top")
    im=result["gradcam"]; aspect=im.width/im.height
    box_w,box_h=.345,.185
    image_w=min(box_w,box_h*aspect);image_h=image_w/aspect
    cam=fig.add_axes([.10+(box_w-image_w)/2,.087+(box_h-image_h)/2,image_w,image_h])
    cam.imshow(im);cam.axis("off")
    fig.text(.538,.316,"WHAT TO DO WITH THIS",fontsize=8.5,color="#92713a",weight="bold",va="top")
    guidance=follow_up_note(int(result["grade"]))
    fig.text(.538,.287,textwrap.fill(guidance,47),fontsize=9.6,color=ink,va="top",linespacing=1.35)
    urgent="Sudden vision changes or new floaters need prompt professional attention, regardless of this result."
    fig.text(.538,.14,textwrap.fill(urgent,43),fontsize=8.7,color="#775d35",weight="bold",va="top",linespacing=1.25)
    # One quiet footer keeps the educational boundary visible without competing.
    fig.text(.085,.048,"Research prototype · not clinically validated · cannot diagnose or set urgency. Keep clinician-recommended dilated eye exams.",fontsize=8.1,color=muted,va="top")
    fig.text(.085,.030,"Educational references: National Eye Institute · American Academy of Ophthalmology DR Preferred Practice Pattern (2025)",fontsize=7.7,color="#858b80",va="top")
    buffer=io.BytesIO()
    with PdfPages(buffer) as pdf:pdf.savefig(fig,facecolor=fig.get_facecolor())
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

st.markdown("""<div class="hero"><div class="eyebrow">University research · IDRiD</div><h1>Looking closer, learning more.</h1><p>A five-grade retinal image study, with its results and model reasoning made easier to explore.</p></div><div class="note"><b>Research prototype.</b> Experimental results; not for clinical decisions.</div>""",unsafe_allow_html=True)
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
                probability=base64.b64encode(probability_plot(result["probabilities"])).decode("ascii")
                st.html(f'<img src="data:image/png;base64,{probability}" alt="Fixed-scale vertical scores for all five DR grades" style="display:block;width:100%;height:auto;pointer-events:none;user-select:none;border-radius:8px;">')
                st.markdown(f'<div class="guidance-card"><b>Professional follow-up context</b><br>{follow_up_note(int(result["grade"]))}</div>',unsafe_allow_html=True)
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
