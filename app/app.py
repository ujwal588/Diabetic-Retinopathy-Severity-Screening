import streamlit as st
from src.pipeline import predict
from pathlib import Path
import tempfile
st.title("DR screening research demo")
st.warning("Academic prototype only; not a diagnosis.")
f=st.file_uploader("Fundus photo",type=["jpg","jpeg","png"])
if f:
    st.image(f)
    with tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix,delete=False) as t:
        t.write(f.getvalue()); path=t.name
    if st.button("Predict and generate Grad-CAM"):
        predict(path)
        cam=Path(__file__).resolve().parents[1]/"artifacts"/"gradcam"/(Path(path).stem+"_gradcam.png")
        st.image(str(cam),caption="Grad-CAM sensitivity visualization")
