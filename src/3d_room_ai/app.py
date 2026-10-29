import gradio as gr
from PIL import Image

from make_depth import make_depth

demo = gr.Interface(
    fn = make_depth,
    inputs=gr.Image(type="filepath", width=600, height=700),
    outputs= gr.Image(width=600, height=700),
    flagging_mode = "never"
)

demo.launch()