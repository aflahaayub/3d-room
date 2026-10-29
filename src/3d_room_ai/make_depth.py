from transformers import pipeline

depth_est = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def make_depth(img):
    result = depth_est(img)
    return result["depth"]
