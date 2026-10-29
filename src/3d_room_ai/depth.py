from transformers import pipeline
from PIL import Image

estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
img = Image.open("room.jpeg")
result = estimator(img)

print(dir(estimator))

print(result['depth'])

result['depth'].save('room_depth.png')

print(f'photo size: {img.size}')
print(f'result: {result}')
print(f'predicted depth: {result['predicted_depth']}')
print(f'depth map shape: {result["predicted_depth"].shape}')

print(f'smallest depth: {result["predicted_depth"].min()}')
print(f'biggest depth: {result["predicted_depth"].max()}')