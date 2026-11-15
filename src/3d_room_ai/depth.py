from transformers import pipeline
from PIL import Image, ImageOps
import numpy as np
import json

estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
img = Image.open("D:/ML/project-oct/3d-room-ai/IKEA.jpeg")
result = estimator(img)

print(dir(estimator))

print(result['depth'])

result['depth'].save('ikea_depth.png')

# print(f'photo size: {img.size}')
# print(f'result: {result}')
# print(f'predicted depth: {result['predicted_depth']}')
# print(f'depth map shape: {result["predicted_depth"].shape}')

# print(f'smallest depth: {result["predicted_depth"].min()}')
# print(f'biggest depth: {result["predicted_depth"].max()}')


def to_threed(img): 
    newImg = ImageOps.exif_transpose(img)
    print(f'{img.size}')
    print(f'{newImg.size}')
    f = newImg.getexif().get_ifd(0x8769).get(0xA405)

    pixel_f = f / 43.27 * 5000 ##  It comes from the 36 × 24 mm frame: √(36² + 24²) = 43.27 and Calculated from your photo’s size: √(3000² + 4000²) = 5000

    ## divided by 4 so we only use the 1/4 value

    w = newImg.size[0] // 4
    h = newImg.size[1] // 4

    centre_w = w / 2
    centre_h = h / 2

    shrink_f = pixel_f/4
    # print(dir(img))

    print(f'{shrink_f} , {centre_w} ,  {centre_h}')

    small = newImg.resize((w, h))
    newResult = estimator(small)

    print(f'predicted depth: {newResult['predicted_depth']}')
    print(f'depth map shape: {newResult["predicted_depth"].shape}')

    print(f'{newResult}')

    # print (f'{dir(img.getexif())}')

    # Shrink to 1/4 size, then collect the values
    camera = {
        "width": w,
        "height": h,
        "fx": round(shrink_f, 2),
        "fy": round(shrink_f, 2),
        "cx": centre_w,
        "cy": centre_h,
    }
    pred_depth = newResult["predicted_depth"]

    x, y = 2245 // 4, 1863 // 4

    print(f'a thing: {pred_depth[x, y]} , type: {type(pred_depth[x, y])}')

    d_thing = pred_depth[y, x].item() # convert to Float

    Z_thing = 2.5 # jarak thing sesungguhnya dari camera
    scale = d_thing * Z_thing # convert pixel to meter

    print(f'Scale: {scale}')

    armchair_y, armchair_x = 1368 // 4, 2573 // 4

    armchair_d = pred_depth[armchair_y, armchair_x].item()
    Z = scale / armchair_d

    print(f'armchair: {armchair_d} , the distance: {Z}')


    X = (armchair_x - centre_w) * Z / round(shrink_f, 2)
    Y = (armchair_y - centre_h) * Z / round(shrink_f, 2)

    # lamp: 2323, 2374 px

    # gaming chair: 986, 2643 px


    def pixel_to_3d(u, v, d):
        Z = scale / d
        X = (u - centre_w) * Z / round(shrink_f, 2)
        Y = (v - centre_h) * Z / round(shrink_f, 2)
        return X, Y, Z


    print(f'{pixel_to_3d(x, y, d_thing)}')


    d = pred_depth.squeeze().cpu().numpy() # pytorch into numpy table of depth
    u, v = np.meshgrid(np.arange(w), np.arange(h)) # make the position of every pixel

    X,Y,Z = pixel_to_3d(u, v, d)

    print(f'X: {X[547, 353]}')

    colours = np.array(small.convert("RGB")).reshape(-1, 3)


    with open("room-ikea.ply", "w") as file:
        file.write(f"""ply
    format ascii 1.0
    element vertex {w * h}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    """)
        for x, y, z, (r, g, b) in zip(X.ravel(), Y.ravel(), Z.ravel(), colours):
            file.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

    # Write them to the file
    with open("camera.json", "w") as file:
        json.dump(camera, file, indent=2)


to_threed(img)