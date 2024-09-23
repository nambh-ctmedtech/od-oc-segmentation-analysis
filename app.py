from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
import io
import numpy as np
import os
import cv2
import torch
import base64

from fundus_circle_cropping.fundus_cropping import fundus_image
from load_lwnet import load_models
from utils.get_loaders import get_test_dataset
from optic_disc_segmentation import *

#================================================================#

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8 = load_models()

@app.post("/object-to-image")
async def cropping_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_stream = io.BytesIO(image_bytes)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)

    cropped_image = np.array(fundus_image(image))

    cropped_path = "images/"
    if not os.path.isdir(cropped_path):
        os.makedirs(cropped_path)
    for f in os.listdir(cropped_path):
        if os.path.exists(cropped_path + f): os.remove(cropped_path + f)

    cv2.imwrite(f"{cropped_path}{file.filename}", cropped_image)
    
    test_loader = get_test_dataset(cropped_path)
    segmented_mask, mask_disc, mask_cup = prediction_eval(model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, test_loader, device)

    cdr = disc_cup_analysis(segmented_mask)
    isnt = ISNT(mask_cup, mask_disc, 'r')

    for f in os.listdir(cropped_path):
        if os.path.exists(cropped_path + f): os.remove(cropped_path + f)

    cv2.imwrite(cropped_path + f"seg_{file.filename}", segmented_mask)

    alpha_channel = segmented_mask[:, :, 3] / 255
    overlay_colors = segmented_mask[:, :, :3]

    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    h, w = segmented_mask.shape[:2]
    background_subsection = cropped_image[0:h, 0:w]

    composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

    cropped_image[0:h, 0:w] = composite

    _, encoded_image = cv2.imencode('.png', cropped_image)
    base64_image = base64.b64encode(encoded_image).decode("utf-8")

    results = {
        # "overlayed_image": f"data:image/png;base64,{base64_image}",
        "CDR_analysis": cdr,
        "ISNT_analysis": isnt
    }
    
    return JSONResponse(content=jsonable_encoder(results))
    
    # image_byte_array = io.BytesIO(encoded_image)

    # return StreamingResponse(image_byte_array, media_type="image/png")

#================================================================#

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)