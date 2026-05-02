from diffusers import StableDiffusionPipeline
import torch
import cv2
from google.colab.patches import cv2_imshow

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to("cpu")

prompt = "A peaceful mountain sunrise with golden light reflecting on a lake, soft mist, highly detailed, 4K"

image = pipe(prompt).images[0]

image.save("sunrise.png")

print("Image generated")

img = cv2.imread('sunrise.png')
cv2_imshow(img)
